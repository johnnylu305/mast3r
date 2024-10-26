import sys
sys.path.append("/home/dsr/Documents/mad3d/IsaacLab")
import source
import os
import numpy as np
import glob
import torchvision.transforms as tvf
import cv2
import matplotlib.pyplot as plt
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
sys.path.append("/home/dsr/Documents/mad3d/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/single_drone")
from utils import OccupancyGrid, get_seen_face


import mast3r.utils.path_to_dust3r
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import _resize_pil_image
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.demo import get_3D_model_from_scene
from dust3r.image_pairs import make_pairs
from dust3r.viz import *



def drone_to_camera_pose(xyz, rpy):
    x, y, z = xyz
    roll, pitch, yaw = rpy
    
    rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)   
    #q = rotation.as_quat()

 
    # Local translation
    # camera is 5cm below and 5cm forward the drone
    local_translation = np.array([0.05, 0.0, -0.05])
    
    # Transform local translation to global frame
    global_translation = rotation.apply(local_translation)
    
    # New global position
    new_x = x + global_translation[0]
    new_y = y + global_translation[1]
    new_z = z + global_translation[2]
    
    return [new_x, new_y, new_z]


def get_new_poses(data, num_lines, txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # ensure we have enough lines to read
    if len(lines) >= num_lines:
        # read new lines
        for i in range(len(data), num_lines, 1):
            line = lines[i].strip()
            # ros coordinate
            # time1 time2 x y z roll pitch yaw camera_pitch 
            parts = line.split()
    
            if len(parts) == 9:
                xyz = [float(parts[2]), float(parts[3]), float(parts[4])]
                rpy = [float(parts[5]), float(parts[6]), float(parts[7])] 
                cp = float(parts[8])
                xyz = drone_to_camera_pose(xyz, rpy)
                rpy[1] += cp
                data.append(xyz+rpy)
        return True

    return False


def get_new_images(imgs, num_img, img_root):
    img_paths = sorted(glob.glob(os.path.join(img_root, "*.jpg")))
    
    # see new images
    if len(img_paths) >= num_img:
        # read new images
        for i in range(len(imgs), num_img, 1):
            img = Image.open(img_paths[i])
            img = np.array(img)
            imgs.append(img)
        return True

    return False


def load_images(imgs_org, size, square_ok=False, verbose=True):

    ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    imgs = []
    for img in imgs_org:
        img = Image.fromarray(img)

        W1, H1 = img.size
        
        # resize long side to 512
        img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size

        #plt.imshow(img)
        #plt.show()

        if verbose:
            print(f' - resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    return imgs


def run_duster(imgs, model=None):
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 30

    if model is None:
        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    images = load_images(imgs, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    # unknown initial pose
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)   
    #show_raw_pointcloud(scene.get_pts3d(), scene.imgs)
    
    # get local raw depth map
    depthmaps = scene.get_depthmaps(raw=True)
    
    # get virtual world poses
    vw_poses = scene.get_im_poses()

    return scene, depthmaps, vw_poses


def run_duster_with_known(imgs, E, I, model=None):
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    if model is None:
        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    images = load_images(imgs, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    # known initial pose
    n = len(imgs)
    T = torch.tensor([[-1, 0, 0], 
                      [0, -1, 0], 
                      [0,  0, 1]], dtype=torch.float32)  # Change of basis matrix
    E_rdf_rot = np.matmul(T, E[:, :3, :3])
    E_rdf_trans = np.matmul(T, E[:, :3, 3:4])
    E_rdf = np.concatenate((E_rdf_rot, E_rdf_trans), axis=-1)
    E_rdf = np.concatenate((E_rdf, E[:, 3:4, :]), axis=1)
    masks = [True for i in range(n)] 
    I *= 384./1080.
    poses = []
    intrinsic = []
    for i, m in enumerate(masks):
        if m:
            poses.append(E_rdf[i])
            intrinsic.append(I)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    #scene.preset_pose(poses, masks)
    #scene.preset_intrinsics(intrinsic, masks)
    #scene.preset_focal([I.diagonal()[:2].mean()])
    #scene.preset_principal_point([I[:2, 2]])
    loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)   
    show_raw_pointcloud(scene.get_pts3d(), scene.imgs)
    
    # get local raw depth map
    depthmaps = scene.get_depthmaps(raw=True)
    
    # get virtual world poses
    vw_poses = scene.get_im_poses()

    return scene, depthmaps, vw_poses


def set_axes_equal(ax):
    """
    Set equal scaling on all axes of a 3D plot.
    This ensures that the x, y, and z axes are equally spaced.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = np.max([x_range, y_range, z_range])

    mid_x = (x_limits[0] + x_limits[1]) * 0.5
    mid_y = (y_limits[0] + y_limits[1]) * 0.5
    mid_z = (z_limits[0] + z_limits[1]) * 0.5

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)


def visualize_points_with_labels(duster_P, ros_P):
    """
    Function to visualize two sets of 3D points with equal axis scaling and point labels.
    
    Parameters:
    - duster_P: List of 3D points (first set, shown in red).
    - ros_P: List of 3D points (second set, shown in blue).
    """
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot duster_P points in red
    ax.scatter(duster_P[:, 0], duster_P[:, 1], duster_P[:, 2], c='r', label='Duster P', marker='o')

    # Plot ros_P points in blue
    ax.scatter(ros_P[:, 0], ros_P[:, 1], ros_P[:, 2], c='b', label='ROS P', marker='^')

    # Add labels to the points
    for i in range(len(duster_P)):
        ax.text(duster_P[i, 0], duster_P[i, 1], duster_P[i, 2], '%d' % (i+1), color='red')

    for i in range(len(ros_P)):
        ax.text(ros_P[i, 0], ros_P[i, 1], ros_P[i, 2], '%d' % (i+1), color='blue')

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Duster P (red) vs ROS P (blue) with labels')

    # Show legend
    ax.legend()

    # Set equal scaling on the axes
    set_axes_equal(ax)

    # Display the plot
    plt.show()


def umeyama(X, Y):
    """
    Copyright: Carlo Nicolini, 2013
    Code adapted from the Mark Paskin Matlab version from
    http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m
    See paper by Umeyama (1991)
    """

    X = X.T
    Y = Y.T

    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)

    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc * Xc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U, D, V = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = V.T.copy()

    r = np.linalg.matrix_rank(Sxy)
    S = np.eye(m)

    if r < (m - 1):
        raise ValueError('not enough independent measurements')

    if (np.linalg.det(Sxy) < 0):
        S[-1, -1] = -1
    elif (r == m - 1):
        if (np.linalg.det(U) * np.linalg.det(V) < 0):
            S[-1, -1] = -1

    R = np.dot(np.dot(U, S), V.T)
    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return c, R, t


def rdf_to_ruf(pose_matrices):
    converted_poses = []
    
    for pose in pose_matrices:
        # Extract rotation matrix (R) and translation vector (T)
        rotation_matrix = pose[:3, :3]
        translation_vector = pose[:3, 3]
        
        # Apply coordinate transformation: 
        # Convert from right-down-forward to right-up-forward
        # This involves flipping the Y axis.
        flip_y = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, 1]])

        # Update rotation matrix and translation vector
        rotation_matrix = flip_y @ rotation_matrix
        translation_vector = flip_y @ translation_vector

        # Convert rotation matrix to a rotation vector (Rodrigues' rotation vector)
        r = R.from_matrix(rotation_matrix)
        rotation_vector = r.as_rotvec()  # This gives the axis-angle rotation vector

        # Normalize the rotation vector to get a unit vector
        rotation_unit_vector = rotation_vector / np.linalg.norm(rotation_vector)

        # Extract translation (x, y, z)
        x, y, z = translation_vector

        # Append the result as [x, y, z, rotation_unit_vector]
        converted_poses.append([x, y, z]+rotation_unit_vector.tolist())
    
    return converted_poses


def to_rotation_unit_vector(roll, yaw, pitch):
    # Step 1: Convert roll, yaw, pitch from degrees to radians
    roll_rad = np.radians(roll)
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    
    # Step 2: Create a rotation matrix from the Euler angles with 'xyz' convention (roll, yaw, pitch)
    r = R.from_euler('xyz', [roll_rad, yaw_rad, pitch_rad])
    
    # Step 3: Convert the rotation matrix to a rotation vector
    rotation_vector = r.as_rotvec()
    
    # Step 4: Normalize the rotation vector to get the unit vector
    rotation_unit_vector = rotation_vector / np.linalg.norm(rotation_vector)
    
    return rotation_unit_vector


def rdf_to_flu(pose):
    T_conv = np.array([
        [0,   0,  1,  0],
        [-1,  0,  0,  0],
        [0,  -1,  0,  0],
        [0,   0,  0,  1]   
    ])
    
    # Apply the conversion matrix to the input transformation matrix
    new_pose = T_conv @ pose
    
    return new_pose


def rpy_to_rotation_matrix(poses):
    transformation_matrices = []
    
    for pose in poses:
        x, y, z, roll, pitch, yaw = pose
        
        # Create a rotation matrix from roll, pitch, yaw
        rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        rotation_matrix = rotation.as_matrix()
        
        # Create a homogeneous transformation matrix [R|T]
        transformation_matrix = np.eye(4)  # Start with identity matrix
        transformation_matrix[:3, :3] = rotation_matrix  # Set rotation part
        transformation_matrix[:3, 3] = [x, y, z]         # Set translation part
        
        # Append to the list of transformation matrices
        transformation_matrices.append(transformation_matrix)
    
    return transformation_matrices


def visualize_3d_points(pts3d):
    # Create a new figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract X, Y, Z coordinates from the list of 3D points
    pts3d = pts3d.reshape(-1, 3)[::50]
    
    xs = [p[0] for p in pts3d]
    ys = [p[1] for p in pts3d]
    zs = [p[2] for p in pts3d]
    
    # Plot the points using scatter
    ax.scatter(xs, ys, zs, c='r', marker='o')
    
    # Set axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    # Set title
    ax.set_title("3D Points Visualization")
    
    # Show the plot
    plt.show()


def visualize_valid_3d_points(pts3d, mask):
    """
    Visualize 3D points using a provided mask to filter valid points.
    
    Parameters:
    - pts3d: Array of 3D points with shape (h, w, 3) or equivalent.
    - mask: Boolean array of the same shape as (h, w) indicating valid points.
    """
    # Reshape the 3D points and the mask to align
    pts3d = pts3d.reshape(-1, 3)
    mask = mask.reshape(-1)

    # Filter the 3D points based on the mask (only keep valid points)
    valid_pts3d = pts3d[mask][::50]
    
    # Extract X, Y, Z coordinates from the valid 3D points
    xs = valid_pts3d[:, 0]
    ys = valid_pts3d[:, 1]
    zs = valid_pts3d[:, 2]
    
    # Create a new figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the valid points using scatter
    ax.scatter(xs, ys, zs, c='r', marker='o')
    
    # Set axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    # Set title
    ax.set_title("3D Points Visualization")

    set_axes_equal(ax)  

    # Show the plot
    plt.show()


def transform_to_depth_map(world_pts, pose):
    n, h, w, _ = world_pts.shape  # Get the dimensions of the input world points
    
    depth_maps = np.empty((n, h, w))  # Placeholder for storing the depth maps for each batch

    # Loop over each frame in the batch
    for i in range(n):
        # Extract the rotation and translation from the pose (C2W) for frame i
        R = pose[i, :3, :3]  # Rotation matrix (3x3)
        T = pose[i, :3, 3]   # Translation vector (3,)

        # Invert the C2W pose matrix to get W2C
        R_inv = R.T            # Inverted (transposed) rotation matrix
        T_inv = -R_inv @ T     # Inverted translation vector

        # Add homogeneous coordinates to world points if needed
        if world_pts.shape[-1] == 3:
            world_pts_homogeneous = np.concatenate([world_pts[i], np.ones((h, w, 1))], axis=-1)  # Add homogeneous coordinate
        else:
            world_pts_homogeneous = world_pts[i]  # Already has homogeneous coordinates

        world_pts_homogeneous = world_pts_homogeneous.reshape(-1, 4)  # Flatten to (h*w, 4)

        # Construct the W2C transformation matrix for frame i
        pose_inv = np.eye(4)
        pose_inv[:3, :3] = R_inv  # Set the inverse rotation
        pose_inv[:3, 3] = T_inv   # Set the inverse translation

        # Apply the W2C transformation to the world points
        local_pts_homogeneous = (pose_inv @ world_pts_homogeneous.T).T  # Transform to local coordinates
        local_pts = local_pts_homogeneous[:, :3]  # Extract (x, y, z)

        local_pts = local_pts.reshape(h, w, 3)  # Reshape back to (h, w, 3)

        # Convert to depth map (z-values in the local camera frame)
        depth_map = local_pts[..., 2]  # Extract the z-values as the depth map

        # Store the depth map for the current frame
        depth_maps[i] = depth_map

    return depth_maps


def transform_to_pts_map(world_pts, pose):
    n, h, w, _ = world_pts.shape  # Get the dimensions of the input world points
    
    local_ptss = np.empty((n, h, w, 3))  # Placeholder for storing the depth maps for each batch

    # Loop over each frame in the batch
    for i in range(n):
        # Extract the rotation and translation from the pose (C2W) for frame i
        R = pose[i, :3, :3]  # Rotation matrix (3x3)
        T = pose[i, :3, 3]   # Translation vector (3,)

        # Invert the C2W pose matrix to get W2C
        R_inv = R.T            # Inverted (transposed) rotation matrix
        T_inv = -R_inv @ T     # Inverted translation vector

        # Add homogeneous coordinates to world points if needed
        if world_pts.shape[-1] == 3:
            world_pts_homogeneous = np.concatenate([world_pts[i], np.ones((h, w, 1))], axis=-1)  # Add homogeneous coordinate
        else:
            world_pts_homogeneous = world_pts[i]  # Already has homogeneous coordinates

        world_pts_homogeneous = world_pts_homogeneous.reshape(-1, 4)  # Flatten to (h*w, 4)

        # Construct the W2C transformation matrix for frame i
        pose_inv = np.eye(4)
        pose_inv[:3, :3] = R_inv  # Set the inverse rotation
        pose_inv[:3, 3] = T_inv   # Set the inverse translation

        # Apply the W2C transformation to the world points
        local_pts_homogeneous = (pose_inv @ world_pts_homogeneous.T).T  # Transform to local coordinates
        local_pts = local_pts_homogeneous[:, :3]  # Extract (x, y, z)

        local_pts = local_pts.reshape(h, w, 3)  # Reshape back to (h, w, 3)

        local_ptss[i] = local_pts

    return local_ptss


def apply_cRT_pts(pts, c, R, T):
    # (n, h, w, 3)
    n, h, w, _ = pts.shape
    pts_reshaped = pts.reshape(-1, 3)  

    # rotate
    rotated_pts = R @ pts_reshaped.T

    # scale
    scaled_pts = c * rotated_pts

    # translation
    translated_pts = (scaled_pts + T[:, None]).T

    # reshape
    transformed_pts = translated_pts.reshape(-1, h, w, 3)

    return transformed_pts


def apply_cRT_to_E(E, c, R, T):
    n = E.shape[0]

    # rotation and translation
    R_current = E[:, :3, :3]
    T_current = E[:, :3, 3]

    # rotate to correct camera orientation
    R_prime = R@R_current

    # scale, rotate, translate to correct camera position
    T_prime = c * (R @ T_current.T).T + T

    E_prime = np.empty_like(E)  
    E_prime[:, :3, :3] = R_prime
    E_prime[:, :3, 3] = T_prime
    E_prime[:, 3, :] = [0, 0, 0, 1]

    return E_prime


def get_rescaled_depths(imgs, E, model, I=None):
    """
    Input
    E: (#imgs, 4, 4) in forward, left, up
    imgs: (#imgs, h, w, 3)
    Output
    scaled_pts: (#imgs, 384, 512, 3)
    masks: (#imgs, 384, 512)
    duster_rw_poses: (#imgs, 4, 4)
    scene.imgs: (#imgs, 384, 512, 3)
    """  

    #print(imgs.shape)
    #print(E.shape) 
    
    assert len(imgs)==len(E) 

    # run duster
    scene, duster_depthmaps, duster_vw_poses = run_duster(imgs, model)
    #scene, duster_depthmaps, duster_vw_poses = run_duster_with_known(imgs, E, I)
    # (#imgs, 384*512)
    duster_depthmaps = duster_depthmaps.detach()
    # (#imgs, 4, 4) in right, down, forward
    duster_vw_poses = duster_vw_poses.detach().cpu().numpy()
    #print(duster_depthmaps.shape)
    #print(duster_vw_poses.shape)

    # find C (scaling): 1, R (rotation): (3, 3), T (translation): (3,) from duster to real world
    c, R, T = umeyama(duster_vw_poses[:, :3, 3], E[:, :3, 3])
    #print(c)
    #print(R.shape)
    #print(T.shape)

    # get point map in duster world frame
    pts3d = scene.get_pts3d_custom(scene.depth_to_pts3d_custom(duster_depthmaps, torch.tensor(duster_vw_poses).cuda()))

    pts3d = torch.stack(pts3d).detach().cpu().numpy()
    # transform point map from duster world frame to real world frame
    pts3d = apply_cRT_pts(pts3d, c, R, T)
    visualize_valid_3d_points(pts3d[0], scene.get_masks()[0].detach().cpu().numpy())

    # transform extrinsic from duster to real world frame
    duster_rw_poses = apply_cRT_to_E(duster_vw_poses, c, R, T)
    #visualize_points_with_labels(duster_rw_poses[:, :3, 3], E[:, :3, 3])

    
    # get real world local depth (z) : (#imgs, 384, 512)
    scaled_depth = transform_to_depth_map(pts3d, duster_rw_poses)
    #plt.imshow(scaled_depth[0])
    #plt.show()
    #print(scaled_depth.shape)
    
    # get real world local point cloud: (#imgs, 384, 512, 3)
    scaled_pts = transform_to_pts_map(pts3d, duster_rw_poses)
    #print(scaled_pts.shape)
    # (#imgs, 384, 512)
    masks = torch.stack(scene.get_masks()).detach().cpu().numpy()
    #print(masks.shape)
    #print(duster_rw_poses.shape)

    #pts3d = scene.get_pts3d_custom(scene.depth_to_pts3d_custom(torch.tensor(scaled_pts[:, :, :, 2]).cuda().reshape(-1, 384*512), torch.tensor(duster_rw_poses).cuda()))
    #show_raw_pointcloud_with_cams(scene.imgs, pts3d, [m.cpu() for m in scene.get_masks()],
    #                              scene.get_focals(), duster_rw_poses,
    #                              point_size=2, cam_size=0.15, cam_color=None)
    
    return scaled_pts, masks, duster_rw_poses, np.array(scene.imgs)


def visualize_world_point_cloud(local_pts3d, masks, duster_poses, duster_imgs):
    # Initialize an empty list to collect all world points and colors
    world_points = []
    colors = []

    # Iterate over each frame
    for i in range(local_pts3d.shape[0]):
        # Extract the mask, pose, local points, and colors for the current frame
        mask = masks[i]  # Shape (384, 512)
        pose = duster_poses[i]  # Shape (4, 4)
        pts3d = local_pts3d[i].reshape(-1, 3)  # Reshape to (384*512, 3)
        img_colors = duster_imgs[i].reshape(-1, 3)  # Flatten colors for all pixels in the frame

        # Filter points and colors based on the mask
        valid_pts3d = pts3d[mask.flatten() > 0]
        valid_colors = img_colors[mask.flatten() > 0]

        # Transform local points to world points
        valid_pts3d_h = np.concatenate([valid_pts3d, np.ones((valid_pts3d.shape[0], 1))], axis=1)  # Add homogeneous coord.
        world_pts3d = (pose @ valid_pts3d_h.T).T[:, :3]  # Transform to world frame

        # Store world points and colors
        world_points.append(world_pts3d)
        colors.append(valid_colors)  # Colors as uint8 in RGB for trimesh

    # Concatenate all points and colors from all frames
    world_points = np.concatenate(world_points, axis=0)
    colors = np.concatenate(colors, axis=0) #/ 255.0

    # Create the point cloud in Trimesh
    cloud = trimesh.PointCloud(world_points, colors=colors)
    
    # Display the point cloud
    cloud.show()


def visualize_occ_grid(occ_grid, voxel_size):
    """
    Visualizes the occupancy grid using Trimesh.
    
    Parameters:
    - occ_grid: A 3D numpy array where 1 indicates occupied cells and 0 indicates empty cells.
    """
    # Create an empty list to store voxel meshes
    voxel_meshes = []

    # Loop through the occupancy grid and create a cube for each occupied cell
    for i in range(occ_grid.shape[0]):
        for j in range(occ_grid.shape[1]):
            for k in range(occ_grid.shape[2]):
                if occ_grid[i, j, k] == 1:  # If the cell is occupied
                    # Create a cube (voxel) at position (i, j, k)
                    voxel = trimesh.creation.box(extents=(voxel_size, voxel_size, voxel_size))
                    # Translate the voxel to its correct position in space
                    voxel.apply_translation([i * voxel_size, j * voxel_size, k * voxel_size])
                    # Append to the list of voxel meshes
                    voxel_meshes.append(voxel)

    # Combine all the voxel meshes into a single mesh
    scene = trimesh.util.concatenate(voxel_meshes)

    # Visualize the scene
    scene.show()


def get_occ_and_face(grid, obv_face, points_3d_cam, poses, masks, env_size, grid_size, device):
    # local 3d point map: (batch, h*w, 3)
    points_3d_cam = points_3d_cam[masks].reshape(1, -1, 3)   
    # world 3d point map: (batch, h*w, 3)
    ones = torch.ones((1, points_3d_cam.shape[1], 1), device=device)
    points_3d_cam = torch.cat((points_3d_cam, ones), dim=-1)  # (1, h*w, 4)
    points_3d_world = torch.matmul(points_3d_cam, poses.T)  # (1, h*w, 4)
    points_3d_world = points_3d_world[..., :3] / points_3d_world[..., 3:4]
    # add a small offset to z to prevent noise removing floor
    points_3d_world[0, :, 2] += env_size/(grid_size)*0.9
    

    mask_x = points_3d_world[0, :, 0].abs() < env_size/2-1e-3
    mask_y = points_3d_world[0, :, 1].abs() < env_size/2-1e-3
    mask_z = (points_3d_world[0, :, 2] < env_size-1e-3) & (points_3d_world[0,:,2]>=0) 

    # Combine masks to keep rows where both xyz are within the range
    mask = mask_x & mask_y & mask_z
            
    offset = torch.tensor([env_size/2, env_size/2, 0]).to(device)

    if points_3d_world[0][mask].shape[0] > 0:
        pos_w = poses[:3, 3]
        ratio = grid_size/env_size
        grid.trace_path_and_update(0, torch.floor(pos_w+offset)*ratio, 
                                   torch.floor((points_3d_world[0]+offset))*ratio)
        grid.update_log_odds(0, torch.floor((points_3d_world[0][mask]+offset)*ratio),
                             occupied=True)
        new_face = get_seen_face(torch.unique(
                    torch.floor((points_3d_world[0][mask]+offset)*ratio).int(), dim=0), 
                    torch.floor((pos_w+offset)*ratio), grid_size, device)
        obv_face[0] = torch.logical_or(obv_face[0], new_face)

    
    probability_grid = grid.log_odds_to_prob(grid.grid)

    return probability_grid, obv_face
    


def get_nbv(local_pts3d, masks, duster_poses, duster_imgs, model):
    # get observation
    # imgs: (batch, 2*3, 300, 300), 
    # poses: (batch, 50*5+1) in ros (flu), 
    # grid: (batch, 10, 20, 20, 20) occ,x,y,z,face

    # assume env size is 3m x 3m x 3m
    env_size = 3
    grid_size = 20
    device = 'cuda'
    decrement = 0.01
    increment = 1.0
    max_log_odds = 10.
    min_log_odds = -10.

    # images alreadly normalized
    obv_imgs = np.zeros((1, 2, 300, 300, 3))
    obv_imgs[0, 0] = np.array(cv2.resize(duster_imgs[-1], (300, 300)))
    if len(duster_imgs) >= 2:
        obv_imgs[0, 1] = np.array(cv2.resize(duster_imgs[-2], (300, 300)))

    # poses
    obv_poses = np.zeros((1, 50, 5))
    obv_poses[0, :len(duster_poses), :3] = duster_poses[:, :3, 3]
    # TODO: rethinking about this
    obv_poses[0, :len(duster_poses), :3] /= env_size
    # convert rotation matrices to roll, pitch, yaw in radians
    r = R.from_matrix(duster_poses[:, :3, :3])
    rpy = r.as_euler('xyz', degrees=False)
    obv_poses[0, :len(duster_poses), 3:] = rpy[:, 1:]
    obv_poses[0, :len(duster_poses), 3:] /= 3.15
    obv_poses = np.concatenate([obv_poses.reshape(1, -1), (np.array([[len(duster_poses)]])/50)], axis=1)
    #print(obv_poses)

    # grid
    # occ
    obv_occ = np.zeros((1, 20, 20, 20, 10))
    obv_occ[0, :, :, :, 0] += 0.5
    # xyz
    x_coords = np.linspace(-env_size/2.0, env_size/2.0, grid_size)
    y_coords = np.linspace(-env_size/2.0, env_size/2.0, grid_size)
    z_coords = np.linspace(0.0, env_size, grid_size)
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    obv_occ[0, :, :, :, 1:4] = np.stack((x_mesh, y_mesh, z_mesh), axis=-1)/env_size
    # occ grid and face
    grid_xyz_size = (1, grid_size, grid_size, grid_size)
    grid = OccupancyGrid(env_size, grid_xyz_size, decrement, increment, 
                         max_log_odds, min_log_odds, device)
    grid.grid[0] = 0
    face = torch.zeros(1, grid_size, grid_size, grid_size, 6, device=device)
    for i in range(len(local_pts3d)):
        occ, face = get_occ_and_face(grid, face, torch.tensor(local_pts3d[i]).to(device).float(), 
                                     torch.tensor(duster_poses[i]).to(device).float(), 
                                     torch.tensor(masks[i]).to(device), env_size, grid_size, device)
        # visualize
        hard_occ = torch.where(occ[0, :, :, :] >= 0.6, 1, 0)
        #visualize_occ_grid(hard_occ, env_size/grid_size)

    obv_occ[0, :, :, :, 0] = occ.cpu().numpy()
    obv_occ[0, :, :, :, 4:] = face.cpu().numpy()

    obv = {"pose_step": obv_poses,
           "img": np.transpose(obv_imgs, (0, 1, 4, 2, 3)).reshape(-1, 2 * 3, 300, 300),
           "occ": np.transpose(obv_occ, (0, 4, 1, 2, 3))}

    actions, _ = model.predict(obv)

    xyz = actions[:, :3]
    xyz = (xyz + np.array([0., 0., 1.])) * np.array([env_size/2.0 - 0.5, env_size/2.0 - 0.5, env_size/4.0])
    yaw = actions[:, 3:4] * np.pi
    pitch = (actions[:, 4:5] + 1/5.) / 2. * np.pi * 5/6.

    return actions

def main():
    img_root = os.path.join(os.sep, "home", "dsr", "Documents", "mad3d", "mast3r", "dataset", "opera_house_marker_40d")
    
    # text path
    txt_file = os.path.join(img_root, "transform_record.txt")    

    # images
    imgs = []

    # poses
    poses = []


    # initial duster
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    duster_model = AsymmetricMASt3R.from_pretrained(model_name).cuda()

    # initial rl model
    model_name = os.path.join(os.sep, "home", "dsr", "Documents", "mad3d", "IsaacLab", "logs", "sb3", "Isaac-Quadcopter-Direct-v1", "camera_image_face_rand", "model_13824000_steps")
    nbv_model = PPO.load(model_name)

    I = np.array([[986.78, 0, 721.19], [0, 964.98, 547.47], [0, 0, 0]])

    for i in range(3, 10, 1):
        # initial images (at least 2)
        while not get_new_images(imgs, i, img_root):
            print(f"waiting for images")
        print(f"Successfully read new images")

        # corresponding poses
        while not get_new_poses(poses, i, txt_file):
            print(f"waiting for poses")       
        print(f"Successfully read new poses")


        poses_trans = np.array(rpy_to_rotation_matrix(poses))
        # coordinate system is forward, left, up
        local_pts3d, masks, duster_poses, duster_imgs = get_rescaled_depths(np.array(imgs), poses_trans, duster_model, I)
        
        destination = get_nbv(local_pts3d, masks, duster_poses, duster_imgs, nbv_model)
        #exit()
        #if i%3==0:
        #    visualize_world_point_cloud(local_pts3d, masks, duster_poses, duster_imgs)



if __name__ == '__main__':
    main()
