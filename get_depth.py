import sys
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import _resize_pil_image
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.demo import get_3D_model_from_scene
from dust3r.image_pairs import make_pairs
from dust3r.viz import *


sys.path.append("/home/dsr/Documents/mad3d/Metric3D")
from infer_depth import run_metric3d

import os
import numpy as np
import glob
import torchvision.transforms as tvf
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from scipy.spatial.transform import Rotation as R


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


def resize_depth_image(depth, size, square_ok=False, verbose=True):
    # Get original dimensions
    H1, W1 = depth.shape
    
    # Resize the longer side to 'size', maintaining aspect ratio
    if W1 > H1:
        new_W = size
        new_H = int(H1 * (size / W1))
    else:
        new_H = size
        new_W = int(W1 * (size / H1))

    # Create a square zero-padded array if square_ok is True
    if square_ok:
        depth_resized = np.zeros((size, size), dtype=depth.dtype)  # square canvas
        resized_depth = cv2.resize(depth, (new_W, new_H))  # resize keeping aspect ratio
        depth_resized[:new_H, :new_W] = resized_depth  # place resized depth in the top-left corner
    else:
        # Simply resize without padding if square_ok is False
        depth_resized = cv2.resize(depth, (new_W, new_H))
    
    # Updated dimensions after resizing
    W, H = depth_resized.shape[1], depth_resized.shape[0]
    cx, cy = W // 2, H // 2

    # Calculate cropping dimensions
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if not square_ok and W == H:
        halfh = 3 * halfw // 4

    # Crop the center
    cropped_depth = depth_resized[cy - halfh: cy + halfh, cx - halfw: cx + halfw]

    # Get final cropped dimensions
    W2, H2 = cropped_depth.shape[1], cropped_depth.shape[0]

    if verbose:
        print(f' - resolution {W1}x{H1} --> {W2}x{H2}')

    return cropped_depth


def run_duster(imgs):
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

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

   




def main():
    # image root path
    img_root = os.path.join(os.sep, "home", "dsr", "Documents", "mad3d", "mast3r", "dataset", "opera_house_marker_40d")
    
    # text path
    txt_file = os.path.join(img_root, "transform_record.txt")    

    # images
    imgs = []

    # poses
    poses = []

    # initial images (at least 2)
    while not get_new_images(imgs, 3, img_root):
        print(f"waiting for images")
    print(f"Successfully read new images")

    # corresponding poses
    while not get_new_poses(poses, 3, txt_file):
        print(f"waiting for poses")       
    print(f"Successfully read new poses")

    # run dust3r inference to get:
    # virtual world K, P, local pts map
    scene, duster_depthmaps, duster_vw_poses = run_duster(imgs)

    # show original pcd
    pts3d = scene.get_pts3d_custom(scene.depth_to_pts3d_custom(duster_depthmaps, duster_vw_poses))
    #show_raw_pointcloud(pts3d, scene.imgs)
    show_raw_pointcloud_with_cams(scene.imgs, scene.get_pts3d(), [m.cpu() for m in scene.get_masks()],
                                  scene.get_focals(), scene.get_im_poses(),
                                  point_size=2, cam_size=0.3, cam_color=None)

    # get metric depth 
    intrinsic = [2840, 2842, 2006, 1524]
    data_basic=dict(
        canonical_space = dict(
            # img_size=(540, 960),
            focal_length=2841,
        ),
        depth_range=(0, 1),
        depth_normalize=(0.1, 200),
        crop_size = (616, 1064),  # %28 = 0
        clip_depth_range=(0.1, 200),
        vit_size=(616,1064)
    )
    metric_depth = run_metric3d(imgs[0], model="metric3d_vit_large", intrinsic=intrinsic, data_basic=data_basic)

    # find scale
    h, w = scene.imshapes[0]
    duster_depthmap = duster_depthmaps[0].detach().reshape(h, w).cpu().numpy()
    metric_depth_res = resize_depth_image(metric_depth, size=512)

    mask = metric_depth_res != 0
    ratio = (metric_depth_res[mask]/duster_depthmap[mask]).reshape(-1)
    median_value = np.median(ratio)
    sorted_indices = np.argsort(np.abs(ratio - median_value))
    n = 100  
    scale = np.mean(ratio[sorted_indices[:n]])

    print(f"Mono depth scale {scale}")

    #plt.subplot(1, 3, 1)
    #plt.imshow(duster_depthmap)
    #plt.subplot(1, 3, 2)
    #plt.imshow(metric_depth_res)
    #plt.subplot(1, 3, 3)
    #plt.imshow(duster_depthmap*scale)
    #plt.show()

    # virtual world (right, up, forward) to real world (ros: forward, left, up)
    duster_vw_flu_poses = rdf_to_flu(duster_vw_poses.detach().cpu().numpy())
    poses_trans = np.array(rpy_to_rotation_matrix(poses))
    #print(duster_vw_flu_poses.shape)
    #print(poses_trans.shape)

    #c, R, T = umeyama(duster_vw_flu_poses[:, :3, 3], poses_trans[:, :3, 3])

    c, R, T = umeyama(duster_vw_poses[:, :3, 3].detach().cpu().numpy(), poses_trans[:, :3, 3])

    pts3d = scene.get_pts3d_custom(scene.depth_to_pts3d_custom(duster_depthmaps, torch.tensor(duster_vw_poses).cuda()))

    pts3d = torch.stack(pts3d).detach().cpu().numpy()
    pts3d = apply_cRT_pts(pts3d, c, R, T)
    #visualize_valid_3d_points(pts3d[0], scene.get_masks()[0].detach().cpu().numpy())
    duster_rw_poses = apply_cRT_to_E(duster_vw_poses.detach().cpu().numpy(), c, R, T)
    #visualize_points_with_labels(duster_rw_poses[:, :3, 3], poses_trans[:, :3, 3])
    
    scaled_depth = transform_to_depth_map(pts3d, duster_rw_poses)

    plt.imshow(scaled_depth[0])
    plt.show()

    """
    print(R.shape, T.shape)
    print(duster_vw_flu_poses[:, :3, 3].shape)

    print("c, R, T", c, R, T)    
    print("duster P", (c*R@duster_vw_flu_poses[:, :3, 3].T + T[:, None]).T)
    print("ros P", poses_trans[:, :3, 3])


    visualize_points_with_labels((c*R@duster_vw_poses[:, :3, 3].detach().cpu().numpy().T + T[:, None]).T, poses_trans[:, :3, 3])

    pts3d = scene.get_pts3d()

    #pts3d = scene.get_pts3d_custom(scene.depth_to_pts3d_custom(duster_depthmaps, torch.tensor(duster_vw_flu_poses).cuda()))

    T_conv = np.array([
        [0,   0,  1],
        [-1,  0,  0],
        [0,  -1,  0],  
    ])
    
    # rdf to flu
    #p = T_conv@pts3d[0].detach().cpu().numpy().reshape(-1, 3).T
    p = pts3d[0].detach().cpu().numpy().reshape(-1, 3).T

    #visualize_3d_points(pts3d[0].detach().cpu().numpy())

    print(p.shape)
    print((c*R@p+T[:, None]).T.shape)

    p = (c*R@p+T[:, None]).T.reshape(384, 512, 3)
  
    visualize_valid_3d_points(p, scene.get_masks()[0].detach().cpu().numpy())

    # show original pcd
    #pts3d = scene.get_pts3d_custom(scene.depth_to_pts3d_custom(duster_depthmaps, torch.tensor(duster_vw_flu_poses).cuda()))
    #show_raw_pointcloud(pts3d, scene.imgs)
    #show_raw_pointcloud_with_cams(scene.imgs, pts3d, [m.cpu() for m in scene.get_masks()],
    #                              scene.get_focals(), duster_vw_flu_poses,
    #                              point_size=2, cam_size=0.3, cam_color=None)

    """

if __name__ == '__main__':
    main()
