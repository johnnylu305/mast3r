from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.demo import get_3D_model_from_scene
from dust3r.image_pairs import make_pairs
from dust3r.viz import *


import os
import numpy as np
from PIL import Image

from scipy.spatial.transform import Rotation as R


def matrix_to_euler_angles(matrix):
    """ Extract roll, pitch, yaw from a 3x3 rotation matrix (in radians). """
    r = R.from_matrix(matrix[:3, :3])
    return r.as_euler('xyz', degrees=True)


def adjust_intrinsic_matrix(K, original_size, new_size, crop_box):
    original_width, original_height = original_size
    new_width, new_height = new_size
    
    # Scaling factor for width and height
    scale_w = new_width / original_width
    scale_h = new_height / original_height
    
    # Adjust focal lengths (f_x and f_y)
    K[0, 0] *= scale_w  # Adjust f_x
    K[1, 1] *= scale_h  # Adjust f_y
    
    # Adjust the principal point (c_x and c_y) based on scaling
    K[0, 2] *= scale_w  # Adjust c_x
    K[1, 2] *= scale_h  # Adjust c_y
    
    # Crop the principal point
    left, top, _, _ = crop_box
    K[0, 2] -= left  # Adjust c_x based on the left crop
    K[1, 2] -= top   # Adjust c_y based on the top crop
    
    return K

def adjust_intrinsic(K, img_path, resize_size=512, square_ok=False):
    img = Image.open(img_path)
    original_width, original_height = img.size
    
    # Step 1: Resize the image such that the long side becomes 512
    aspect_ratio = original_width / original_height
    if original_width > original_height:
        new_width = resize_size
        new_height = int(resize_size / aspect_ratio)
    else:
        new_height = resize_size
        new_width = int(resize_size * aspect_ratio)
    
    new_size = (new_width, new_height)
    img_resized = img.resize(new_size)
    
    # Step 2: Calculate crop box (centered crop)
    cx, cy = new_width // 2, new_height // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    
    if not square_ok and new_width == new_height:
        halfh = (3 * halfw) // 4
    
    crop_box = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
    
    # Perform cropping
    img_cropped = img_resized.crop(crop_box)
    
    # Step 3: Adjust the intrinsic matrix
    adjusted_K = adjust_intrinsic_matrix(K, (original_width, original_height), new_size, crop_box)
    
    return adjusted_K


def create_unity_camera_pose(x, y, z, roll_deg, yaw_deg, pitch_deg):
    """
    Create a 4x4 camera pose matrix given x, y, z, roll, yaw, and pitch in Unity's left-handed system.
    
    Parameters:
    x, y, z (float): The translation values (position in 3D space).
    roll_deg, yaw_deg, pitch_deg (float): Rotation values in degrees for roll (around X), yaw (around Y), and pitch (around Z).
    
    Returns:
    np.ndarray: The 4x4 camera pose matrix.
    """
    #pitch_deg = 0
    #roll_deg = 0
    # Create the rotation matrix using scipy, following the ZYX (Yaw, Pitch, Roll) order for Unity
    r = R.from_euler('zyx', [[roll_deg, yaw_deg, pitch_deg]], degrees=True)
    
    # Get the rotation matrix from the Rotation object
    rotation_matrix = r.as_matrix()
    
    # Create the full 4x4 camera pose matrix
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = rotation_matrix
    camera_pose[:3, 3] = np.array([x, y, z])
    
    return camera_pose


if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    #images = load_images(['dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png'], size=512)
    images = load_images(['/home/dsr/Documents/indoor_demo/MAD3DDataset/double_drones_opera_house/cam01/drone2_2024-08-06_18-04-20-234.png', '/home/dsr/Documents/indoor_demo/MAD3DDataset/double_drones_opera_house/cam01/drone2_2024-08-06_18-04-41-492.png'], size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    # unknown initial pose
    #scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    #loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)   
    #show_raw_pointcloud(scene.get_pts3d(), scene.imgs)


    # known pose
    # First known pose
    
    """
    transformation_matrix_1 = np.array([[-3.99755654e-01,  1.22123956e-01, -9.08449864e-01, 8.27700000e-01],
                           [1.10726104e-03,  9.91148403e-01,  1.32753975e-01, 1.61280000e+00],
                           [9.16621073e-01,  5.20632609e-02, -3.96352401e-01, 1.32900000e-01],
                           [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

    # Second known pose
    transformation_matrix_2 = np.array([[-0.62794311,  0.10174008, -0.77158046,  0.6891],
                           [0.00169437,  0.99159467,  0.12937207,  1.5824],
                           [0.7782574 ,  0.07993096, -0.62283743,  0.3512],
                           [0.00000000,  0.00000000,  0.00000000,  1.00000000]])
    """

    transformation_matrix_1 = create_unity_camera_pose(-1.4041, 0.7341, 0.4594, 358.8948, 107.3505, 356.0876)

    transformation_matrix_2 = create_unity_camera_pose(-1.0758, 0.7244, 0.8118, 358.8884, 125.4811, 356.1120)

    # Convert rotation matrices to roll, pitch, yaw angles in degrees (RDF)
    unity_rpy = matrix_to_euler_angles(transformation_matrix_1)

    print("Unity Pose (RDF) RPY:", unity_rpy)
    
    """
    transformation_matrix_1 = np.array([
    [-0.23990719764430002, -0.8802962825141344, 0.409295726264687, 1.3947694985346608],
    [0.8204323296915305, 0.04153850398097329, 0.5702327113911485, 2.512560396324243],
    [-0.5189752681604244, 0.4726023780270853, 0.7122581437974531, 3.0784748314369788],
    [0, 0, 0, 1]
])

    transformation_matrix_2 = np.array([
    [-0.5956119945374981, -0.7324816919582682, 0.3297224938297349, 1.62237453756682],
    [0.6632602957964362, -0.21690120661880966, 0.716267895638823, 2.609601349430533],
    [-0.45313588690791107, 0.645309567246197, 0.6150149839650577, 3.0006323429822923],
    [0, 0, 0, 1]
])
    

    # Convert rotation matrices to roll, pitch, yaw angles in degrees (RDF)
    opengl_rpy = matrix_to_euler_angles(transformation_matrix_1)

    print("OpenGL Pose (RDF) RPY:", opengl_rpy)
    """

    conversion_matrix = np.array([[1.0,  0.0,  0.0,  0.0],
                                  [0.0, -1.0,  0.0,  0.0],
                                  [0.0,  0.0,  1.0,  0.0],
                                  [0.0,  0.0,  0.0,  1.0]])

    #transformation_matrix_1 = conversion_matrix @ transformation_matrix_1
    #transformation_matrix_2 = conversion_matrix @ transformation_matrix_2


    known_intrinsic = np.array([[2840.08728419576, 0.0, 2006.8516906258146],
                                 [0.0, 2842.499106885914, 1524.2463053694473],
                                 [0.0, 0.0, 1.0]])

    known_intrinsic = adjust_intrinsic(known_intrinsic, '/home/dsr/Documents/indoor_demo/MAD3DDataset/double_drones_opera_house/cam01/drone1_2024-08-06_18-04-12-194.png', resize_size=512, square_ok=False)

    known_pose = torch.tensor(transformation_matrix_1).to(device)
    known_pose_2 = torch.tensor(transformation_matrix_2).to(device)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.preset_pose([known_pose, known_pose_2])
    #scene.preset_intrinsics([known_intrinsics, known_intrinsics])
    scene.preset_focal([known_intrinsic.diagonal()[:2].mean()])
    #scene.preset_principal_point([known_intrinsic[:2, 2]])
    loss = scene.compute_global_alignment(init='known_poses', niter=niter, schedule=schedule, lr=lr)


    #show_raw_pointcloud(scene.get_pts3d(), scene.imgs)
    show_raw_pointcloud_with_cams(scene.imgs, scene.get_pts3d(), [m.cpu() for m in scene.get_masks()], 
                                  scene.get_focals(), scene.get_im_poses(),
                                  point_size=2, cam_size=0.3, cam_color=None)
