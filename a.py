import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Input array of xyz, yaw, pitch values (in radians)
viewpoints = [
    np.array([[-0.9106915 , -0.35869193,  1.5       ,  0.03682264,  0.56596261]]),
    np.array([[ 1.        , -0.0445565 ,  1.5       , -0.07497325, -1.70059419]]),
    np.array([[ 1.        , -0.19564176,  1.5       ,  0.00416196, -2.24816656]]),
    np.array([[ 0.63641173, -0.04201406,  1.5       ,  0.03324832, -2.28935552]]),
    np.array([[ 0.94665867, -0.34533122,  1.5       ,  0.07879943, -1.79868162]]),
    np.array([[-0.79603714, -0.55136257,  1.5       ,  0.26035082,  0.92895687]]),
    np.array([[-0.46497941,  0.31439728,  1.5       ,  0.29064447,  0.27422833]])
]

# Initialize a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define a function to convert yaw and pitch to a direction vector in a forward-left-up system
def yaw_pitch_to_vector(yaw, pitch):
    # Forward-left-up system: x -> forward, y -> left, z -> up
    x = np.cos(pitch) * np.cos(yaw)  # Forward direction (x-axis)
    y = np.cos(pitch) * np.sin(yaw)  # Left direction (y-axis)
    z = np.sin(pitch)                # Up direction (z-axis)
    return np.array([x, y, z])

# Visualize each viewpoint as a point and its looking vector
for idx, vp in enumerate(viewpoints, start=1):
    xyz = vp[0][:3]  # Position (x, y, z)
    yaw = vp[0][3]   # Yaw angle
    pitch = vp[0][4] # Pitch angle
    
    # Plot the viewpoint as a point
    ax.scatter(xyz[0], xyz[1], xyz[2], color='b', label=f'Viewpoint {idx}')
    
    # Label the points with their index
    ax.text(xyz[0], xyz[1], xyz[2], f'{idx}', fontsize=12, color='black')
    
    # Calculate the looking direction vector and plot it as an arrow
    direction = yaw_pitch_to_vector(yaw, pitch)
    ax.quiver(xyz[0], xyz[1], xyz[2], direction[0], direction[1], direction[2], length=0.5, color='r')

# Plot the origin (0, 0, 0) as a reference point
ax.scatter(0, 0, 0, color='g', s=100, label='Origin')
ax.text(0, 0, 0, 'Origin', fontsize=12, color='black')

# Set plot labels and show the 3D plot
ax.set_xlabel('X (Forward)')
ax.set_ylabel('Y (Left)')
ax.set_zlabel('Z (Up)')
ax.set_title('Viewpoints and Looking Vectors in Forward-Left-Up System')
plt.show()

