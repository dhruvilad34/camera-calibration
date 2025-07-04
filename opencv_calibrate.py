import cv2
import numpy as np

# Load 2D or 3D points from a file, skipping malformed lines and the count
def load_points(file_path, dims):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # skip the first line (point count)
        points = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != dims:
                continue  # skip malformed or empty lines
            points.append([float(x) for x in parts]) #converts strings to floats 
        return np.array(points, dtype=np.float32)

# Load 3D and 2D points
object_points = load_points("3D.txt", 3).reshape(-1, 1, 3) #shape: (N,1,3)
image_points = load_points("2D.txt", 2).reshape(-1, 1, 2) #Shape: (N,1,2)

# Provide an initial guess for K (intrinsic camera matrix)
K = np.array([
    [500, 0, 256],   # fx,  0, cx
    [0, 500, 256],   #  0, fy, cy
    [0,   0,   1]    #homogeneous coordinates row 
], dtype=np.float32)

# Assume zero distortion initially
dist = np.zeros(5)

# Calibrate using OpenCV with initial guess for K
flags = cv2.CALIB_USE_INTRINSIC_GUESS
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    [object_points], [image_points], (512, 512), K, dist, flags=flags
    #list of 3D arrays , list of 2D arrays, image size, initial guess for intrinsic matrix, guess for distortion(all zeros)
)

# Compute the 3x4 projection matrix [R|t]
R, _ = cv2.Rodrigues(rvecs[0])
Rt = np.hstack((R, tvecs[0]))  # Combine rotation and translation
P = K @ Rt  # Final 3x4 projection matrix

# Output calibration matrix
print("===> OpenCV Calibration Matrix (3x4) <===")
print(P)

# Reproject 3D points to 2D and compute error
projected, _ = cv2.projectPoints(object_points, rvecs[0], tvecs[0], K, dist)

#rashape projected ponits to match shape of image_points for error computation
projected = projected.reshape(-1, 2)

#calculate the eduliden error b/w actual and projectd 2D pomts 
error = np.linalg.norm(projected - image_points.reshape(-1, 2), axis=1)

#output avarage reprojection error (in pixels)
print("\nAverage Reprojection Error (OpenCV):", np.mean(error))
