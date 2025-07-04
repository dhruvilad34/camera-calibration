import numpy as np

# Load 2D or 3D points from a file, skipping malformed lines and the count
def load_points(file_path, dims):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # skip the first line (point count)
        points = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != dims:
                continue #skip malformed lines 
            points.append([float(x) for x in parts]) #converts strings to floats 
        return np.array(points, dtype=np.float64)

# Load 3D and 2D points
object_points = load_points("3D.txt", 3) #each row x,y,z 
image_points = load_points("2D.txt", 2) #each row: u,v 

# Build matrix A for Direct Linear Transform (DLT)
A = []
for i in range(len(object_points)):
    X, Y, Z = object_points[i]
    u, v = image_points[i]
    A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]) #1st row corresponds to x-equation 
    A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]) #2nd row corresponds to y-equation
A = np.array(A)

# Solve using SVD
_, _, Vt = np.linalg.svd(A)
P = Vt[-1].reshape(3, 4)  # Last row of V^T gives the solution

# Print calibration matrix
print("===> SVD Calibration Matrix (3x4) <===")
print(P)

#stage : 01 
# Project 3D points into 2D using the matrix P
homogeneous_3D = np.hstack((object_points, np.ones((len(object_points), 1))))  # [X Y Z 1]

#stage : 02 
#multiply each 3D ponits into 2D with projection matrix 
projected = (P @ homogeneous_3D.T).T #shape: (N,3)

#stage : 03
#converts projects 3D points into 2D by dividing by Z (homogeneous normalization)
projected = projected[:, :2] / projected[:, 2, np.newaxis]  # Normalize by Z

# Compute average pixel error
error = np.linalg.norm(projected - image_points, axis=1)
print("\nAverage Reprojection Error (SVD):", np.mean(error))
