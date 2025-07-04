# Camera Calibration & 3D Reconstruction

This project performs pinhole camera calibration using OpenCV and SVD-based linear methods. It reconstructs 3D world coordinates from 2D projections and visualizes accuracy with error metrics and debugging plots.

## Features
- Intrinsic and extrinsic parameter estimation
- 3D reconstruction from image projections
- Fundamental matrix computation
- Visualization of calibration accuracy

## Technologies Used
- Python
- OpenCV
- NumPy
- Linear Algebra (SVD)
- Matplotlib

## Files
- `opencv_calibrate.py`: Uses OpenCV’s built-in calibration tools.
- `svd_calibrate.py`: Custom calibration using SVD and matrix decomposition.
- `3D.txt`: Input 3D coordinates for world reference.

## Usage
```bash
python opencv_calibrate.py
python svd_calibrate.py
