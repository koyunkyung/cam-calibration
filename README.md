# Camera Calibration with Zhang's Method: Implementation and Comparative Analysis
This repository implements camera calibration using Zhang's method and compares it with OpenCV's built-in calibration.

## Execution Flow

### Step 1: Corner Detection
```bash
python data/loader.py
```
Detects chessboard corners from calibration images.

**Output:** `results/corner_detection/` - Visualize corner detection for all images

### Step 2: Zhang's Calibration
```bash
python zhang_cam.py
```
Three-stage calibration process:
1. Initial linear estimation (DLT)
2. Initial distortion parameter estimation in order to enhance refinement quality
3. Non-linear refinement (Levenberg-Marquardt)

**Outputs:**
- `results/zhang_calibration.json` - Calibration parameters and quantitative summary of reprojection errors for all images
- `results/reprojection/zhang/stage1_linear/` - Visualize stage 1 reprojection errors
- `results/reprojection/zhang/stage2_distortion/` - Visualize stage 2 reprojection errors
- `results/reprojection/zhang/stage3_refined/` - Visualize stage 3 reprojection errors
- `results/undistorted/zhang/` - Undistorted images (based on final distortion parameters from stage 3)

### Step 3: OpenCV Calibration
```bash
python opencv_cam.py
```
Calibration using OpenCV's built-in function for comparison.

**Outputs:**
- `results/opencv_calibration.json` - Calibration parameters and quantitative summary of reprojection errors for all images
- `results/reprojection/opencv/` - Visualize reprojection errors
- `results/undistorted/opencv/` - Undistorted images

## Results

### Calibration Parameters
JSON files contain:
- Camera intrinsic matrix
- Distortion coefficients
- Rotation and translation vectors
- RMS reprojection error
- Summary statistics(mean, min, max, std.) of reprojection errors for all images

### Visual Validation
- **Reprojection images:** Red dots (detected) vs Green dots (reprojected)
- **Undistorted images:** Verify distortion correction quality
- **Zhang's stages:** Track calibration improvement across three stages


## Project Structure
├── data/
│   └── loader.py              # Corner detection and data preprocessing
├── zhang_cam.py               # Zhang's calibration implementation
├── opencv_cam.py              # OpenCV calibration for comparison
└── results/
    ├── corner_detection/      # Detected corners visualization
    ├── zhang_calibration.json # Zhang's calibration results
    ├── opencv_calibration.json# OpenCV calibration results
    ├── reprojection/
    │   ├── zhang/
    │   │   ├── stage1_linear/      # Initial DLT estimation
    │   │   ├── stage2_distortion/  # Distortion parameter estimation
    │   │   └── stage3_refined/     # Non-linear refinement (LM)
    │   └── opencv/                 # OpenCV reprojection errors
    └── undistorted/
        ├── zhang/             # Images undistorted by Zhang's method
        └── opencv/            # Images undistorted by OpenCV