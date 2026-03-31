# 3D Vision-Guided Pick and Place Pipeline

This repository contains the software pipeline for integrating an Intel RealSense RGB-D camera with a uARM Swift Pro robotic arm. 

Unlike standard 2D top-down setups, this system utilizes a **True 3D Hand-Eye Calibration Pipeline**. It calculates a $4 \times 4$ Homogeneous Transformation Matrix, allowing the camera to be mounted at any angle or height. The system natively understands 3D space, automatically adjusting grip heights for stacked objects or objects of varying sizes without hardcoded Z-height tables.

## 🛠 Hardware Setup
1. **Robotic Arm:** uARM Swift Pro.
2. **Camera:** Intel RealSense (e.g., D435/D415) mounted in an "Eye-to-Hand" configuration (fixed to the environment, looking at the robot workspace).
3. **Calibration Tool:** A printed **ArUco Marker** (Dictionary: `DICT_4X4_50`, ID: `0`).

## 📐 Physical Preparation (Calibration Phase)
Before running the calibration pipeline, you must prepare the physical workspace:
1. **Clear the Workspace:** Remove all objects, cubes, and bins from the table. The arm will move through a pre-programmed 3D grid and needs a collision-free environment.
2. **Attach the ArUco Marker:** Mount the ArUco marker securely to the top of the uARM's gripper. It must lay perfectly flat relative to the arm's base.
3. **Configure TCP Offsets:** Open `collect_calibration_data.py` and ensure the `OFFSET_X`, `OFFSET_Y`, and `OFFSET_Z` variables perfectly match the physical distance from the very tip of the closed gripper to the absolute center of the ArUco marker.

---

## 🚀 Pipeline Execution Order

The pipeline is split into three distinct steps. Steps 1 and 2 only need to be run when the camera or robot base is physically moved. Step 3 is your daily operational script.

### Step 1: Data Collection
**Command:** `python collect_calibration_data.py`
* **Action:** The system runs a threaded background loop moving the arm to 14 distinct $(X, Y, Z)$ coordinates in space. Simultaneously, the RealSense camera tracks the ArUco marker in 3D space using a sub-pixel solver. 
* **TCP Injection:** The physical offsets are mathematically injected during this phase so the final dataset aligns with the gripper tip, not the marker.
* **Output:** Generates `calibration_data.npz` containing paired spatial arrays.

### Step 2: Matrix Solver
**Command:** `python solve_3d_matrix.py`
* **Action:** Consumes the collected `.npz` dataset. It uses OpenCV's RANSAC-backed `estimateAffine3D` to calculate the optimal $4 \times 4$ transformation matrix ($T_{cam}^{arm}$) that aligns the camera's 3D universe with the robot's 3D universe.
* **Validation:** Prints the Root Mean Square Error (RMSE) in millimeters. A successful calibration will yield an RMSE of `< 5.0 mm`.
* **Output:** Generates `T_cam_to_arm.npy`.

### Step 3: Live 3D Sorting
**Command:** `python main_sorter.py`
* **Prep:** Remove the ArUco marker from the gripper. Place target objects in the workspace.
* **Action:** Loads the YOLOv8 model and the 3D Transformation Matrix. 
* **Interaction:** The UI displays real-time YOLO bounding boxes. Click on any green bounding box to execute a pick-and-place operation.

---

## 🧠 System Architecture & Module Breakdown

* **`main_sorter.py`**: The core threaded application. Handles the UI, matrix multiplication ($P_{arm} = T_{cam}^{arm} \times P_{cam}$), kinematic safety checks, and background hardware commands.
* **`depth_utils.py`**: Interfaces with the RealSense SDK. Features a **Median ROI Filter** that samples a 5x5 grid of pixels around a target to eliminate raw infrared noise, yielding highly stable 3D coordinates.
* **`color_detection.py`**: Runs the YOLOv8 (`best.pt`) inference to output 2D bounding boxes and class labels (colors).
* **`collect_calibration_data.py`**: The automated data generation script for Hand-Eye calibration.
* **`solve_3d_matrix.py`**: The mathematical solver for aligning the 3D point clouds.
* **`constants.py`**: Stores serial ports, approach heights, speeds, and bin drop-off coordinates.

## 🛡 Built-in Safety Features
* **Kinematic Radial Boundaries:** `main_sorter.py` continuously calculates the distance of detected objects from the arm's base motor ($R = \sqrt{X^2 + Y^2}$). If an object falls inside the mechanical deadzone ($< 120$mm) or out of reach ($> 320$mm), the UI marks it **UNREACHABLE** in red and software-blocks the click to prevent motor strain.
* **Dynamic Table Floor:** The Z-height is calculated based on the physical top of the object minus 10mm for grip security. A `max()` function safety floor prevents the arm from ever being commanded to drive the gripper into the table itself.