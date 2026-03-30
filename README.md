# Vision-Guided Robotic Color Sorting System
**Version 1.0 | March 2026**

## Project Overview
An AI-powered pick-and-place system using a **uARM Swift Pro** and an **Intel RealSense D435i**. The system utilizes a top-down camera mount to detect colored cubes, determine stack heights, and sort them into bins using a YOLOv8 computer vision model.

## Features
* **AI Detection**: Uses `best.pt` (YOLOv8) for robust color identification.
* **Stack Awareness**: Calculates object height using median-filtered depth data to pick from 1, 2, or 3-cube stacks.
* **Interactive UI**: Multithreaded OpenCV feed allows "Click-to-Pick" without freezing the video stream.
* **High-Speed Execution**: Optimized for up to 20,000 mm/min with configurable gripper delays for stability.

## Hardware Setup
1. **Camera**: RealSense D435i mounted 550mm above the table (top-down).
2. **Arm**: uARM Swift Pro connected via USB at `/dev/ttyACM0`.
3. **Workspace**: Flat white surface for maximum contrast.

## Installation & Usage
1. Ensure `pyrealsense2`, `ultralytics`, and `uarm` SDK are installed in your `.venv`.
2. **Calibrate**: Run `python3 calibration.py`. Manually guide the arm to ~12 points, lock (A), and click the tip in the feed. Save with (S).
3. **Verify**: Run `python3 verify_calib.py` to ensure the arm lands where you click.
4. **Operate**: Run `python3 main_sorter.py`. Click a detected cube in the live feed to sort it.

## Configuration
All physical constants (Speed, Table Depth, Port, Bin Coordinates) are located in `constants.py`.