import numpy as np
import cv2

def solve_transformation_matrix():
    print("Loading calibration dataset...")
    try:
        data = np.load("calibration_data.npz")
        arm_pts = data['arm_pts']
        cam_pts = data['cam_pts']
    except FileNotFoundError:
        print("Error: 'calibration_data.npz' not found. Run the collector first.")
        return

    print(f"Processing {len(arm_pts)} valid 3D point pairs...")

    # estimateAffine3D finds the matrix that maps source (Cam) to destination (Arm)
    # It returns a 3x4 matrix and a list of inliers (points it decided to trust)
    success, affine_matrix, inliers = cv2.estimateAffine3D(cam_pts, arm_pts, ransacThreshold=5.0)

    if not success:
        print("CRITICAL ERROR: Could not solve the transformation matrix.")
        print("The data might be too flat (lacking Z-axis variation) or too noisy.")
        return

    # The result is 3x4. We convert it to a standard 4x4 Homogeneous matrix 
    # by adding [0, 0, 0, 1] to the bottom row.
    T_cam_to_arm = np.vstack((affine_matrix, [0.0, 0.0, 0.0, 1.0]))

    # --- Error Verification (RMSE) ---
    # Let's test how good the matrix actually is by feeding the camera points 
    # back into it and seeing how close they get to the real arm points.
    errors = []
    for i in range(len(cam_pts)):
        # Convert cam point to 4x1 vector: [X, Y, Z, 1]
        cam_vec = np.array([cam_pts[i][0], cam_pts[i][1], cam_pts[i][2], 1.0]).reshape(4, 1)
        
        # Multiply by our new matrix
        predicted_arm_vec = T_cam_to_arm @ cam_vec
        predicted_arm_pt = predicted_arm_vec[:3].flatten()
        
        # Calculate distance (error) between prediction and reality
        dist = np.linalg.norm(predicted_arm_pt - arm_pts[i])
        errors.append(dist)

    rmse = np.sqrt(np.mean(np.square(errors)))
    
    print("\n--- MATRIX SOLVED ---")
    print(T_cam_to_arm)
    print(f"\nCalibration Accuracy (RMSE): {rmse:.2f} mm")
    
    if rmse < 5.0:
        print("Result: EXCELLENT (Sub-5mm accuracy).")
    elif rmse < 15.0:
        print("Result: ACCEPTABLE.")
    else:
        print("Result: POOR. You may want to re-run the data collection.")

    np.save("T_cam_to_arm.npy", T_cam_to_arm)
    print("Saved as 'T_cam_to_arm.npy'")

if __name__ == "__main__":
    solve_transformation_matrix()