import cv2
import numpy as np
from depth_utils import initialize_realsense, get_3d_camera_coordinate

# Global to store clicked coordinates
last_click_px = None

def transform_cam_to_arm(pt_3d_cam, T_matrix):
    """Converts a Camera 3D point (meters) to an Arm 3D point (mm) using the 4x4 matrix."""
    # 1. Convert meters to mm
    x_mm, y_mm, z_mm = pt_3d_cam[0] * 1000, pt_3d_cam[1] * 1000, pt_3d_cam[2] * 1000
    
    # 2. Create a 4x1 Homogeneous Vector [X, Y, Z, 1]
    cam_vec = np.array([x_mm, y_mm, z_mm, 1.0])
    
    # 3. Matrix Multiplication
    arm_vec = T_matrix @ cam_vec
    
    # 4. Return the X, Y, Z (ignoring the trailing 1.0)
    return arm_vec[:3]

def on_mouse(event, u, v, flags, param):
    global last_click_px
    if event == cv2.EVENT_LBUTTONDOWN:
        last_click_px = (u, v)

def run_pointer_test():
    print("Loading Matrix...")
    try:
        T_matrix = np.load("T_cam_to_arm.npy")
    except FileNotFoundError:
        print("Error: T_cam_to_arm.npy not found.")
        return

    pipe, aln, flts, intrinsics = initialize_realsense()
    
    cv2.namedWindow("3D Pointer Test")
    cv2.setMouseCallback("3D Pointer Test", on_mouse)
    
    print("\n--- 3D POINTER TEST ---")
    print("Click anywhere on the feed. The system will calculate the Arm's target XYZ.")

    global last_click_px
    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = aln.process(frames)
            
            depth_frame = aligned.get_depth_frame()
            if hasattr(depth_frame, 'as_depth_frame'):
                depth_frame = depth_frame.as_depth_frame()
                
            color = np.asanyarray(aligned.get_color_frame().get_data())
            
            if last_click_px:
                u, v = last_click_px
                cv2.drawMarker(color, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                
                # 1. Get Camera 3D
                pt_3d_cam = get_3d_camera_coordinate(depth_frame, u, v, intrinsics)
                
                if pt_3d_cam:
                    # 2. Transform to Arm 3D
                    arm_xyz = transform_cam_to_arm(pt_3d_cam, T_matrix)
                    
                    arm_x, arm_y, arm_z = [round(val, 1) for val in arm_xyz]
                    
                    cv2.putText(color, f"Target Arm XYZ: {arm_x}, {arm_y}, {arm_z}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(color, "Invalid Depth at click.", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("3D Pointer Test", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pointer_test()