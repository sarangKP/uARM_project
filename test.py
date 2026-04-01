import cv2
import numpy as np
import time
from uarm.wrapper import SwiftAPI
from depth_utils import initialize_oak, get_3d_camera_coordinate
import constants

# Globals for UI interaction
clicked_pt = None
trigger_move = False

def on_mouse(event, x, y, flags, param):
    global clicked_pt, trigger_move
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pt = (x, y)
        trigger_move = True

def test_transformation_matrix():
    global clicked_pt, trigger_move

    print("Loading transformation matrix...")
    try:
        T_cam_to_arm = np.load("T_cam_to_arm.npy")
    except FileNotFoundError:
        print("ERROR: 'T_cam_to_arm.npy' not found. Run solve_3d_matrix.py first.")
        return

    print("Connecting to uArm...")
    arm = SwiftAPI(port=constants.PORT)
    arm.waiting_ready()
    arm.set_mode(0)
    arm.set_position(x=150, y=0, z=150, speed=10000, wait=True)

    print("Initializing OAK-D Lite...")
    device, q_rgb, q_depth, intrinsics = initialize_oak()

    cv2.namedWindow("Matrix Test - L-Shape")
    cv2.setMouseCallback("Matrix Test - L-Shape", on_mouse)
    
    print("\n--- READY ---")
    print("1. Click any physical object on the live feed.")
    print("2. The arm will hover over it, then drop down.")
    print("3. Press 'Q' to quit and return home.")

    try:
        while True:
            color = q_rgb.get().getCvFrame()
            depth = q_depth.get().getFrame()

            if trigger_move and clicked_pt:
                u, v = clicked_pt
                trigger_move = False # Reset flag immediately
                
                # 1. Get Camera 3D Coordinate (Using our new 5x5 median logic)
                cam_pt_m = get_3d_camera_coordinate(depth, u, v, intrinsics)
                
                if cam_pt_m:
                    # Convert to mm and pad to 4x1 vector [X, Y, Z, 1]
                    cam_vec = np.array([
                        cam_pt_m[0] * 1000.0,
                        cam_pt_m[1] * 1000.0,
                        cam_pt_m[2] * 1000.0,
                        1.0
                    ]).reshape(4, 1)

                    # 2. Apply the Calibration Matrix
                    arm_vec = T_cam_to_arm @ cam_vec
                    target_x = float(arm_vec[0, 0])
                    target_y = float(arm_vec[1, 0])
                    target_z = float(arm_vec[2, 0])

                    print(f"\n[CLICK] ({u}, {v})")
                    print(f"  -> Camera: {cam_vec[:3].flatten()} mm")
                    print(f"  -> Arm:    X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f}")

                    # --- 3. SAFETY LIMITS & EXECUTION ---
                    # Don't let the arm crash into the table (assume table is ~0 Z)
                    safe_z = max(target_z, 15.0) 
                    
                    # Hover height: 80mm above target, but at least Z=100
                    hover_z = max(safe_z + 80.0, 100.0) 
                    
                    print(f"  -> Executing L-Shape Move...")
                    # Phase 1: Move X/Y at safe hover height
                    arm.set_position(x=target_x, y=target_y, z=hover_z, speed=10000, wait=True)
                    time.sleep(0.2)
                    
                    # Phase 2: Plunge down to the object
                    arm.set_position(x=target_x, y=target_y, z=safe_z, speed=5000, wait=True)
                    
                    print("  -> Move complete.")
                else:
                    print(f"\n[ERROR] Invalid depth at ({u}, {v}). Too close/far or no texture.")

            # Draw UI
            cv2.putText(color, "Click object to move arm (L-Shape)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if clicked_pt:
                cv2.drawMarker(color, clicked_pt, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            cv2.imshow("Matrix Test - L-Shape", color)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("\nShutting down, returning home...")
        arm.set_position(x=150, y=0, z=150, wait=True)
        device.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_transformation_matrix()