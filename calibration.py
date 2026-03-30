import cv2
import numpy as np
import time
from uarm.wrapper import SwiftAPI
import constants
from depth_utils import initialize_realsense, get_frames

def run_manual_calibration():
    print(f"Connecting to uARM on {constants.PORT}...")
    arm = SwiftAPI(port=constants.PORT)
    arm.waiting_ready()
    arm.set_mode(0)
    
    # Initialize Gripper to CLOSED immediately
    print("Initializing: Closing gripper for point precision...")
    arm.set_gripper(catch=True, wait=True)
    
    pipe, aln, flts = initialize_realsense()
    
    image_pts = []
    arm_pts = []
    current_arm_pos = None
    last_clicked_pixel = None

    def on_mouse(event, x, y, flags, param):
        nonlocal last_clicked_pixel
        if event == cv2.EVENT_LBUTTONDOWN:
            last_clicked_pixel = (x, y)

    cv2.namedWindow("Manual Calibration")
    cv2.setMouseCallback("Manual Calibration", on_mouse)

    print("\n--- MANUAL CALIBRATION (GRIPPER CLOSED) ---")
    print("D: Detach (Move arm) | A: Attach (Lock & Read) | Click: Mark Tip")
    print("N: Next Point        | S: Save & Exit           | Q: Quit")

    try:
        while True:
            _, color = get_frames(pipe, aln, flts)
            
            # Status Overlay
            status = f"Points Collected: {len(image_pts)}"
            cv2.putText(color, status, (10, 30), 1, 1.0, (0, 255, 0), 1)
            cv2.putText(color, "Gripper: LOCKED CLOSED", (10, 50), 1, 1.0, (255, 255, 0), 1)
            
            if last_clicked_pixel:
                cv2.drawMarker(color, last_clicked_pixel, (0, 0, 255), cv2.MARKER_CROSS, 10, 2)

            cv2.imshow("Manual Calibration", color)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('d'):
                # Detach servos but keep gripper closed
                arm.set_servo_detach()
                print("Arm FREE - Gripper remains CLOSED")

            elif key == ord('a'):
                arm.set_servo_attach()
                # Re-assert gripper closure just in case of mechanical slip
                arm.set_gripper(catch=True) 
                time.sleep(0.5)
                pos = arm.get_position()
                current_arm_pos = (pos[0], pos[1])
                print(f"Locked at: {current_arm_pos}")

            elif key == ord('n'):
                if current_arm_pos and last_clicked_pixel:
                    arm_pts.append(current_arm_pos)
                    image_pts.append(last_clicked_pixel)
                    print(f"Point {len(image_pts)} Stored.")
                    last_clicked_pixel = None
                else:
                    print("Action Required: Press 'A' to lock and click the closed tip!")

            elif key == ord('s'):
                if len(image_pts) >= 4:
                    src = np.array(image_pts, dtype='float32')
                    dst = np.array(arm_pts, dtype='float32')
                    M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                    np.savez("calib.npz", M=M)
                    print("Calibration Success! 'calib.npz' created.")
                    break
                else:
                    print(f"Need at least 4 points (Current: {len(image_pts)})")

            elif key == ord('q'):
                break
    finally:
        # Final safety: keep gripper closed and re-attach servos
        arm.set_gripper(catch=True)
        arm.set_servo_attach()
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_manual_calibration()