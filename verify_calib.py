import cv2
import numpy as np
from uarm.wrapper import SwiftAPI
import constants
from depth_utils import initialize_realsense, get_frames

def verify():
    # Load the matrix you just created
    try:
        data = np.load("calib.npz")
        M = data['M']
        print("Calibration matrix loaded.")
    except:
        print("Error: calib.npz not found. Run calibration first.")
        return

    arm = SwiftAPI(port=constants.PORT)
    arm.waiting_ready()
    arm.set_mode(0)
    arm.set_gripper(catch=True) # Keep it closed for visual check
    arm.set_position(x=150, y=0, z=150, wait=True)

    pipe, aln, flts = initialize_realsense()

    def on_click(event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Transform pixel (u, v) to Arm (x, y)
            pixel_pt = np.array([u, v, 1.0]).reshape(3, 1)
            arm_pt = M @ pixel_pt
            arm_pt /= arm_pt[2] # Normalize
            
            x_arm, y_arm = arm_pt[0][0], arm_pt[1][0]
            
            print(f"Clicked: ({u}, {v}) -> Arm: X={x_arm:.1f}, Y={y_arm:.1f}")
            # Move to clicked position at a safe Z
            arm.set_position(x=x_arm, y=y_arm, z=constants.APPROACH_Z, wait=True)

    cv2.namedWindow("Verify Accuracy")
    cv2.setMouseCallback("Verify Accuracy", on_click)

    try:
        while True:
            _, color = get_frames(pipe, aln, flts)
            cv2.putText(color, "Click to move arm tip to that spot", (20, 30), 1, 1.2, (255, 255, 0), 2)
            cv2.imshow("Verify Accuracy", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    verify()