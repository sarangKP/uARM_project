import cv2
import numpy as np
import time
import threading # Added for background tasks
import constants
from uarm.wrapper import SwiftAPI
from depth_utils import initialize_realsense, get_frames
from color_detection import get_detections
from stack_detection import get_stack_info

# Global flags
current_detections = []
arm_is_busy = False # Prevent multiple clicks from overlapping

def transform_pixel_to_arm(u, v, M):
    pixel_pt = np.array([u, v, 1.0]).reshape(3, 1)
    arm_pt = M @ pixel_pt
    arm_pt /= arm_pt[2]
    return arm_pt[0][0], arm_pt[1][0]

def background_pick_task(arm, x, y, z_pick, color_label, M_matrix):
    global arm_is_busy
    arm_is_busy = True
    spd = constants.ARM_SPEED
    
    try:
        bin_pos = constants.BIN_POSITIONS.get(color_label, {'x': 150, 'y': 0, 'z': 150})
        
        # 1. Move to Target
        arm.set_position(x=x, y=y, z=constants.APPROACH_Z, speed=spd, wait=True)
        arm.set_gripper(catch=False, wait=True)
        arm.set_position(x=x, y=y, z=z_pick, speed=spd, wait=True)
        
        # 2. Secure Grab
        arm.set_gripper(catch=True, wait=True)
        time.sleep(constants.PICK_DELAY) # <--- New Constant
        
        # 3. High-Speed Transfer
        arm.set_position(x=x, y=y, z=constants.APPROACH_Z, speed=spd, wait=True)
        arm.set_position(x=bin_pos['x'], y=bin_pos['y'], z=constants.APPROACH_Z, speed=spd, wait=True)
        arm.set_position(x=bin_pos['x'], y=bin_pos['y'], z=bin_pos['z'], speed=spd, wait=True)
        
        # 4. Release
        arm.set_gripper(catch=False, wait=True)
        time.sleep(constants.PLACE_DELAY) # <--- New Constant
        
        # 5. Return Home
        arm.set_position(x=bin_pos['x'], y=bin_pos['y'], z=constants.APPROACH_Z, speed=spd, wait=True)
        arm.set_position(x=150, y=0, z=150, speed=spd, wait=True)
        
    finally:
        arm_is_busy = False

def on_mouse_click(event, u, v, flags, param):
    global current_detections, arm_is_busy
    # Only allow click if arm isn't already doing something
    if event == cv2.EVENT_LBUTTONDOWN and not arm_is_busy:
        for obj in current_detections:
            x1, y1, x2, y2 = obj['box']
            if x1 <= u <= x2 and y1 <= v <= y2:
                # Prepare data for the thread
                depth_frame, arm_obj, M_matrix = param
                num_cubes, height, z_pick = get_stack_info(depth_frame, obj['center'][0], obj['center'][1])
                x_arm, y_arm = transform_pixel_to_arm(obj['center'][0], obj['center'][1], M_matrix)
                
                # Start the background thread
                t = threading.Thread(target=background_pick_task, 
                                     args=(arm_obj, x_arm, y_arm, z_pick, obj['color'], M_matrix))
                t.start()
                break

def main():
    global current_detections
    M = np.load("calib.npz")['M']
    arm = SwiftAPI(port=constants.PORT)
    arm.waiting_ready()
    arm.set_mode(0)
    
    pipe, aln, flts = initialize_realsense()
    cv2.namedWindow("Live Sorter")
    
    print("System Ready. Feed is now independent of Arm movement.")

    try:
        while True:
            depth, color = get_frames(pipe, aln, flts)
            
            # Pass fresh depth and arm objects to the mouse callback
            cv2.setMouseCallback("Live Sorter", on_mouse_click, param=(depth, arm, M))
            
            if not arm_is_busy:
                current_detections = get_detections(color)
            
            # Draw UI
            for obj in current_detections:
                x1, y1, x2, y2 = obj['box']
                color_bgr = (0, 255, 0) if not arm_is_busy else (100, 100, 100)
                cv2.rectangle(color, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(color, obj['color'], (x1, y1-10), 1, 1.2, color_bgr, 2)

            if arm_is_busy:
                cv2.putText(color, "ARM BUSY - PLEASE WAIT", (200, 30), 1, 1.5, (0, 0, 255), 2)

            cv2.imshow("Live Sorter", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        arm.set_position(x=150, y=0, z=150, wait=True)

if __name__ == "__main__":
    main()