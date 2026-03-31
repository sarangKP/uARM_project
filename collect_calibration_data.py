import time
import numpy as np
import cv2
import threading
import constants
from uarm.wrapper import SwiftAPI
from depth_utils import initialize_realsense
from aruco_test import detect_aruco_3d

# Globals to share data between the Vision thread and Arm thread
latest_marker_3d = None
collection_status = "Initializing..."
is_running = True

def arm_routine(arm, calibration_points, arm_pts_collected, cam_pts_collected):
    """Background thread: Moves the arm and samples the live vision data."""
    global collection_status, latest_marker_3d, is_running
    
    try:
        for i, target_pos in enumerate(calibration_points):
            if not is_running: break # Safety exit
            
            # 1. Move
            collection_status = f"Moving to {i+1}/{len(calibration_points)}"
            arm.set_position(x=target_pos[0], y=target_pos[1], z=target_pos[2], speed=10000, wait=True)
            
            # 2. Settle
            collection_status = "Stabilizing..."
            time.sleep(1.0) 
            
            # 3. Capture
            collection_status = "Capturing Data..."
            time.sleep(0.5) # Allow vision thread to update the global variable
            
            # 4. Record
            if latest_marker_3d:
                cam_mm = [val * 1000 for val in latest_marker_3d]
                
                # TCP Offset Injection
                OFFSET_X, OFFSET_Y, OFFSET_Z = 20.0, 0.0, 80.0
                actual_marker_pos = (
                    target_pos[0] + OFFSET_X,
                    target_pos[1] + OFFSET_Y,
                    target_pos[2] + OFFSET_Z
                )
                
                arm_pts_collected.append(actual_marker_pos)
                cam_pts_collected.append(cam_mm)
                print(f" SUCCESS: Mapped {actual_marker_pos} -> {cam_mm}")
            else:
                print(f" FAILED: Marker lost at {target_pos}")

        # 5. Save
        if len(arm_pts_collected) >= 4:
            np.savez("calibration_data.npz", 
                     arm_pts=np.array(arm_pts_collected, dtype=np.float32), 
                     cam_pts=np.array(cam_pts_collected, dtype=np.float32))
            collection_status = f"Success! Saved {len(arm_pts_collected)} points."
        else:
            collection_status = "Failed: Need at least 4 points."
            
    finally:
        collection_status = "Returning Home..."
        arm.set_position(x=150, y=0, z=150, wait=True)
        is_running = False # Tell the vision loop to close

def run_live_collection():
    """Main thread: Handles the RealSense feed and UI."""
    global latest_marker_3d, collection_status, is_running
    
    arm = SwiftAPI(port=constants.PORT)
    arm.waiting_ready()
    arm.set_mode(0)
    
    pipe, aln, flts, intrinsics = initialize_realsense()
    
    calibration_points = [
        (150, -100, 50), (200, -100, 50), (250, -100, 50),
        (150, 0, 50),    (200, 0, 50),    (250, 0, 50),
        (150, 100, 50),  (200, 100, 50),  (250, 100, 50),
        (150, -100, 120), (250, -100, 120),
        (150, 100, 120),  (250, 100, 120),
        (200, 0, 150)
    ]
    
    arm_pts_collected = []
    cam_pts_collected = []
    
    # Start the arm movement in the background
    arm_thread = threading.Thread(target=arm_routine, args=(arm, calibration_points, arm_pts_collected, cam_pts_collected))
    arm_thread.start()
    
    print("\n--- LIVE CALIBRATION STARTED ---")
    print("Press 'Q' in the video window to Emergency Stop.")
    
    try:
        while is_running:
            # 1. Get Frames
            frames = pipe.wait_for_frames()
            aligned = aln.process(frames)
            depth_frame = aligned.get_depth_frame()
            if hasattr(depth_frame, 'as_depth_frame'): 
                depth_frame = depth_frame.as_depth_frame()
            color_image = np.asanyarray(aligned.get_color_frame().get_data())
            
            # 2. Update Global Marker Position
            # detect_aruco_3d handles drawing the bounding box automatically
            latest_marker_3d, _ = detect_aruco_3d(color_image, depth_frame, intrinsics)
            
            # 3. Draw UI Overlays
            cv2.putText(color_image, f"Status: {collection_status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(color_image, f"Points Collected: {len(arm_pts_collected)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 4. Show Live Feed
            cv2.imshow("Live Calibration", color_image)
            
            # Emergency Stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nEmergency Stop Triggered!")
                is_running = False
                break
                
    finally:
        is_running = False
        arm_thread.join() # Wait for arm to stop safely
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_collection()