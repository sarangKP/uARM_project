import cv2
import numpy as np
import time
import threading
import constants
from uarm.wrapper import SwiftAPI
from depth_utils import initialize_realsense, get_3d_camera_coordinate
from color_detection import get_detections

current_detections = []
arm_is_busy = False

# uARM Swift Pro safe physical limits
MIN_RADIUS = 120.0 
MAX_RADIUS = 320.0 

def transform_cam_to_arm(pt_3d_cam, T_matrix):
    x_mm, y_mm, z_mm = pt_3d_cam[0] * 1000, pt_3d_cam[1] * 1000, pt_3d_cam[2] * 1000
    cam_vec = np.array([x_mm, y_mm, z_mm, 1.0])
    arm_vec = T_matrix @ cam_vec
    return arm_vec[:3]

def background_pick_task(arm, x, y, z_surface, color_label):
    global arm_is_busy
    arm_is_busy = True
    spd = constants.ARM_SPEED

    z_pick = z_surface - 10.0 
    z_pick = max(constants.TABLE_Z + 2.0, z_pick)
    
    # --- NEW: Dynamic Safe Travel Plane ---
    # We define the "highest point" as either 150mm (standard safe height) 
    # or 50mm ABOVE the object itself, whichever is taller.
    travel_z = max(150.0, z_surface + 50.0)

    try:
        bin_pos = constants.BIN_POSITIONS.get(color_label, {'x': 150, 'y': 0, 'z': 150})
        
        # 1. Lift straight up to Safe Travel Plane (from current position)
        current_pos = arm.get_position()
        if current_pos:
            arm.set_position(x=current_pos[0], y=current_pos[1], z=travel_z, speed=spd, wait=True)
            
        # 2. Move horizontally to hover perfectly over the target
        arm.set_position(x=x, y=y, z=travel_z, speed=spd, wait=True)
        arm.set_gripper(catch=False, wait=True)
        
        # 3. Dive straight down to exact 3D height
        arm.set_position(x=x, y=y, z=z_pick, speed=spd, wait=True)
        
        # 4. Secure Grab
        arm.set_gripper(catch=True, wait=True)
        time.sleep(constants.PICK_DELAY)
        
        # 5. Lift straight back up to Safe Travel Plane
        arm.set_position(x=x, y=y, z=travel_z, speed=spd, wait=True)
        
        # 6. High-Speed Horizontal Transfer to bin
        arm.set_position(x=bin_pos['x'], y=bin_pos['y'], z=travel_z, speed=spd, wait=True)
        
        # 7. Dive into bin
        arm.set_position(x=bin_pos['x'], y=bin_pos['y'], z=bin_pos['z'], speed=spd, wait=True)
        
        # 8. Release
        arm.set_gripper(catch=False, wait=True)
        time.sleep(constants.PLACE_DELAY)
        
        # 9. Lift straight out of bin
        arm.set_position(x=bin_pos['x'], y=bin_pos['y'], z=travel_z, speed=spd, wait=True)
        
        # 10. Return Home
        arm.set_position(x=150, y=0, z=150, speed=spd, wait=True)
        
    finally:
        arm_is_busy = False

def on_mouse_click(event, u, v, flags, param):
    global current_detections, arm_is_busy
    if event == cv2.EVENT_LBUTTONDOWN and not arm_is_busy:
        depth_frame, intrinsics, arm_obj, T_matrix = param
        
        for obj in current_detections:
            x1, y1, x2, y2 = obj['box']
            if x1 <= u <= x2 and y1 <= v <= y2:
                pt_3d_cam = get_3d_camera_coordinate(depth_frame, obj['center'][0], obj['center'][1], intrinsics)
                
                if pt_3d_cam:
                    arm_x, arm_y, arm_z = transform_cam_to_arm(pt_3d_cam, T_matrix)
                    radius = np.hypot(arm_x, arm_y) # Calculates sqrt(x^2 + y^2)
                    
                    if MIN_RADIUS <= radius <= MAX_RADIUS:
                        print(f"\nTargeting {obj['color']} cube at XYZ: ({arm_x:.1f}, {arm_y:.1f}, {arm_z:.1f})")
                        t = threading.Thread(target=background_pick_task, 
                                             args=(arm_obj, arm_x, arm_y, arm_z, obj['color']))
                        t.start()
                    else:
                        print(f"\nClick Ignored: Object is out of reach (Radius: {radius:.1f}mm)")
                break

def main():
    global current_detections
    print("Loading 3D Transformation Matrix...")
    try:
        T_matrix = np.load("T_cam_to_arm.npy")
    except FileNotFoundError:
        print("Error: T_cam_to_arm.npy not found.")
        return

    arm = SwiftAPI(port=constants.PORT)
    arm.waiting_ready()
    arm.set_mode(0)
    
    pipe, aln, flts, intrinsics = initialize_realsense()
    cv2.namedWindow("Live 3D Sorter")
    
    print("3D System Ready. Kinematic boundaries active.")

    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = aln.process(frames)
            
            depth_frame = aligned.get_depth_frame()
            if hasattr(depth_frame, 'as_depth_frame'):
                depth_frame = depth_frame.as_depth_frame()
                
            color = np.asanyarray(aligned.get_color_frame().get_data())
            
            cv2.setMouseCallback("Live 3D Sorter", on_mouse_click, param=(depth_frame, intrinsics, arm, T_matrix))
            
            if not arm_is_busy:
                current_detections = get_detections(color)
            
            # --- NEW UI DRAWING LOGIC ---
            for obj in current_detections:
                x1, y1, x2, y2 = obj['box']
                u, v = obj['center']
                
                # Check reachability for the UI
                reachable = False
                pt_3d_cam = get_3d_camera_coordinate(depth_frame, u, v, intrinsics)
                if pt_3d_cam:
                    arm_x, arm_y, _ = transform_cam_to_arm(pt_3d_cam, T_matrix)
                    if MIN_RADIUS <= np.hypot(arm_x, arm_y) <= MAX_RADIUS:
                        reachable = True

                # Set colors based on state
                if arm_is_busy:
                    color_bgr, text = (100, 100, 100), obj['color']
                elif not reachable:
                    color_bgr, text = (0, 0, 255), "UNREACHABLE"
                else:
                    color_bgr, text = (0, 255, 0), obj['color']

                cv2.rectangle(color, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(color, text, (x1, y1-10), 1, 1.2, color_bgr, 2)

            if arm_is_busy:
                cv2.putText(color, "ARM BUSY", (200, 30), 1, 1.5, (0, 0, 255), 2)

            cv2.imshow("Live 3D Sorter", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        arm.set_position(x=150, y=0, z=150, wait=True)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()