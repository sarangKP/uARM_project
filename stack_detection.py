import numpy as np
import cv2
import constants
from depth_utils import initialize_realsense, get_frames

def get_stack_info(depth_data, u, v):
    u_min, u_max = max(0, u-2), min(639, u+3)
    v_min, v_max = max(0, v-2), min(479, v+3)
    
    roi = depth_data[v_min:v_max, u_min:u_max]
    valid_samples = roi[roi > 0]
    
    if len(valid_samples) < 12:
        return 0, 0, constants.APPROACH_Z

    object_depth_mm = np.median(valid_samples)
    height_mm = constants.TABLE_DEPTH_MM - object_depth_mm
    
    # Logic: If height < 12mm, it's just table noise. 
    # If 12-37mm, it's 1 cube. 37-62mm, it's 2 cubes, etc.
    num_cubes = int(max(0, min(3, round(height_mm / constants.CUBE_HEIGHT))))
    
    # Default to 1 if we see something but it's low
    if height_mm > 10.0 and num_cubes == 0:
        num_cubes = 1

    pick_z = constants.PICK_Z_OFFSET + (max(0, num_cubes - 1)) * constants.CUBE_HEIGHT
    return num_cubes, round(height_mm, 1), pick_z

if __name__ == "__main__":
    pipe, aln, flts = initialize_realsense()
    print("Press 'q' to stop. Center the cubes under the crosshair.")
    
    try:
        while True:
            depth, color = get_frames(pipe, aln, flts)
            
            # Sample center of screen (320, 240)
            count, raw_h, z_val = get_stack_info(depth, 320, 240)
            
            # UI Overlay
            cv2.drawMarker(color, (320, 240), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(color, f"Height: {raw_h}mm", (30, 60), 1, 1.5, (255, 255, 0), 2)
            cv2.putText(color, f"Cubes: {count}", (30, 100), 1, 2, (0, 255, 0), 3)
            cv2.putText(color, f"Arm Z: {z_val}mm", (30, 140), 1, 1.5, (255, 0, 255), 2)
            
            cv2.imshow("Stack Debugger", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()