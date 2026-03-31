import cv2
import cv2.aruco as aruco
import numpy as np
from depth_utils import initialize_realsense, get_3d_camera_coordinate

# Initialize ArUco Dictionary (4x4 grid, looking for up to 50 unique IDs)
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters()

def detect_aruco_3d(color_image, depth_frame, intrinsics):
    """Detects an ArUco marker and returns its 3D coordinate."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    
    if ids is not None and len(ids) > 0:
        # Get corners of the first detected marker
        c = corners[0][0]
        
        # Calculate the center pixel
        center_x = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
        center_y = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
        
        # Draw bounding box and center crosshair for UI
        cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
        cv2.drawMarker(color_image, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
        
        # Extract 3D point using our previous module
        pt_3d = get_3d_camera_coordinate(depth_frame, center_x, center_y, intrinsics)
        return pt_3d, (center_x, center_y)
        
    return None, None

if __name__ == "__main__":
    pipe, aln, flts, intrinsics = initialize_realsense()
    print("Looking for ArUco 4x4 Marker...")
    
    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = aln.process(frames)
            
            # Safe casting for depth
            depth_frame = aligned.get_depth_frame()
            if hasattr(depth_frame, 'as_depth_frame'):
                depth_frame = depth_frame.as_depth_frame()
                
            color_image = np.asanyarray(aligned.get_color_frame().get_data())
            
            # Run Detection
            pt_3d, center_px = detect_aruco_3d(color_image, depth_frame, intrinsics)
            
            if pt_3d:
                # Format to millimeters
                x, y, z = [round(val * 1000, 1) for val in pt_3d]
                cv2.putText(color_image, f"Marker 3D (mm): X:{x} Y:{y} Z:{z}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(color_image, "No Marker Detected", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("ArUco 3D Tracker", color_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()