import pyrealsense2 as rs
import numpy as np
import cv2
import constants

def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # Start pipeline and capture the active profile
    profile = pipeline.start(config)
    
    # Extract the camera intrinsics from the depth stream
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    intrinsics = depth_profile.get_intrinsics()

    align = rs.align(rs.stream.color)

    filters = [
        rs.threshold_filter(constants.DEPTH_MIN, constants.DEPTH_MAX),
        rs.spatial_filter(),
        rs.temporal_filter(),
        rs.hole_filling_filter()
    ]
    
    # Return all 4 necessary objects
    return pipeline, align, filters, intrinsics

def get_frames(pipeline, align, filters):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()

    for f in filters:
        depth_frame = f.process(depth_frame)

    depth_data = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_data, color_image


def get_3d_camera_coordinate(depth_frame, u, v, intrinsics):
    """Converts a 2D pixel and its depth into a 3D point using a Median ROI filter."""
    
    if hasattr(depth_frame, 'as_depth_frame'):
        depth_frame = depth_frame.as_depth_frame()
        
    # --- NEW: Median ROI Filter (5x5 grid) ---
    half_roi = 2 
    
    # Safely clamp boundaries to screen resolution (640x480)
    u_min, u_max = max(0, u - half_roi), min(639, u + half_roi)
    v_min, v_max = max(0, v - half_roi), min(479, v + half_roi)
    
    valid_depths = []
    
    # Iterate through the grid to collect valid depth samples
    for curr_v in range(v_min, v_max + 1):
        for curr_u in range(u_min, u_max + 1):
            dist = depth_frame.get_distance(curr_u, curr_v)
            if dist > 0: # Ignore unmeasured/black pixels
                valid_depths.append(dist)
                
    if not valid_depths:
        return None # Entire grid was invalid
        
    # Calculate the median to eliminate extreme noise spikes
    median_depth = np.median(valid_depths)
    
    # Deproject using the stabilized median depth
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], median_depth)
    
    return point_3d

if __name__ == "__main__":
    pipe, aln, flts, intrinsics = initialize_realsense()
    try:
        while True:
            frames = pipe.wait_for_frames()
            depth_frame = aln.process(frames).get_depth_frame()
            
            # Test center pixel
            pt_3d = get_3d_camera_coordinate(depth_frame, 320, 240, intrinsics)
            print(f"Camera 3D Point (meters): {pt_3d}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        pipe.stop()