import pyrealsense2 as rs
import numpy as np
import cv2
import constants

def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Filter Setup
    filters = [
        rs.threshold_filter(constants.DEPTH_MIN, constants.DEPTH_MAX),
        rs.spatial_filter(),
        rs.temporal_filter(),
        rs.hole_filling_filter()
    ]
    return pipeline, align, filters

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

if __name__ == "__main__":
    pipe, aln, flts = initialize_realsense()
    try:
        while True:
            depth, color = get_frames(pipe, aln, flts)
            # Get depth at center pixel
            center_depth = depth[240, 320]
            
            # Visual feedback
            cv2.putText(color, f"Center Depth: {center_depth}mm", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("RealSense Test", color)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()