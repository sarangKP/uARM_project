import depthai as dai
import numpy as np
import cv2
import constants

# ---------------------------------------------------------------------------
# OAK-D Lite hardware notes (baked into this module):
#
#   CAM_A  = centre RGB sensor  (fixed-focus or auto-focus variant)
#   CAM_B  = left  mono sensor  (OV7251, 640×480 @ up to 200 FPS)
#   CAM_C  = right mono sensor  (OV7251, 640×480 @ up to 200 FPS)
#   Baseline = 75 mm  (wider than D435i 50 mm → better accuracy at range)
#
#   Depth source: PASSIVE stereo — no IR projector.
#   MinZ ≈ 20 cm with extended disparity, ≈ 35 cm without.
#   Ideal working range: 40 cm – 150 cm.
#
#   Deprojection (replaces rs2_deproject_pixel_to_point):
#     X = (u - cx) * Z / fx
#     Y = (v - cy) * Z / fy
#     Z = depth_mm / 1000.0   (OAK depth frame is uint16, millimetres)
#
#   Intrinsics are read live from on-device EEPROM — no hardcoding needed.
# ---------------------------------------------------------------------------

RGB_W, RGB_H = 640, 480

# ---------------------------------------------------------------------------
# Version detection — depthai v2 vs v3 have incompatible pipeline APIs
# ---------------------------------------------------------------------------
_DAI_MAJOR = int(dai.__version__.split(".")[0])
_v3_pipeline_keepalive = None

import depthai as dai
import numpy as np
import cv2
import constants

RGB_W, RGB_H = 640, 480
_pipeline_keepalive = None  # Prevents v3 pipeline from being garbage collected

def build_pipeline(device) -> tuple:
    """
    Builds a strictly v3 DepthAI pipeline for RGB + Subpixel Stereo Depth.
    """
    pipeline = dai.Pipeline(device)
    FPS = 15

    # 1. Setup Camera Nodes
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # 2. Request Outputs (CRITICAL FIX: Request streams before linking)
    out_rgb = cam_rgb.requestOutput((RGB_W, RGB_H), type=dai.ImgFrame.Type.BGR888p, fps=FPS)
    out_left = mono_left.requestOutput((640, 480), type=dai.ImgFrame.Type.GRAY8, fps=FPS)
    out_right = mono_right.requestOutput((640, 480), type=dai.ImgFrame.Type.GRAY8, fps=FPS)

    # 3. Setup Stereo Node (High Precision for Arm Calibration)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(True) # CRITICAL for < 5mm accuracy
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(RGB_W, RGB_H)
    
    # 4. Link Mono Streams to Stereo Node
    out_left.link(stereo.left)
    out_right.link(stereo.right)

    return pipeline, out_rgb, stereo.depth

def initialize_oak() -> tuple:
    """
    Boots the v3 device, starts the pipeline, and extracts intrinsics.
    """
    global _pipeline_keepalive
    
    device = dai.Device()
    pipeline, rgb_source, depth_source = build_pipeline(device)
    
    # Create non-blocking queues
    q_rgb = rgb_source.createOutputQueue(maxSize=1, blocking=False)
    q_depth = depth_source.createOutputQueue(maxSize=1, blocking=False)
    
    pipeline.start()
    _pipeline_keepalive = pipeline # Store globally
    
    intrinsics = get_rgb_intrinsics(device)

    print(f"[OAK-D Lite] Ready (v3 API Strict). "
          f"fx={intrinsics['fx']:.1f} fy={intrinsics['fy']:.1f} "
          f"cx={intrinsics['cx']:.1f} cy={intrinsics['cy']:.1f}")
    
    return device, q_rgb, q_depth, intrinsics

def get_rgb_intrinsics(device) -> dict:
    """
    Reads factory-calibrated RGB intrinsics from on-device EEPROM.
    Scales the matrix to our working resolution (RGB_W x RGB_H).
    """
    calib = device.readCalibration()
    M = np.array(calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A, RGB_W, RGB_H
    ))
    return {
        'fx': M[0][0], 'fy': M[1][1],
        'cx': M[0][2], 'cy': M[1][2],
        'width': RGB_W, 'height': RGB_H,
    }


def deproject_pixel_to_point(u: int, v: int,
                              depth_m: float,
                              intrinsics: dict) -> tuple:
    """
    Standard pinhole back-projection (replaces rs2_deproject_pixel_to_point).

    StereoDepth outputs rectified frames, so no distortion correction needed.

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth_m

    Returns (X, Y, Z) in metres. Convention: +X right, +Y down, +Z into scene.
    """
    X = (u - intrinsics['cx']) * depth_m / intrinsics['fx']
    Y = (v - intrinsics['cy']) * depth_m / intrinsics['fy']
    return (X, Y, depth_m)


# ---------------------------------------------------------------------------
# Adaptive ROI + closest-cluster depth sampler
# ---------------------------------------------------------------------------

_ROI_SCALE    = 0.6   # Use centre 60% of bbox to avoid edge mixed-pixels
_MIN_VALID_PX = 5     # Minimum valid pixels to trust the result
_CLUSTER_BW_MM = 40.0 # Group depths within 40 mm of nearest surface


def get_3d_camera_coordinate(depth_frame_np: np.ndarray,
                              u: int, v: int,
                              intrinsics: dict,
                              bbox: tuple = None) -> tuple | None:
    """
    Returns the 3D camera-space coordinate using a tight center-weighted median.
    Optimized for Passive Stereo (OAK-D) to avoid occlusion latching.
    """
    h, w = depth_frame_np.shape
    depth_min_mm = constants.DEPTH_MIN * 1000.0
    depth_max_mm = constants.DEPTH_MAX * 1000.0

    # 1. Define a tight 5x5 ROI exactly at the center pixel.
    # We ignore the bbox completely to ensure the robot arm never pollutes the data.
    half = 2 
    u_min, u_max = max(0, u - half), min(w - 1, u + half)
    v_min, v_max = max(0, v - half), min(h - 1, v + half)

    # 2. Extract and filter pixels within hardware limits
    roi = depth_frame_np[v_min:v_max + 1, u_min:u_max + 1].flatten()
    valid = roi[(roi > depth_min_mm) & (roi < depth_max_mm)]

    # 3. Require at least 3 valid pixels to avoid random noise spikes
    if valid.size < 3:
        return None

    # 4. Use pure median. Do not use np.min() as it catches stereo noise/occlusions.
    median_depth_m = float(np.median(valid)) / 1000.0

    # 5. Deproject at the exact visual center
    return deproject_pixel_to_point(u, v, median_depth_m, intrinsics)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Initialising OAK-D Lite...")
    device, q_rgb, q_depth, intrinsics = initialize_oak()

    cv2.namedWindow("OAK-D Lite -- depth test")
    last_click = [None]

    def on_mouse(event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            last_click[0] = (u, v)

    cv2.setMouseCallback("OAK-D Lite -- depth test", on_mouse)
    print("Click on any object to get its 3D coordinate. Press Q to quit.")

    try:
        while True:
            color = q_rgb.get().getCvFrame()
            depth = q_depth.get().getFrame()

            if last_click[0]:
                u, v = last_click[0]
                cv2.drawMarker(color, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

                pt_3d = get_3d_camera_coordinate(depth, u, v, intrinsics)

                if pt_3d:
                    x_mm = round(pt_3d[0] * 1000, 1)
                    y_mm = round(pt_3d[1] * 1000, 1)
                    z_mm = round(pt_3d[2] * 1000, 1)
                    label = f"Camera XYZ (mm): {x_mm}, {y_mm}, {z_mm}"
                    cv2.putText(color, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    print(f"({u},{v}) -> X={x_mm} Y={y_mm} Z={z_mm} mm")
                else:
                    cv2.putText(color, "No valid depth (texture/range issue)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    print(f"({u},{v}) -> No valid depth")

            cv2.imshow("OAK-D Lite -- depth test", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        device.close()
        cv2.destroyAllWindows()