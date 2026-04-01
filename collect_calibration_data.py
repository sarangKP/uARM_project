import time
import numpy as np
import cv2
import threading
import constants
from uarm.wrapper import SwiftAPI
from depth_utils import initialize_oak
from aruco_test import detect_aruco_3d

# Globals shared between the vision thread and the arm thread
latest_marker_3d  = None
collection_status = "Initializing..."
is_running        = True


def arm_routine(arm, calibration_points, arm_pts_collected, cam_pts_collected):
    global collection_status, latest_marker_3d, is_running

    try:
        for i, target_pos in enumerate(calibration_points):
            if not is_running:
                break

            collection_status = f"Moving to {i + 1}/{len(calibration_points)}"
            arm.set_position(x=target_pos[0], y=target_pos[1], z=target_pos[2], speed=10000, wait=True)

            collection_status = "Stabilizing..."
            time.sleep(1.0)

            collection_status = "Capturing..."
            time.sleep(0.5)

            if latest_marker_3d is not None:
                cam_mm = [val * 1000.0 for val in latest_marker_3d]
                actual_marker_pos = (
                    target_pos[0] + constants.MARKER_OFFSET_X,
                    target_pos[1] + constants.MARKER_OFFSET_Y,
                    target_pos[2] + constants.MARKER_OFFSET_Z,
                )

                arm_pts_collected.append(actual_marker_pos)
                cam_pts_collected.append(cam_mm)
                print(f"  SUCCESS ({i + 1}): arm={actual_marker_pos}  cam={cam_mm}")

                # --- STEP 4.5: IMMEDIATE SAVE (The Fix) ---
                np.savez(
                    "calibration_data.npz",
                    arm_pts=np.array(arm_pts_collected, dtype=np.float32),
                    cam_pts=np.array(cam_pts_collected, dtype=np.float32),
                )
                # ------------------------------------------
            else:
                print(f"  FAILED  ({i + 1}): marker not visible at {target_pos}")

        collection_status = f"Finished! {len(arm_pts_collected)} pairs secured."

    except Exception as e:
        print(f"\n[CRASH DETECTED] Saving current data before exit: {e}")
    finally:
        collection_status = "Returning home..."
        arm.set_position(x=150, y=0, z=150, wait=True)
        is_running = False


def run_live_collection():
    """
    Main thread: runs the OAK-D Lite feed, tracks the ArUco marker,
    and feeds depth data to the arm thread via latest_marker_3d.
    """
    global latest_marker_3d, collection_status, is_running

    arm = SwiftAPI(port=constants.PORT)
    arm.waiting_ready()
    arm.set_mode(0)

    # --- OAK-D Lite initialisation (replaces initialize_realsense) ---
    device, q_rgb, q_depth, intrinsics = initialize_oak()

    # Calibration grid:
    # 3×3 XY grid at Z=50, same 4 corners at Z=120, plus one high centre.
    # Added two low points at Z=20 to anchor the matrix in the pick zone
    # (objects sit between Z=0 and Z=50, so this removes extrapolation risk).
    calibration_points = [
        # Z = 50 mm layer (main workspace)
        (150, -100, 50), (200, -100, 50), (250, -100, 50),
        (150,    0, 50), (200,    0, 50), (250,    0, 50),
        (150,  100, 50), (200,  100, 50), (250,  100, 50),
        # Z = 120 mm layer (adds vertical span)
        (150, -100, 120), (250, -100, 120),
        (150,  100, 120), (250,  100, 120),
        # Z = 150 mm (top anchor)
        (200, 0, 150),
        # Z = 20 mm (low anchor — covers actual pick heights)
        (150, -100, 20), (250, 100, 20),
    ]

    arm_pts_collected = []
    cam_pts_collected = []

    arm_thread = threading.Thread(
        target=arm_routine,
        args=(arm, calibration_points, arm_pts_collected, cam_pts_collected),
    )
    arm_thread.start()

    print("\n--- LIVE CALIBRATION STARTED ---")
    print("Press Q to emergency-stop.")

    try:
        while is_running:
            # Get latest frames from OAK
            color_frame = q_rgb.get().getCvFrame()       # BGR numpy
            depth_frame = q_depth.get().getFrame()       # uint16 numpy (mm)

            # Detect ArUco and update shared 3D position
            # detect_aruco_3d draws the bounding box onto color_frame in-place
            latest_marker_3d, _ = detect_aruco_3d(
                color_frame, depth_frame, intrinsics
            )

            # UI overlays
            cv2.putText(color_frame, f"Status: {collection_status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(color_frame, f"Points: {len(arm_pts_collected)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Live Calibration", color_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nEmergency stop triggered.")
                is_running = False
                break

    finally:
        is_running = False
        arm_thread.join()
        device.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_collection()