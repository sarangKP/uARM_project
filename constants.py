# ~/UARM_Intern/constants.py

# --- Hardware & Serial ---
PORT               = '/dev/ttyACM0'
ARM_SPEED          = 20000  

# --- Vision & AI ---
MODEL_PATH         = '/home/user/UARM_Intern/best.pt'
CONF_THRESHOLD     = 0.6    # Only detect objects with >60% confidence
DEPTH_MIN          = 0.3    # RealSense clipping (meters)
DEPTH_MAX          = 1.5    # RealSense clipping (meters)

# --- Kinematic Safety Boundaries ---
MIN_RADIUS         = 120.0  # Inner deadzone to prevent motor strain (mm)
MAX_RADIUS         = 320.0  # Maximum physical reach of the arm (mm)
TABLE_Z            = -3.37  # Absolute floor. Gripper will never go below this (mm).
APPROACH_Z         = 80.0   # Hover height while moving across the table (mm)

# --- Calibration Offsets (Hand-Eye TCP) ---
# Physical distance from the absolute tip of the closed gripper to the center of the ArUco marker
MARKER_OFFSET_X    = 30.0   
MARKER_OFFSET_Y    = 0.0    
MARKER_OFFSET_Z    = 80.0   

# --- Timing & Routing ---
PICK_DELAY         = 1.0    # Seconds to wait after closing gripper
PLACE_DELAY        = 0.5    # Seconds to wait after opening gripper

BIN_POSITIONS = {
    'red':    {'x': 200, 'y': -150, 'z': 50},
    'blue':   {'x': 200, 'y': 150,  'z': 50},
    'yellow': {'x': 150, 'y': -150, 'z': 50},
    'orange': {'x': 150, 'y': 150,  'z': 50}
}
