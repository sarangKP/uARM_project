# ~/UARM_Intern/constants.py

PORT               = '/dev/ttyACM0'
CAMERA_HEIGHT      = 550.0  
DEPTH_MIN          = 0.3    
DEPTH_MAX          = 1.5    
TABLE_DEPTH_MM     = 581.0  # <--- Updated to your measured value

# Arm Physicals
TABLE_Z            = -3.37  
PICK_Z_OFFSET      = 3.07   # The Z for a single cube
CUBE_HEIGHT        = 25.0
APPROACH_Z         = 80.0


MODEL_PATH = '/home/user/UARM_Intern/best.pt'
CONF_THRESHOLD = 0.6  # Only detect objects with >60% confidence


# ~/UARM_Intern/constants.py (Add these)
BIN_POSITIONS = {
    'red':    {'x': 200, 'y': -150, 'z': 50},
    'blue':   {'x': 200, 'y': 150,  'z': 50},
    'yellow': {'x': 150, 'y': -150, 'z': 50},
    'orange': {'x': 150, 'y': 150,  'z': 50}
}

# ~/UARM_Intern/constants.py

# ... previous constants ...
ARM_SPEED          = 20000  
PICK_DELAY         = 1.0    # Seconds to wait after closing gripper
PLACE_DELAY        = 0.5    # Seconds to wait after opening gripper