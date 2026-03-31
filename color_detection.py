import cv2
from ultralytics import YOLO
import constants

# Initialize model globally to avoid reloading every frame
model = YOLO(constants.MODEL_PATH)

def get_detections(color_image):
    """
    Runs inference and returns a list of detected cubes.
    Output format: [{'color': 'red', 'center': (u, v), 'box': [x1, y1, x2, y2]}, ...]
    """
    results = model(color_image, conf=constants.CONF_THRESHOLD, verbose=False)
    detections = []
    
    for r in results:
        for box in r.boxes:
            # Extract coordinates and class
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
            label = model.names[int(box.cls[0])].lower()
            
            detections.append({
                'color': label,
                'center': (u, v),
                'box': [int(x1), int(y1), int(x2), int(y2)]
            })
            
    return detections

# Test Snippet
if __name__ == "__main__":
    from depth_utils import initialize_realsense, get_frames
    pipe, aln, flts = initialize_realsense()
    try:
        while True:
            _, color = get_frames(pipe, aln, flts)
            items = get_detections(color)
            for item in items:
                u, v = item['center']
                cv2.rectangle(color, (item['box'][0], item['box'][1]), (item['box'][2], item['box'][3]), (0, 255, 0), 2)
                cv2.putText(color, item['color'], (u, v-10), 1, 1.5, (0, 255, 0), 2)
            cv2.imshow("AI Detection Test", color)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        pipe.stop()