import cv2
import numpy as np
import time
from ultralytics import YOLO


model = YOLO('yolov10n.pt')  

# Define the SOURCE points calculated from the 'calculate_cordinates'
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]], dtype=np.float32)

# target points defined
TARGET_WIDTH = 25 
TARGET_HEIGHT = 250  

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ],
    dtype=np.float32,
)
#apply perspective transform between source and target
M = cv2.getPerspectiveTransform(SOURCE, TARGET)

# Store vehicle positions across frames
vehicle_positions = {}
vehicle_speeds = {}
vehicle_last_update = {}


cap = cv2.VideoCapture("vehicles.mp4")  

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_time = 1 / fps  # Time between frames

resize_scale = 0.5  
output_width = int(frame_width * resize_scale)
output_height = int(frame_height * resize_scale)


fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter("output.mp4", fourcc, fps, (output_width, output_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply perspective transform
    warped_frame = cv2.warpPerspective(frame, M, (TARGET_WIDTH, TARGET_HEIGHT))
  
    frame = cv2.resize(frame, (output_width, output_height))

    
    results = model(frame.copy())  

    detected_vehicles = {}
    for r in results:
        for box in r.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            if int(cls) in [2, 3, 5, 7]:#YOLOV10- cars trucks motorcycles and buses
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2  
                detected_vehicles[(center_x, center_y)] = (x1, y1, x2, y2)

    # match the current detection with past detections using the id's
    for (center_x, center_y), (x1, y1, x2, y2) in detected_vehicles.items():
        closest_id = None
        min_dist = float("inf")

        for track_id, (prev_x, prev_y, prev_time) in vehicle_positions.items():
            distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)

            if distance < min_dist and distance < 50:  
                min_dist = distance
                closest_id = track_id

        if closest_id is not None:
            prev_x, prev_y, prev_time = vehicle_positions[closest_id]
            distance_px = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)

            
            real_distance = (distance_px / frame_height) * 250 
            speed_kmh = (real_distance / frame_time) * 3.6  
            
            #make the speed of the vehicle constant for some time for less fluctuations
            current_time = time.time()
            if closest_id in vehicle_last_update:
                time_diff = current_time - vehicle_last_update[closest_id]
                if time_diff < 2:
                    speed_kmh = (vehicle_speeds[closest_id] * 0.9) + (speed_kmh * 0.1)  
                else:
                    speed_kmh = (vehicle_speeds[closest_id] * 0.6) + (speed_kmh * 0.4)  

            
            vehicle_speeds[closest_id] = speed_kmh
            vehicle_last_update[closest_id] = current_time
            
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {closest_id}: {int(speed_kmh)} km/h", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

           
            vehicle_positions[closest_id] = (center_x, center_y, current_time)
        else:
           
            new_id = len(vehicle_positions) + 1
            vehicle_positions[new_id] = (center_x, center_y, time.time())
            vehicle_speeds[new_id] = 0  
            vehicle_last_update[new_id] = time.time()

    
    out.write(frame)

    
    cv2.imshow("Speed Detection", cv2.resize(frame, (1280, 720)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
