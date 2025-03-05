# Vehicle Speed Detection Using YOLOv10

## Overview
This project detects and tracks vehicles in a video using the YOLOv10 object detection model. It calculates the speed of vehicles using perspective transformation and real-world scaling.

## Features
- Detects cars, trucks, motorcycles, and buses using YOLOv10.
- Applies a perspective transformation to map pixel distances to real-world distances.
- Tracks vehicles across frames and calculates their speed in km/h.
- Generates an output video with bounding boxes and speed annotations.

## Installation
### Requirements
Ensure you have Python and the following dependencies installed:

```bash
pip install -r requirements.txt
```

### Running the Project
```bash
python detection.py
```

## Mathematical Calculations for Perspective Transform
### Source and Target Points
The perspective transform maps four points in the video (SOURCE) to a real-world coordinate system (TARGET).

- **SOURCE points** (measured manually from the input video):
  ```
  SOURCE = np.array([[1250, 780], [2300, 800], [5000, 2159], [-525, 2159]], dtype=np.float32)
  ```
- **TARGET points** (real-world equivalent based on known road dimensions):
  ```
  TARGET_WIDTH = 25  # Road width in meters
  TARGET_HEIGHT = 250  # Road length in meters
  TARGET = np.array(
      [
          [0, 0],
          [TARGET_WIDTH - 1, 0],
          [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
          [0, TARGET_HEIGHT - 1],
      ],
      dtype=np.float32,
  )
  ```
- The transformation matrix is computed using:
  ```python
  M = cv2.getPerspectiveTransform(SOURCE, TARGET)
  ```

### Speed Calculation
Speed is estimated using the change in pixel position of the vehicle between frames and then converting it into real-world distance.

#### Step 1: Compute the pixel distance moved by the vehicle
```python
distance_px = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
```

#### Step 2: Convert pixel distance to real-world distance (meters)
```python
real_distance = (distance_px / frame_height) * TARGET_HEIGHT
```

#### Step 3: Calculate speed in km/h
```python
speed_kmh = (real_distance / frame_time) * 3.6
```
where:
- `frame_time = 1 / fps` is the time between two frames.
- `3.6` is the conversion factor from m/s to km/h.

## Video Files
- **Input Video:** `vehicles.mp4`
- **Output Video:** `output.mp4`

To include the videos in GitHub, upload them to the repository and reference them in the README:
```markdown
![Original Video](vehicles.mp4)
![Processed Output](output.mp4)
```
Alternatively, if GitHub doesn’t support video previews, you can host them elsewhere and link them in the README.

## Acknowledgments
- YOLOv10 model from Ultralytics
- OpenCV for computer vision processing



https://github.com/user-attachments/assets/99980ffc-8b97-4c20-9b74-6fad9e1fb10f

