import cv2
import numpy as np


image_path = "Screenshot 2025-03-05 212606.png"  
image = cv2.imread(image_path)


resize_factor = 0.5  
resized_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)

points = []
def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
      
        original_x = int(x / resize_factor)
        original_y = int(y / resize_factor)
        points.append((original_x, original_y))
        
        print(f"Point {len(points)}: {original_x, original_y}")
        
        
        cv2.circle(resized_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", resized_image)


cv2.imshow("Select Points", resized_image)
cv2.setMouseCallback("Select Points", get_points)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Selected Points:", points)
