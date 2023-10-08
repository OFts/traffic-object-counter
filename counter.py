import cv2
import numpy as np
from object_detection import ObjectDetection
from fps_handler import FPSHandler
import math

od = ObjectDetection()

cap  = cv2.VideoCapture("cofal.mp4")

fps_handler = FPSHandler()  # Initialize FPS controller
record_video = False

# Save frames as video file if record_video is True
if record_video:
  # Define output video specifications
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # To create mp4 video
  fps = 30.0
  frame_size = (int(cap.get(3)), int(cap.get(4)))
  out = cv2.VideoWriter('vehicles_count.mp4', fourcc, fps, frame_size) 

# Signature
dev_name = "Dev by Oscar Fts"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
name_color = (255, 255, 255)

# Get dimensions
ret, frame = cap.read()
height, width = frame.shape[:2]
# Get text dimensions
(name_width, name_height), _ = cv2.getTextSize(dev_name, font, font_scale, 2)
# Corner position
name_x = width - name_width - 10
name_y = height - 5

# --------------------------------- Variables -------------------------------- #

# Load class names from the 'dnn_model/classes.txt' file (adjust the path if necessary).
with open('dnn_model\classes.txt', 'r') as f:
  classes = f.read().strip().split('\n')

# Define a list of random colors for the classes
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8').tolist()

# Initial position for printing text
x_count, y_count = 10, 30
text_height = 25

# Number of frames to process
last_frame = 600

# Frame counter
count = 0

# Variables initialization
cp_pre_frame = []
tracking_objects = {}
crossing_count = {}
track_id = 0

# Limit line
x1 = 450
y1 = 150
x2 = 1250
y2 = 350
a = x1 - x2
b = y2 - y1
c = (y1 * a) + (x1 * b)

# --------------------------------- Functions -------------------------------- #
# Line cross detection function 
def cross_line(pp, cp):
  d_prev =pp[1] * a + pp[0] * b
  d = cp[1] * a + cp[0] * b
  #print(d)
  # Determina si el centroide cruzó la línea
  if d_prev <= c and d > c: # Going up
      return True
  if d_prev >= c and d < c: # Going down
      return True
  return False # Not crossing

# ----------------------------------- Loop ----------------------------------- #

# Main loop
while True:
  ret, frame = cap.read()
  count += 1
  if not ret:
    break

  # Center point current frame
  cp_crnt_frame = []
  
  # Limit line color
  lim_line_color = (0, 0, 255)
  
  (class_ids, scores, boxes) = od.detect(frame)
  
  for idx, box in enumerate(boxes):
    (x, y, w, h) = box
    
    cx = int((x + x + w) / 2.0)
    cy = int((y + y + h) / 2.0)
    
    class_id = class_ids[idx]
    confidence = scores[idx]
    
    cp_crnt_frame.append((cx, cy, class_id))

    # Get color and label for this class
    color = colors[class_id]
    label = "{}: {:.2f}".format(classes[class_id], confidence)
    
    # Calculates text size
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    label_height = label_size[1] + 10  # añade un margen de 10 pixels

    # Draw a rectangle background with a margin
    cv2.rectangle(frame, (x, y - label_height), (x + label_size[0], y), color, -1)

    # Draw the text on top of the rectangle (set the position and color to black)
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Draw the rectangle around the object
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw the centroid
    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
  
  if count <= 2:
    for pt in cp_crnt_frame:
      for pt2 in cp_pre_frame:
        distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
        if distance < 20:
          tracking_objects[track_id] = pt
          track_id += 1
  else:
    tracking_objects_copy = tracking_objects.copy()
    cp_crnt_frame_copy = cp_crnt_frame.copy()
    for object_id, pt2 in tracking_objects_copy.items():
      object_exists = False
      for pt in cp_crnt_frame_copy:
        distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
        if distance < 20:
          tracking_objects[object_id] = pt
          object_exists = True
          if cross_line(pt2, pt):
            #print("cross")
            lim_line_color = (0, 255, 255)
            if pt[2] == 1:
              class_id = 3
            else:
              class_id = pt[2]
            try:
              crossing_count[class_id]
            except:
              crossing_count[class_id] = 1
            else:
              crossing_count[class_id] += 1
          if pt in cp_crnt_frame:
            cp_crnt_frame.remove(pt)
          continue
      if not object_exists:
        tracking_objects.pop(object_id)
    for pt in cp_crnt_frame:
      tracking_objects[track_id] = pt # New object
      track_id += 1
  
  # Add object count to upper left corner
  y_count = 30
  for class_id, quantity in crossing_count.items():
    text = f"{classes[class_id]}: {quantity}"
    # Calcula el ancho del texto
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    
    # Draw the black rectangle in the background
    cv2.rectangle(frame, (x_count, y_count - text_height), (x_count + text_width, y_count), (0, 0, 0), -1)
    
    # Draw the white text
    cv2.putText(frame, text, (x_count, y_count - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    y_count += 30  # Aumenta la posición en y para que los textos no se superpongan

  # Limit line
  cv2.line(frame, (x1, y1), (x2, y2), lim_line_color, 2)
  
  # Signature
  cv2.rectangle(frame, (name_x - 5, name_y - name_height - 10), (name_x + name_width + 5, name_y), (0, 0, 0), -1)
  cv2.putText(frame, dev_name, (name_x, name_y - 5), font, font_scale, name_color, 2)
  
  # FPS
  fps_handler.update()
  fps_handler.draw_fps(frame)  # Draw FPS in a rectangle
  
  # for object_id, pt in tracking_objects.items():
  #   cv2.circle(frame, pt[0:2], 20, (0, 255, 0), 2)
  #   cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 255, 0), 2)
  
  #print("Tracking objects")
  #print(tracking_objects)
  print("Frame:", count, crossing_count)
  
  if record_video:
    # Write the frame to the output video
    out.write(frame)
  
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1)
  if key == 27 or count == last_frame:  # Si se presiona la tecla 'Esc', termina el bucle
    break

cap.release()
cv2.destroyAllWindows()