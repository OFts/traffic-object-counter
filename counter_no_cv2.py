import cv2
import numpy as np
from object_detection import ObjectDetection
import math

od = ObjectDetection()

cap  = cv2.VideoCapture("cofal.mp4")

# Cargar los nombres de las clases desde el archivo 'coco.names' (ajusta la ruta si es necesario)
with open('dnn_model\classes.txt', 'r') as f:
  classes = f.read().strip().split('\n')

# Definir una lista de colores aleatorios para las clases
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8').tolist()

# Posición inicial para imprimir el texto
x_count, y_count = 10, 30
text_height = 25

# Frame counter
count = 0

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

# Main loop
while True:
  ret, frame = cap.read()
  count += 1
  if not ret:
    break

  # Point current frame
  cp_crnt_frame = []
  
  (class_ids, scores, boxes) = od.detect(frame)

  
  for idx, box in enumerate(boxes):
    (x, y, w, h) = box
    
    cx = int((x + x + w) / 2.0)
    cy = int((y + y + h) / 2.0)
    
    class_id = class_ids[idx]    
    cp_crnt_frame.append((cx, cy, class_id))

  
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
          #print(cross_line(pt2, pt))
          if cross_line(pt2, pt):
            #print("cross")
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
  
  
  # for object_id, pt in tracking_objects.items():
  #   cv2.circle(frame, pt[0:2], 20, (0, 255, 0), 2)
  #   cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 255, 0), 2)
  
  #print("Tracking objects")
  #print(tracking_objects)
  print("Frame:", count, crossing_count)


cap.release()
cv2.destroyAllWindows()