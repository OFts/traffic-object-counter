import cv2
import numpy as np
from object_detection import ObjectDetection

od = ObjectDetection()

cap  = cv2.VideoCapture("cofal.mp4")

# Cargar los nombres de las clases desde el archivo 'coco.names' (ajusta la ruta si es necesario)
with open('dnn_model\classes.txt', 'r') as f:
  classes = f.read().strip().split('\n')

# Definir una lista de colores aleatorios para las clases
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8').tolist()

while True:
  _, frame = cap.read()

  (class_ids, scores, boxes) = od.detect(frame)

  for idx, box in enumerate(boxes):
    (x, y, w, h) = box
    class_id = class_ids[idx]
    confidence = scores[idx]

    # Obtener color y etiqueta para esta clase
    color = colors[class_id]
    label = "{}: {:.2f}".format(classes[class_id], confidence)
    
    # Calcula el tamaño del texto
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    label_height = label_size[1] + 10  # añade un margen de 10 pixels

    # Dibuja un rectángulo de fondo con un margen
    cv2.rectangle(frame, (x, y - label_height), (x + label_size[0], y), color, -1)

    # Dibuja el texto encima del rectángulo (ajusta la posición y el color a negro)
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Dibuja el rectángulo alrededor del objeto
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1)
  if key == 27:  # Si se presiona la tecla 'Esc', termina el bucle
    break

cap.release()
cv2.destroyAllWindows()