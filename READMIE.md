# Traffic Object Counter

This is a simple application for the detection and counting of objects crossing a defined line in a recording of vehicular traffic. The main code to run is `counter.py`. This code was made for a 1280px wide, 720px high recording at 30 frames per second.

## How to use
This code makes use of the following libraries:
- numpy
- OpenCV (compiled for CUDA 11.8)
- math
- YOLOv4

## Improvement opportunities
To improve the performance of the algorithm there are different tasks that can be performed.
- Use an algorithm such as SORT or DeepSORT, for tracking hidden or occluded objects.
- Improve the algorithm for counting vehicles crossing the line (use a solution like Shapely).
- Implement a Region of Interest (ROI) to speed up object detection.
- Implement a more updated version of YOLO.