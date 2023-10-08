import cv2
import time

class FPSHandler:
    def __init__(self):
        self.fps_start_time = time.time()
        self.fps = 0
        self.frame_count = 0

    def update(self):
        self.frame_count += 1
        fps_end_time = time.time()
        time_diff = fps_end_time - self.fps_start_time

        if time_diff > 1:
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.fps_start_time = fps_end_time

        return self.fps

    def draw_fps(self, frame, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(0, 0, 255), thickness=2):
        cv2.putText(frame, "FPS: {:.2f}".format(self.fps), position, font, scale, color, thickness)