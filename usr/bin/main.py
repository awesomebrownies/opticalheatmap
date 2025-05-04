#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import time

from threading import Thread, Lock

#Multi-threading tutorial: https://sihabsahariar.medium.com/a-multi-threading-approach-for-faster-opencv-video-streaming-in-python-7324f11dbd2f

#All functions inside are multi-threading camera stream function setups
class CameraStream:
    def __init__(self, stream_id=0):
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=25)
        self.prev_gray = None
        self.lock = Lock()

        self.stream_id = stream_id
        self.vcap = cv.VideoCapture(self.stream_id)
        if not self.vcap.isOpened():
            print(f"Error: Cannot open camera {self.stream_id}")
            exit(0)

        self.grabbed, self.frame = self.vcap.read()
        if not self.grabbed:
            print("Error: Can't read first frame from camera.")
            exit(0)

        self.prev_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        
        self.output_frame = self.frame.copy()
        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.vcap.read()
            if not grabbed:
                print("Error: Failed to grab frame.")
                self.stop()
                break
            
            curr_frame = frame.copy()
            curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
            
            with self.lock:
                prev_gray_local = self.prev_gray.copy()

            #motion detection portion of code
            foreground_mask = self.bg_subtractor.apply(curr_frame)
            _, foreground_mask = cv.threshold(foreground_mask, 20, 255, cv.THRESH_BINARY)
            
            blur_img = cv.GaussianBlur(curr_gray, (3, 3), 0)
            #first determine canny thresholds by finding the average gray value in the image
            average_intensity = np.mean(curr_gray)
            lower_threshold = max(0, (1.0 - 0.33) * average_intensity)
            upper_threshold = min(255, (1.0 + 0.33) * average_intensity)
            edges = cv.Canny(blur_img, lower_threshold, upper_threshold)
            
            #color the edges
            edge_color = np.zeros_like(curr_frame)
            edge_color[edges > 0] = [255, 255, 255]
            
            #abs difference between frames for motion estimation
            diff = cv.absdiff(curr_gray, prev_gray_local)
            _, motion = cv.threshold(diff, 20, 255, cv.THRESH_BINARY)

            #downsampling for performance
            small_diff = cv.resize(motion, (curr_frame.shape[1]//4, curr_frame.shape[0]//4))
            small_mask = cv.resize(foreground_mask, (curr_frame.shape[1]//4, curr_frame.shape[0]//4))
            
            #upsample for display
            display_mask = cv.resize(small_mask, (curr_frame.shape[1], curr_frame.shape[0]))
            display_diff = cv.resize(small_diff, (curr_frame.shape[1], curr_frame.shape[0]))
            
            #start with the original frame as base
            combined_output = np.zeros_like(curr_frame)#curr_frame.copy()
            
            #apply motion color mapping - yellow for slow motion, red for faster
            height, width = curr_frame.shape[:2]
            for y in range(0, height, 4):
                for x in range(0, width, 4):
                    if display_mask[y, x] > 0:
                        intensity = display_diff[y, x]
                        color = max(0, 255 - intensity)
                        
                        combined_output[y, x] = [0, color, 255]
            
            alpha = 0
            mask = edges > 0
            if np.any(mask):  #check for valid pixels
                try:
                    #extract masked regions first to verify they're valid
                    frame_region = combined_output[mask]
                    edge_region = edge_color[mask]
                    
                    #only proceed if both arrays are valid and have the same shape
                    if frame_region is not None and edge_region is not None and frame_region.shape == edge_region.shape:
                        combined_output[mask] = cv.addWeighted(frame_region, alpha, edge_region, 1-alpha, 0)
                except:
                    pass

            self.frame_count += 1
            current_time = time.time()
            if (current_time - self.last_time) >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.last_time = current_time
                self.frame_count = 0

            cv.putText(combined_output, f"{self.fps:.2f} FPS", (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

            with self.lock:
                self.prev_gray = curr_gray.copy()
                self.output_frame = combined_output.copy()

    def read(self):
        with self.lock:
            return self.output_frame.copy()

    def stop(self):
        self.stopped = True
        if self.t.is_alive():
            self.t.join()
        self.vcap.release()


if __name__ == "__main__":
    camera_stream = CameraStream(stream_id=0)
    camera_stream.start()
    
    num_frames_processed = 0
    start = time.time()
    
    try:
        while True:
            if camera_stream.stopped:
                break
                
            frame = camera_stream.read()
            
            #final display method
            cv.imshow('Motion and Edge Detection', frame)
            
            num_frames_processed += 1
            
            key = cv.waitKey(1)
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        end = time.time()
        camera_stream.stop()
        elapsed = end - start
        fps = num_frames_processed / elapsed
        print("FPS: {:.2f}, Elapsed Time: {:.2f}s, Frames Processed: {}".format(fps, elapsed, num_frames_processed))
        cv.destroyAllWindows()
