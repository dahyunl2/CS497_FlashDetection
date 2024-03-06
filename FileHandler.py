# Open the video file
import os
import sys
import cv2
import numpy as np
from collections import deque
import DangerDetection

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from red_transition_fsm import ChromaticityChecker, State, Region, Buffer
import numpy as np
def filehandler(filename, speed):
    hertz = 3
    if speed > 5 or speed < 2e-1:
      raise ValueError("speed must not exceed 5x and must be positive")
    flash_seconds = 0
    # get file and frame data
    cap = cv2.VideoCapture(filename)

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # frames of a second of video
    frames_per_second = int(frame_rate * speed)

    # sliding window array which accounts for a second of visual data
    dangerous = np.zeros((frames_per_second, frame_height, frame_width, 3), dtype=np.uint8)
    frame_buffer = deque(maxlen=frames_per_second)
    frame_buffer_red = Buffer(4,4)
    # if skipping a second to optimize
    skip = 0
    frame_counter = 0
    start_danger = -1
    # last_danger = -1

    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(frame_counter, frame_counter / frames_per_second)
            print("done")
            break

        # Convert from BGR to HLS
        hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)


        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tristimulus_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        # U = np.zeros(frame_rgb.shape[:2])
        # V = np.zeros(frame_rgb.shape[:2])
        # Rperc = np.zeros(frame_rgb.shape[:2])
        chromacityRerc=np.zeros(frame_rgb.shape[:2])
        for i in range(frame_rgb.shape[0]):
            for j in range(frame_rgb.shape[1]):
                b = np.dot(tristimulus_matrix, frame_rgb[i,j])
                d = (b[0] + 15 * b[1] + 3 * b[2])
                uv=(4 * b[0] / d, 9 * b[1] / d)
                cTotal = np.sum(frame_rgb[i,j])
                rperc = 0 if cTotal == 0 else frame_rgb[i,j,0] / cTotal
                chromacityRerc[i][j]=(uv, rperc)
                #((U,V),Rper)
   
        #Add the currecnt frame to the buffer for red detection
        frame_buffer_red.add_frame(chromacityRerc)
        
        # Add the current frame to the buffer
        frame_buffer.append(hls_frame)

        

        # Skip a second of frames
        # if skip > 0:
        #     skip -= 1
        #     frame_buffer.popleft()
        #     continue

        # Check if we have enough frames for the sliding window
        # print(frame_buffer, frames_per_second)
        if len(frame_buffer) == frames_per_second:
            # Fill the 'dangerous' array with the frames from the buffer
            for i, buf_frame in enumerate(frame_buffer):
                dangerous[i] = buf_frame

            # Process the 'dangerous' array
            flashes = process_dangerous(dangerous, frame_rate)
            if flashes >= hertz and start_danger == -1:
                start_danger = frame_counter
            if flashes < hertz:
                if start_danger >= 0:
                    timestamps.append([start_danger / frame_rate, frame_counter / frame_rate])
                    #print("danger from", start_danger / frames_per_second, "seconds to", frame_counter / frames_per_second, "seconds, frames", start_danger, frame_counter)
                    start_danger = -1
                    #last_danger = frame_counter
                #skip = frames_per_second
            frame_buffer.popleft()

            # print("number of flashes occured is" + str(flashes))
            #print(f"Processing window starting at frame {cap.get(cv2.CAP_PROP_POS_FRAMES) - frames_per_half_second}")
        frame_counter += 1
    cap.release()

    #timestamp merge: Detection of flashes occurs within half-second windows so we want to merge what's close together
    idx = 0
    while idx < len(timestamps):
      stamp = timestamps[idx]
      if idx + 1 == len(timestamps):
        break
      next = timestamps[idx + 1]
      if abs(stamp[1] - next[0]) < .5:
        stamp[1] = next[1]
        timestamps.remove(next)
      else:
        idx += 1

    for st in timestamps:
      print("flashing from", st[0], "to", st[1])