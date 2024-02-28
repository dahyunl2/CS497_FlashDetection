# Open the video file
import cv2
import numpy as np
from collections import deque
import DangerDetection

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