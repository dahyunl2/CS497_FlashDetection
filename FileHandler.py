# Open the video file
import cv2
import numpy as np
from collections import deque
import DangerDetection

def filehandler(filename, speed):
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

    while cap.isOpened():
        count = 0
        ret, frame = cap.read()
        if not ret:
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
   


    # Add the current frame to the buffer
    frame_buffer.append(hls_frame)

    # Check if we have enough frames for the sliding window
    if len(frame_buffer) == frames_per_second:
        # Fill the 'dangerous' array with the frames from the buffer
        for i, buf_frame in enumerate(frame_buffer):
            dangerous[i] = buf_frame

        # Process the 'dangerous' array
        dangerous_segments = DangerDetection.process_dangerous(dangerous, frame_rate)
        if count % 14 == 0 and len(dangerous_segments) > 0:
            print(dangerous_segments)
        count += 1
        #print(f"Processing window starting at frame {cap.get(cv2.CAP_PROP_POS_FRAMES) - frames_per_half_second}")


    cap.release()
    return chromacityRerc
