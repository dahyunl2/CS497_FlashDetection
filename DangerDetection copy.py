import numpy as np

def calculate_average_luminance(segment):
    # Luminance is in the Lightness channel (index 1) of HLS
    return np.mean(segment[:, :, 1])

# Function to process the dangerous array
def process_dangerous(dangerous, frame_rate):
    num_frames = dangerous.shape[0]
    height, width = dangerous.shape[1], dangerous.shape[2]
    segment_height = height // 4
    segment_width = width // 4
    luminance_threshold = 10

    # Initialize luminance tracking and timestamps for each segment
    previous_luminance = np.zeros((4, 4))
    segment_changes = {(i, j): [] for i in range(4) for j in range(4)}

    for frame_idx in range(num_frames):
        for row in range(4):
            for col in range(4):
                y1, y2 = row * segment_height, (row + 1) * segment_height
                x1, x2 = col * segment_width, (col + 1) * segment_width
                segment = dangerous[frame_idx, y1:y2, x1:x2]
                average_luminance = calculate_average_luminance(segment)

                if abs(average_luminance - previous_luminance[row, col]) > luminance_threshold:
                    timestamp = frame_idx / frame_rate
                    segment_changes[(row, col)].append(timestamp)
                #else if color change in saturated red range -> then make it as a flash    
                previous_luminance[row, col] = average_luminance

    # Merge close timestamps to consider them as a single flashing incident
    for key in segment_changes:
        timestamps = segment_changes[key]
        merged_timestamps = []
        start = None
        for t in timestamps:
            if start is None:
                start = t
            elif t - start > num_frames / frame_rate:  # If the gap is larger than the window size
                merged_timestamps.append((start, t))
                start = None
        if start is not None:
            merged_timestamps.append((start, timestamps[-1]))
        segment_changes[key] = merged_timestamps

    return segment_changes