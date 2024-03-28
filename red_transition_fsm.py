"""
This module aims to determine areas in which a video has two or more red flashes
within any one-second period.

https://www.w3.org/WAI/WCAG21/Understanding/three-flashes-or-below-threshold.html#dfn-general-flash-and-red-flash-thresholds
Per WCAG, a red flash is a "pair of opposing transitions involving a saturated red."
From WCAG 2.2, a "pair of opposing transitions involving a saturated red...
is a pair of opposing transitions where, one transition is either to or from a state
with a value R/(R + G + B) that is greater than or equal to 0.8,
and the difference between states is more than 0.2 (unitless) in the
CIE 1976 UCS chromaticity diagram." [[ISO_9241-391]]

We will refer to R/(R + G + B) as "red percentage" and each states' value
in the CIE 1976 UCS chromaticity diagram as "chromaticity value."
"""

import numpy as np
import cv2
from collections import deque
import time
import math

class ChromaticityChecker:
    """
    ChromaticityChecker allows us to determine whether a chromaticity change exceeds a threshold.

    Attributes:
        ct (list): List of the chromaticity values which arrived at a given state
        MAX_CHROMATICITY_DIFF (float): The chromaticity value difference in 
        states for there to be an opposing transition

    Methods:
        push(coordinate): Add chromaticity value to the ChromaticityChecker
        is_above_threshold(other_coord): Determines whether there is a change in chromaticity above the threshold
    """

    def __init__(self):
        """
        Initializes a new instance of a ChromaticityChecker.
        """
        ChromaticityChecker.MAX_CHROMATICITY_DIFF = 0.2
        self.coordinates = set()

    def push(self, coordinate):
        """
        Add a chromaticity coordinate to the ChromaticityChecker.

        Args:
            element((float, float)): The (u', v') chromaticity coordinate to add to the ChromaticityChecker
        """
        self.coordinates.add(coordinate)

    def is_above_threshold(self, other_coord):
        """
        Determines whether there is a change in chromaticity above a threshold.

        Args:
            other_coord((float, float)): The (u', v') chromaticity coordinate of a region of a frame
        """
        # print(self.coordinates)
        coordinates_array = np.array(list(self.coordinates))
        # The chromaticity difference is calculated as SQRT( (u'1 - u'2)^2 + (v'1 - v'2)^2 )
        # differences = []
        # for u, v in self.coordinates:
        #     print("u" + str(u))
        #     print("v" + str(v))
        #     print("otheru" + str(other_coord[0]))
        #     print("otherv" + str(other_coord[1]))
        #     differences.append(math.sqrt((u - other_coord[0])**2 + (v - other_coord[1])**2))
        # differences = np.array(differences)
        # print(differences)
        differences = np.linalg.norm(coordinates_array - other_coord, ord = 2, axis=1)
        # print(differences)
        # print("Other coord is " + str(other_coord))
        # Are any of the changes above the threshold?
        return np.any(differences >= self.MAX_CHROMATICITY_DIFF)

class State:
    """
    Represents a state in the state machine.

    We determine whether there is a flash using the following state machine:

    A: Possible start state.
        Next State:
        A (default), 
        C (if there is a change of MAX_CHROMATICITY_DIFF)
        D (if there is a change of MAX_CHROMATICITY_DIFF and a saturated red)
    B: Possible start state, if we start at a frame containing a saturated red.
        Next State:
        B (default), 
        C (if there is a change of MAX_CHROMATICITY_DIFF)
    C: We have seen a change of MAX_CHROMATICITY_DIFF and a saturated red.
        Next State: 
        C (default),
        E (if there is a change of MAX_CHROMATICITY_DIFF)
    D: We have seen a change of MAX_CHROMATICITY_DIFF.
        Next State:
        D (default),
        E (if there is a change of MAX_CHROMATICITY_DIFF and a saturated red)
    E: We have seen a red flash

    Attributes:
        idx (int): The index at which the possible flash begins
        name (string): 'A', 'B', 'C', 'D', or 'E', indicating our state in the state machine
        chromaticity_checker (ChromaticityChecker): The chromaticity coordinates which allowed us to arrive at this state
        
    Methods:
        __hash__: Returns the hash value of a State
        __eq__: Checks whether two State objects are equal
        __repr__: Returns the string representation of a state
    """
    def __init__(self, name, chromaticity, idx):
        """
        Initializes a new instance of a State.

        Args:
            name (string): 'A', 'B', 'C', 'D', or 'E', indicating our state in the state machine
            chromaticity ((u', v')): The chromaticity value of the current state
            idx (int): The index at which the possible flash begins
        """
        if name not in ['A', 'B', 'C', 'D', 'E']:
            raise ValueError("Invalid state name")
        self.idx = idx
        self.name = name
        self.chromaticity_checker = ChromaticityChecker()
        self.chromaticity_checker.push(chromaticity)

    def __hash__(self):
        """
        Returns the hash value of a State.

        Returns:
            hash (int): The hash value of a State
        """
        return hash((self.name, self.idx))

    def __eq__(self, other):
        """
        Checks whether two State objects are equal

        Returns:
            are_equal (bool): True if the State objects are equal; False otherwise
        """
        if not isinstance(other, State):
            return False
        return self.name == other.name and self.idx == other.idx

    def __repr__(self):
        """
        Creates a string representation of a State

        Returns:
            state_string (string): String representation of the state object
        """
        return f"<State name:{self.name} index:{self.idx}>"
    
    @staticmethod
    def compare_states(state1, state2):
        """
        Determines whether one state is closer to the flash state than another.

        Returns:
            compare_state_value (int): 
                0: the states are not comparable (different indices)
                1: state 1 is closer to the flash state
                2: state 2 is closer to the flash state
                3: states are equal
        """
        # Each state maps to a "quality" (states closer to a flash state have higher quality)
        # E is closest to the flash state
        # A is furthest
        # C is preferable to B or D
        # B and D are not comparable
        if state1.idx != state2.idx and state1.name == state2.name:
            return 3

        quality = {"A": 0, "B": 1, "C": 2, "D": 1, "E": 3}
        
        if quality[state1.name] > quality[state2.name]:
            return 1
        elif quality[state2.name] > quality[state1.name]:
            return 2
        else:
            if state1.name == state2.name:
                return 4
            return 0

class Region:
    """
    Region represents a smaller area within a frame
    (allowing us to see if some part of a frame is flashing).

    Note that each region can have multiple states, 
    and it can arrive at these states using different sequences of frames.

    Attributes:
        buffer (Buffer): The buffer (set of frames) to which this region belongs
        MAX_RED_PERCENTAGE (float): The red percentage for a state to have a "saturated red"
        states (set(State)): The set of states we use in a 
        state machine to determine whether there is a flash
    
    Methods:
        should_transition(state, chromaticity, red_percentage, should_check_red_percentage): 
        Dictates whether we can transition to the next state in the state machine 
        based on whether there is an opposing transition and/or saturated red.

        update_or_add_state(state, state_set, chromaticity): 
        Adds a state to the set of states if it is not present. 
        If we have already reached this state, then we add its 
        chromaticity coordinate to that state's ChromaticityChecker.

        add_start_state(chromaticity, red_percentage, idx, state_set): 
        Adds a start state to the set of states, 
        based on whether the starting frame has a saturated red.

        state_machine(chromaticity, red_percentage): Update the set of states, 
        given the current states and the chromaticity
        and the red percentage of the frame being added.

        flash_idx(): Returns the index within the current buffer at which the flash occurs.
    """
    def __init__(self, buffer):
        """
        Initializes a new instance of a Region.

        Args:
            buffer (Buffer): The buffer (set of frames) to which this region belongs
        """
        self.buffer = buffer

        Region.MAX_RED_PERCENTAGE = 0.8

        self.states = set()

    @staticmethod
    def should_transition(
            state,
            chromaticity,
            red_percentage,
            should_check_red_percentage):
        """
        Dictates whether we can transition to the next state in the state machine
        based on whether there is an opposing transition and/or saturated red.

        Args:
            state (State): The current state in the state machine
            chromaticity ((float, float)): The chromaticity coordinate of the region in the newly-added frame
            red_percentage (float): The red percentage of the region in the newly-added frame
            should_check_red_percentage (bool): Whether the frame must 
            have a saturated red for us to transition
        
        Returns:
            should_transition(bool): Whether or not we can transition to 
            the next state in the state machine
        """
        # If we need a saturated red, but this frame
        # doesn't have a saturated red, then we can't transition
        if should_check_red_percentage and red_percentage < Region.MAX_RED_PERCENTAGE:
            return False

        # If the change in chromaticity exceeds MAX_CHROMATICITY_DIFF,
        # and we have a saturated red if needed, then we can transition
        # print(chromaticity)
        if state.chromaticity_checker.is_above_threshold(chromaticity):
            return True

        return False

    @staticmethod
    def update_or_add_state(state, state_set, chromaticity):
        """
        Adds a state to the set of states if it is not present. If we have already reached
        this state, then we add its chromaticity value to that state's ChromaticityChecker.

        Args:
            state (State): The state we want to add to the state machine
            state_set (set(State)): The set of states we are in in the state machine
            chromaticity ((float, float)): The chromaticity coordinate of the region in the newly-added frame
        """
        states_to_delete = []

        # print("Printing set")
        for s in state_set:
            # print(s)
            comp_value = State.compare_states(s, state)
            if comp_value == 2:
                # This state is better than our previous state
                states_to_delete.append(s)
            elif comp_value == 3:
                # This is an existing state; push the chromaticity value
                s.chromaticity_checker.push(chromaticity)
                return
            elif comp_value == 4:
                states_to_delete.append(s)
            elif comp_value == 1:
                # There is a better state in the set of states
                return
        
        state_set.add(state)

        for s in states_to_delete:
            state_set.remove(s)

    @staticmethod
    def add_start_state(chromaticity, red_percentage, idx, state_set):
        """
        Adds a start state corresponding to the newly-added frame to the state machine.

        Args:
            chromaticity ((float, float)): The chromaticity coordinate of the region in the newly-added frame
            red_percentage (float): The red percentage of the region in the newly-added frame
            idx (int): The index of the frame at which the flash would begin
            state_set(set(State)): The set of states we are in in the state machine
        """
        if red_percentage >= Region.MAX_RED_PERCENTAGE:
            state_b = State('B', chromaticity, idx)
            state_set.add(state_b)
        else:
            state_a = State('A', chromaticity, idx)
            state_set.add(state_a)

    def state_machine(self, chromaticity, red_percentage):
        """
        Update the set of states, given the current states
        and the chromaticity/red percentage of the frame being added.

        Args:
            chromaticity ((u', v')): The chromaticity coordinate of the region in the newly-added frame
            red_percentage (float): The red_percentage of the region in the newly-added frame
        """
        changed_state_set = set()

        for state in self.states:
            # We can stay in the same state
            # state.chromaticity_checker.push(chromaticity)
            Region.update_or_add_state(state, changed_state_set, chromaticity)

            if state.name == 'A':
                # print("hit here\n")
                if Region.should_transition(
                        state, chromaticity, red_percentage, True):
                    # print("Transition from A to C\n")
                    # We can move to state C if the chromaticity
                    # increased/decreased by MAX_CHROMATICITY_DIFF and there is
                    # a saturated red
                    state_c = State('C', chromaticity, state.idx)
                    # print("Reaching C\n")
                    Region.update_or_add_state(state_c, changed_state_set, chromaticity)
                elif Region.should_transition(
                        state, chromaticity, red_percentage, False):
                    state_d = State('D', chromaticity, state.idx)
                    # print("Transition from A to D\n")
                    # print("Reaching D\n")
                    Region.update_or_add_state(state_d, changed_state_set, chromaticity)
            elif state.name == 'B':
                # print("hit here\n")
                if Region.should_transition(
                        state, chromaticity, red_percentage, False):
                    # print("Transition from B to C\n")
                    # We can move to state C if the chromaticity
                    # increased/decreased by MAX_CHROMATICITY_DIFF
                    state_c = State('C', chromaticity, state.idx)
                    # print("Reaching C\n")
                    Region.update_or_add_state(state_c, changed_state_set, chromaticity)
            elif state.name == 'C':
                # print("hit here\n")
                if Region.should_transition(
                        state, chromaticity, red_percentage, False):
                    # print("Transition from C to E\n")
                    # We can move to state D if the chromaticity
                    # increased/decreased by MAX_CHROMATICITY_DIFF
                    state_e = State('E', chromaticity, state.idx)
                    # print("Reaching E\n")
                    Region.update_or_add_state(state_e, changed_state_set, chromaticity)
            elif state.name == 'D':
                # print("hit here\n")
                # print("Transition from D to E\n")
                if Region.should_transition(state, chromaticity, red_percentage, True):
                    state_e = State('E', chromaticity, state.idx)
                    # print("Reaching E\n")
                    Region.update_or_add_state(state_e, changed_state_set, chromaticity)
        
        # This could be our new start state
        Region.add_start_state(
            chromaticity,
            red_percentage,
            self.buffer.idx,
            changed_state_set)

        # print(changed_state_set)
        self.states = changed_state_set

    def flash_idx(self):
        """
        The index in the buffer at which the flash begins.

        Returns:
            idx (int): The index at which the flash begins, 
            or -1 if there is no flash within the region
        """
        for state in self.states:
            if state.name == 'E':
                return state.idx
        return -1

class Buffer:
    """
    Buffer represents the frames which are inside of the sliding window.

    Attributes:
        idx (int): Stores the index in the buffer at which the next frame should be added
        regions (np.array((n, n), dtype=Region)): Stores the regions within each frame
        num_frames (int): The number of frames in the buffer at a given time
        n (int): Dictates the number of regions (i.e., there are n x n regions in a frame)
        frame_rate (float): The frame rate
        red_flash_timestamps (list): The timestamps at which red flashes occur

    Methods:
        get_last_idx(): Get the last index at which a frame was added
        add_frame(frame): Add the chromaticity/red percentage values 
        remove_frame(self, idx): Remove a frame from the buffer
        for each region in the frame to the Region array
        get_red_flash_timestamps(): Get the timestamps at which red flashes occur
    """
    def __init__(self, num_frames, n, frame_rate):
        """
        Initializes a new instance of a Buffer.

        Args:
            num_frames (int): The number of frames in the buffer at a given time
            n (int): Dictates the number of regions (i.e., there are n x n regions in a frame)
        """
        self.idx = 0
        self.regions = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                self.regions[i][j] = Region(self)
        self.num_frames = frame_rate
        self.n = n
        self.frame_rate = frame_rate
        self.red_flash_timestamps = []

    def remove_frame(self, idx):
        """
        Remove a frame from the buffer.
        """
        for i in range(self.n):
            for j in range(self.n):
                self.regions[i][j].states = {item for item in self.regions[i][j].states if item.idx != idx}

    def add_frame(self, frame):
        """
        Add a frame to the buffer.

        Args:
            frame (np.array): An n x n numpy array of lists of form [u, v, red_percentage]
        """
        for i in range(self.n):
            for j in range(self.n):
                u, v, red_percentage = frame[i][j]
                # Update the states based on the current frame
                self.regions[i][j].state_machine((u, v), red_percentage)
                # Determine whether we have reached the flash state (E) for this region
                flash_idx = self.regions[i][j].flash_idx()
                if flash_idx != -1:
                    self.red_flash_timestamps.append([flash_idx / self.frame_rate, (flash_idx + self.num_frames) / self.frame_rate])
                    self.regions[i][j].states = {item for item in self.regions[i][j].states if item.idx != flash_idx}

        self.idx += 1

        # Remove states corresponding to frames which are no longer in the window
        out_idx = self.idx - self.num_frames

        if (out_idx) >= 0:
            self.remove_frame(out_idx)
    
    def get_red_flash_timestamps(self):
        """
        Returns the timestamps at which the red flashes occur.

        Returns:
            red_flash_timestamps (list): The timestamps at which red flashes occur
        """
        return self.red_flash_timestamps
    
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
    frame_buffer_red=Buffer(1,20,frame_rate)

    # if skipping a second to optimize
    skip = 0
    frame_counter = 0
    start_danger = -1
    # last_danger = -1

    timestamps = []

    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # print(frame_counter, frame_counter / frames_per_second)
            # print("done")
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

        flat_frame_rgb = frame_rgb.reshape(-1, 3)

        # Calculate b values for all pixels
        b = np.dot(flat_frame_rgb, tristimulus_matrix.T)

        # Calculate d values for all pixels
        d = b[:, 0] + 15 * b[:, 1] + 3 * b[:, 2]

        # Calculate u and v values for all pixels
        d[d == 0.0] = 999999999
        u = 4 * b[:, 0] / d
        v = 9 * b[:, 1] / d

        # Calculate cTotal for all pixels
        cTotal = np.sum(frame_rgb, axis=2).reshape(-1)

        # Calculate rperc values for all pixels
        cTotal[cTotal == 0.0] = 999999999
        rperc = flat_frame_rgb[:, 0] / cTotal

        # Reshape u, v, and rperc to the original shape
        u = u.reshape(frame_rgb.shape[0], frame_rgb.shape[1])
        v = v.reshape(frame_rgb.shape[0], frame_rgb.shape[1])
        rperc = rperc.reshape(frame_rgb.shape[0], frame_rgb.shape[1])

        # Combine u, v, and rperc into chromacityRerc
        chromacityRerc = np.stack((u, v, rperc), axis=2)
        
        # for i in range(frame_rgb.shape[0]):
        #     for j in range(frame_rgb.shape[1]):
        #         b = np.dot(tristimulus_matrix, frame_rgb[i,j])
        #         d = (b[0] + 15 * b[1] + 3 * b[2])
        #         u=0 if d == 0 else 4 * b[0] / d
        #         v=0 if d == 0 else 9 * b[1] / d
        #         cTotal = np.sum(frame_rgb[i,j])
        #         rperc = 0 if cTotal == 0 else frame_rgb[i,j,0] / cTotal
        #         chromacityRerc[i][j]=[u, v, rperc]
                #((U,V),Rper)
        # t1 = time.time()

        # print("Arr creation " + str(t1 - t0))

        # print(chromacityRerc)

        #Add the currecnt frame to the buffer for red detection
        frame_buffer_red.add_frame(chromacityRerc)
        # t2 = time.time()
        # print("Adding frame " + str(t2 - t1))
        # print(chromacityRerc)
        # Add the current frame to the buffer
        frame_buffer.append(hls_frame)

        # Skip a second of frames
        # if skip > 0:
        #     skip -= 1
        #     frame_buffer.popleft()
        #     continue

        # # Check if we have enough frames for the sliding window
        # # print(frame_buffer, frames_per_second)
        # if len(frame_buffer) == frames_per_second:
        #     # Fill the 'dangerous' array with the frames from the buffer
        #     for i, buf_frame in enumerate(frame_buffer):
        #         dangerous[i] = buf_frame

        #     # Process the 'dangerous' array
        #     flashes = process_dangerous(dangerous, frame_rate)
        #     if flashes >= hertz and start_danger == -1:
        #         start_danger = frame_counter
        #     if flashes < hertz:
        #         if start_danger >= 0:
        #             timestamps.append([start_danger / frame_rate, frame_counter / frame_rate])
        #             #print("danger from", start_danger / frames_per_second, "seconds to", frame_counter / frames_per_second, "seconds, frames", start_danger, frame_counter)
        #             start_danger = -1
        #             #last_danger = frame_counter
        #         #skip = frames_per_second
        #     frame_buffer.popleft()

            # print("number of flashes occured is" + str(flashes))
            #print(f"Processing window starting at frame {cap.get(cv2.CAP_PROP_POS_FRAMES) - frames_per_half_second}")
        # frame_counter += 1
    cap.release()

    end = time.time()

    print("Took " + str(end - start))

    #timestamp merge: Detection of flashes occurs within half-second windows so we want to merge what's close together
    idx = 0
    timestamps = frame_buffer_red.get_red_flash_timestamps()
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

filehandler("sample_videos/artificial_flash/Disco Lights trimmed.mp4", 2)