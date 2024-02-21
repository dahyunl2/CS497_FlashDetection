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
from sortedcontainers import SortedDict

class ChromaticityTree:
    """
    ChromaticityTree allows us to determine the min/max chromaticity for a region.

    Attributes:
        ct (SortedDict): Stores the chromaticity values and their frequency

    Methods:
        push(element): Add chromaticity value to the ChromaticityTree
        pop(element): Remove chromaticity value from the ChromaticityTree
        min(): Determine the minimum chromaticity value in the ChromaticityTree
        max(): Determine the maximum chromaticity value in the ChromaticityTree
    """

    def __init__(self):
        """
        Initializes a new instance of a ChromaticityTree.
        """
        self.ct = SortedDict()

    def push(self, element):
        """
        Add a chromaticity value to the ChromaticityTree.

        Args:
            element(float): The chromaticity value to add to the ChromaticityTree
        """
        self.ct[element] = self.ct.get(element, 0) + 1

    def pop(self, element):
        """
        Removes a chromaticity value from the ChromaticityTree.

        Args:
            element(float): The chromaticity value to remove from the ChromaticityTree
        """
        num_occurrences = self.ct.get(element, 0)

        if num_occurrences == 0:
            raise ValueError("Element is not in tree")
        if num_occurrences == 1:
            del self.ct[element]
        else:
            self.ct[element] = num_occurrences - 1

    def min(self):
        """
        Determine the minimum chromaticity value in the ChromaticityTree.

        Returns:
            min(float): the minimum chromaticity value in the ChromaticityTree.
        """
        if self.ct:
            return next(iter(self.ct))
        raise ValueError("Chromaticity tree has no elements")

    def max(self):
        """
        Determine the maximum chromaticity value in the ChromaticityTree.

        Returns:
            max(float): maximum chromaticity value in the ChromaticityTree.
        """
        if self.ct:
            return next(reversed(self.ct))
        raise ValueError("Chromaticity tree has no elements")

class State:
    """
    Represents a state in the state machine.

    We determine whether there is a flash using the following state machine:

    A: Possible start state.
        Next State:
        A (default), 
        C (if there is a change of MAX_CHROMATICITY_DIFF and we see a saturated red)
    B: Possible start state, if we start at a frame containing a saturated red.
        Next State:
        B (default), 
        D (if there is a change of MAX_CHROMATICITY_DIFF)
    C: We have seen a change of MAX_CHROMATICITY_DIFF and a saturated red.
        Next State: 
        C (default),
        E (if there is a change of MAX_CHROMATICITY_DIFF)
    D: We have seen a single opposing transition involving a saturated red.
        Next State: 
        D (default),
        E (if there is a change of MAX_CHROMATICITY_DIFF and we see a saturated red)
    E: We have seen two opposing transitions involving a saturated red.

    Attributes:
        idx (int): The index at which the possible flash begins
        name (string): 'A', 'B', 'C', 'D', or 'E', indicating our state in the state machine
        chromaticity_tree (ChromaticityTree): The min/max chromaticity values 
        
    Methods:
        __hash__: Returns the hash value of a State
        __eq__: Checks whether two State objects are equal
    """
    def __init__(self, name, chromaticity, idx):
        """
        Initializes a new instance of a State.

        Args:
            name (string): 'A', 'B', 'C', 'D', or 'E', indicating our state in the state machine
            chromaticity (float): The chromaticity value of the current state
            idx (int): The index at which the possible flash begins
        """
        if name not in ['A', 'B', 'C', 'D', 'E']:
            raise ValueError("Invalid state name")
        self.idx = idx
        self.name = name
        self.chromaticity_tree = ChromaticityTree()
        self.chromaticity_tree.push(chromaticity)

    def __hash__(self):
        """
        Returns the hash value of a State.
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

class Region:
    """
    Region represents a smaller area within a frame
    (allowing us to see if some part of a frame is flashing).

    Note that each region can have multiple states, 
    and it can arrive at these states using different sequences of frames.

    Attributes:
        buffer (Buffer): The buffer (set of frames) to which this region belongs
        chromaticity (float): The initial chromaticity value of the region
        red_percentage (float): The initial R/(R + G + B) value of the region 
        MAX_CHROMATICITY_DIFF (float): The chromaticity value difference in 
        states for there to be an opposing transition
        MAX_RED_PERCENTAGE (float): The red percentage for a state to have a "saturated red"
        states (set(State)): The set of states we use in a 
        state machine to determine whether there is a flash
    
    Methods:
        should_transition(state, chromaticity, red_percentage, should_check_red_percentage): 
        Dictates whether we can transition to the next state in the state machine 
        based on whether there is an opposing transition and/or saturated red.

        update_or_add_state(state, state_set): 
        Adds a state to the set of states if it is not present. 
        If we have already reached this state, then we add its 
        chromaticity value to that state's ChromaticityTree.

        add_start_state(chromaticity, red_percentage, idx, state_set): 
        Adds a start state to the set of states, 
        based on whether the starting frame has a saturated red.

        state_machine(chromaticity, red_percentage): Update the set of states, 
        given the current states and the chromaticity
        and the red percentage of the frame being added.

        flash_idx(): Returns the index within the current buffer at which the flash occurs.
    """
    def __init__(self, buffer, chromaticity, red_percentage):
        """
        Initializes a new instance of a Region.

        Args:
            buffer (Buffer): The buffer (set of frames) to which this region belongs
            chromaticity (float): The initial chromaticity value of the region
            red_percentage (float): The initial R/(R + G + B) value of the region 
        """
        self.buffer = buffer
        self.chromaticity = chromaticity
        self.red_percentage = red_percentage

        Region.MAX_CHROMATICITY_DIFF = 0.2
        Region.MAX_RED_PERCENTAGE = 0.8

        self.states = set()
        Region.add_start_state(chromaticity, red_percentage, self.buffer.idx, self.states)

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
            chromaticity (float): The chromaticity of the region in the newly-added frame
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
        if abs(chromaticity - state.chromaticity_tree.max()
               ) >= Region.MAX_CHROMATICITY_DIFF:
            return True

        if abs(chromaticity - state.chromaticity_tree.min()
               ) >= Region.MAX_CHROMATICITY_DIFF:
            return True

        return False

    @staticmethod
    def update_or_add_state(state, state_set):
        """
        Adds a state to the set of states if it is not present. If we have already reached
        this state, then we add its chromaticity value to that state's ChromaticityTree.

        Args:
            state (State): The state we want to add to the state machine
            state_set (set(State)): The set of states we are in in the state machine
        """
        for s in state_set:
            if s == state:
                s.chromaticity_tree.push(state.chromaticity)
                return
        state_set.add(state)

    @staticmethod
    def add_start_state(chromaticity, red_percentage, idx, state_set):
        """
        Adds a start state corresponding to the newly-added frame to the state machine.

        Args:
            chromaticity (float): The chromaticity of the region in the newly-added frame
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
            chromaticity (float): The chromaticity of the region in the newly-added frame
            red_percentage (float): The red_percentage of the region in the newly-added frame
        """
        changed_state_set = set()

        # This could be our new start state
        Region.add_start_state(
            chromaticity,
            red_percentage,
            self.buffer.idx,
            changed_state_set)

        for state in self.states:
            # We do not want to add states for which the starting frame is no
            # longer in the buffer
            if state.idx == self.buffer.get_last_idx():
                continue

            # We always stay in the current state
            Region.update_or_add_state(state, changed_state_set)

            if state.name == 'A':
                if Region.should_transition(
                        state, chromaticity, red_percentage, True):
                    # We can move to state C if the chromaticity
                    # increased/decreased by MAX_CHROMATICITY_DIFF and there is
                    # a saturated red
                    state_c = State('C', chromaticity, self.buffer.idx)
                    Region.update_or_add_state(state_c, changed_state_set)
            elif state.name == 'B':
                if Region.should_transition(
                        state, chromaticity, red_percentage, False):
                    # We can move to state D if the chromaticity
                    # increased/decreased by MAX_CHROMATICITY_DIFF
                    state_d = State('D', chromaticity, self.buffer.idx)
                    Region.update_or_add_state(state_d, changed_state_set)
            elif state.name == 'C':
                if Region.should_transition(
                        state, chromaticity, red_percentage, False):
                    # We can move to state E if the chromaticity
                    # increased/decreased by MAX_CHROMATICITY_DIFF
                    state_e = State('E', chromaticity, self.buffer.idx)
                    Region.update_or_add_state(state_e, changed_state_set)
            elif state.name == 'D':
                if Region.should_transition(
                        state, chromaticity, red_percentage, True):
                    # We can move to state E if the chromaticity
                    # increased/decreased by MAX_CHROMATICITY_DIFF and there is
                    # a saturated red
                    state_e = State('E', chromaticity, self.buffer.idx)
                    Region.update_or_add_state(state_e, changed_state_set)

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

    Methods:
        get_last_idx(): Get the last index at which a frame was added
        add_frame(frame): Add the chromaticity/red percentage values 
        for each region in the frame to the Region array
    """
    def __init__(self, num_frames, n):
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
                # TODO: Change this to use the actual values corresponding to
                # the first frame
                self.regions[i][j] = Region(
                    self, np.random.rand(), np.random.rand())
        self.num_frames = num_frames
        self.n = n

    def get_last_idx(self):
        """
        Get the index at which the previous frame was added

        Returns:
            idx (int): The index at which the previous frame was added
        """
        if self.idx == 0:
            return self.n - 1
        return (self.idx - 1) % self.n

    def add_frame(self, frame):
        """
        Add a frame to the buffer.

        Args:
            frame (np.array): An n x n numpy array of tuples of form (chromaticity, red_percentage)
        """
        self.idx += 1

        for i in range(self.n):
            for j in range(self.n):
                chromaticity, red_percentage = frame[i][j]
                self.regions[i][j].state_machine(chromaticity, red_percentage)
                flash_idx = self.regions[i][j].flash_idx()
                if flash_idx != -1:
                    print(
                        "There is a flash at the region located at row " +
                        str(i) +
                        " and column " +
                        str(j) +
                        " for the buffer starting at " +
                        str(flash_idx))
