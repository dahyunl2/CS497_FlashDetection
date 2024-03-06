import sys
import os


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from red_transition_fsm import ChromaticityChecker, State, Region, Buffer
import numpy as np

# test_ctree = ChromaticityTree()
# test_ctree.push(3)
# test_ctree.push(5)
# # Tests that we correctly find the minimum and maximum element
# assert test_ctree.min() == 3
# assert test_ctree.max() == 5
# # Tests that we correctly update the minimum when element is removed
# test_ctree.pop(3)
# assert test_ctree.min() == 5
# # Tests that we correctly update the minimum when element is added
# test_ctree.push(3)
# assert test_ctree.min() == 3
# # Tests that we can correctly handle duplicate elements
# test_ctree.push(3)
# test_ctree.pop(3)
# assert test_ctree.min() == 3

# buffer = Buffer(4, 4)

# def generate_random_frame(n):
#     chromaticity = np.random.rand(n, n, 2)
#     red_percentage = np.random.rand(n, n)
#     frame = np.empty((n, n), dtype = tuple)
#     for i in range(n):
#         for j in range(n):
#             frame[i, j] = (chromaticity[i, j], red_percentage[i, j])
#     return frame

# def test_frame_n(n):
#     for i in range (n):
#         print(f"\nADDING FRAME {i}\n")
#         frame = generate_random_frame(4)
#         buffer.add_frame(frame)

# test_frame_n(6)

FileHandler.filehandler("/Users/dahyun/Documents/Programfile/CS497/CS497_FlashDetection/sample_videos/natural_flash/40 CAL Arc Flash Suit - Explosion test on ladder.mp4", 1)