import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from video_utils import *

INPUT_VIDEO_PATH = 'Inputs/INPUT.avi'
STABILIZED_VIDEO_PATH = 'Outputs/stabilize.avi'


def stabilize():
    vid_input = cv2.VideoCapture(INPUT_VIDEO_PATH)
    
    vid_params = get_video_parameters(vid_input)
    width, height = (vid_params['width'], vid_params['height'])
    vid_writer_stabilized = init_vid_writer(STABILIZED_VIDEO_PATH, vid_params, True)

    success, current_frame = vid_input.read()
    if not success:
        release_videos([vid_input, vid_writer_stabilized])
        print('Error reading input video during stabilization, Exiting')
        exit()
    
    vid_writer_stabilized.write(current_frame.astype('uint8'))

    for _ in tqdm(range(1, vid_params['frame_count'])):
        success, next_frame = vid_input.read()
        if not success:
            break

        warped_frame = calc_warped_frame(current_frame, next_frame, (width, height))
        vid_writer_stabilized.write(warped_frame.astype('uint8'))

    release_videos([vid_input, vid_writer_stabilized])


def calc_warped_frame(current_frame, next_frame, frame_dimensions):
    sift = cv2.SIFT_create()
    bf_matcher = cv2.BFMatcher()

    keypoints_curr, descriptors_curr = sift.detectAndCompute(current_frame, None)
    keypoints_next, descriptors_next = sift.detectAndCompute(next_frame, None)

    matches = bf_matcher.match(descriptors_curr, descriptors_next)

    curr_frame_pts = np.zeros((len(matches), 1, 2))
    next_frame_pts = np.zeros((len(matches), 1, 2))

    for i, match in enumerate(matches):
        curr_frame_pts[i] = np.array([keypoints_curr[match.queryIdx].pt], dtype=np.float32)
        next_frame_pts[i] = np.array([keypoints_next[match.trainIdx].pt], dtype=np.float32)

    homography, _ = cv2.findHomography(next_frame_pts, curr_frame_pts, method=cv2.RANSAC, confidence=0.99)

    frame_stabilized = cv2.warpPerspective(next_frame, homography, frame_dimensions)

    return frame_stabilized


stabilize()