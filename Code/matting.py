import numpy as np
import cv2
import GeodisTK
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from video_utils import *


ALPHA_AREA_KERNEL_WIDTH = 5
TRIMAP_BBOX_PADDING = 5
DIST_POW = -1
EPSILON = 10 ** -7
BOUNDRY_THRESHOLD = 0.1
PIXEL_NEIGHBORHOOD_WIDTH = 5
log = True

INPUT_VIDEO_PATH = 'Inputs/INPUT.avi'
BINARY_VIDEO_PATH = 'Outputs/binary.avi'
ALPHA_OUT_PATH = 'Outputs/alpha.avi'
MATTED_OUT_PATH = 'Outputs/matted.avi'
OUTPUT_OUT_PATH = 'Outputs/OUTPUT.avi'

def matting_and_tracking():
    vid_input = cv2.VideoCapture(SUBTRACTED_EXTRACTED_VIDEO_PATH)
    vid_binary = cv2.VideoCapture(BINARY_VIDEO_PATH)

    vid_params = get_video_parameters(vid_binary)
    width, height = (vid_params['width'], vid_params['height'])

    vid_writer_alpha = init_vid_writer(ALPHA_OUT_PATH, vid_params, False)
    vid_writer_matting = init_vid_writer(MATTED_OUT_PATH, vid_params, True)
    vid_writer_output = init_vid_writer(OUTPUT_OUT_PATH, vid_params, True)

    background_img = cv2.imread('Inputs/background.jpg')

    for _ in tqdm(range(vid_params['frame_count']-1)):
        success_input, input_frame = vid_input.read()
        success_binary, binary_frame = vid_binary.read()
        binary_frame = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)

        if success_input or success_binary:
            alpha, matted, tracked =  run_matting_and_tracking_on_frame(input_frame, binary_frame, background_img)

            vid_writer_alpha.write(alpha.astype('uint8'))
            vid_writer_matting.write(matted.astype('uint8'))
            vid_writer_output.write(tracked.astype('uint8'))
        else:
            break

    release_videos([vid_input, vid_binary, vid_writer_alpha, vid_writer_matting, vid_writer_output])


def run_matting_and_tracking_on_frame(frame, binary_img, background_img):
    luma_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    trimap = create_initial_trimap(binary_img)
    bbox = get_bounding_box(trimap)
    alpha_map, cropped_alpha = calculate_alpha_map(trimap, luma_frame, bbox)
    matted_frame = calculate_matted_frame(frame, background_img, cropped_alpha, bbox)
    # cv2.imwrite('Outputs/alpha_map.png', alpha_map*255)
    # cv2.imwrite('Outputs/matted_frame.png', matted_frame)

    tracked = cv2.rectangle(np.copy(matted_frame), (bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]), (255,0,0), 2)
    cv2.imwrite('Outputs/matted_frame.png', matted_frame)
    cv2.imwrite('Outputs/tracked.png', tracked)

    return (alpha_map*255), matted_frame, tracked



def create_initial_trimap(binary_img):
    trimap = binary_img / 255
    alpha_kernel = np.ones((ALPHA_AREA_KERNEL_WIDTH,ALPHA_AREA_KERNEL_WIDTH))
    
    img_erosion = cv2.erode(binary_img, alpha_kernel, iterations=2)
    img_dilation = cv2.dilate(binary_img, alpha_kernel, iterations=2)

    alpha_region = img_dilation - img_erosion

    trimap[alpha_region > 0] = 0.5

    if log:
        cv2.imwrite('Outputs/trimap.png', trimap*255 )
    
    return trimap


def calculate_alpha_map(trimap, luma_img, bbox):
    min_row, min_col, max_row, max_col = bbox

    alpha_map = np.copy(trimap)
    cropped_trimap = trimap[min_row: max_row, min_col: max_col] 
    cropped_alpha = alpha_map[min_row: max_row, min_col: max_col]
    cropped_origin = luma_img[min_row: max_row, min_col: max_col]

    alpha_indices = np.where(cropped_trimap == 0.5)
    alpha_pixels = cropped_origin[alpha_indices]
    unique_vals = np.unique(alpha_pixels)

    def_bg_pixels = np.zeros_like(cropped_alpha, dtype=np.uint8)
    def_bg_pixels[cropped_trimap == 0] = 1
    bg_geodist = GeodisTK.geodesic2d_raster_scan(cropped_origin, def_bg_pixels, 1.0, 2)
    d_bg = bg_geodist[alpha_indices]
    bg_pdf = gaussian_kde(cropped_origin[cropped_trimap == 0]) #TODO: check if should be background image
    unique_probs_bg = bg_pdf.evaluate(unique_vals)
    bg_probabilities = np.zeros(alpha_pixels.shape)
    for u, p in zip(unique_vals, unique_probs_bg):
        bg_probabilities[alpha_pixels == u] = p 
    # bg_probabilities = np.zeros_like(alpha_pixels) + 1

 
    def_fg_pixels = np.zeros_like(cropped_alpha, dtype=np.uint8)
    def_fg_pixels[cropped_trimap == 1] = 1
    fg_geodist = GeodisTK.geodesic2d_raster_scan(cropped_origin, def_fg_pixels, 1.0, 2)
    d_fg = fg_geodist[alpha_indices]
    fg_pdf = gaussian_kde(cropped_origin[cropped_trimap == 1])
    unique_probs_fg = fg_pdf.evaluate(unique_vals)
    fg_probabilities = np.zeros(alpha_pixels.shape)
    for u, p in zip(unique_vals, unique_probs_fg):
        fg_probabilities[alpha_pixels == u] = p 

    # fg_probabilities = np.zeros_like(alpha_pixels) + 1


    w_bg = bg_probabilities * ((d_bg + EPSILON)**DIST_POW)
    w_fg = fg_probabilities * ((d_fg + EPSILON)**DIST_POW)

    alpha_values = w_fg / (w_fg + w_bg + EPSILON)

    cropped_alpha[alpha_indices] = alpha_values

    alpha_map[min_row: max_row, min_col: max_col] = cropped_alpha

    return alpha_map, cropped_alpha


def calculate_matted_frame(img, background_img, cropped_alpha, bbox):
    matted_frame = np.copy(background_img)
    matted_box = matted_frame[bbox[0]: bbox[2], bbox[1]: bbox[3]]
    cropped_img = img[bbox[0]: bbox[2], bbox[1]: bbox[3]]
    cropped_bg = matted_frame[bbox[0]: bbox[2], bbox[1]: bbox[3]]

    matted_box[cropped_alpha >= 1-BOUNDRY_THRESHOLD] = cropped_img[cropped_alpha >= 1-BOUNDRY_THRESHOLD]
    matted_box[cropped_alpha < BOUNDRY_THRESHOLD] = cropped_bg[cropped_alpha < BOUNDRY_THRESHOLD]

    boundry_indices = np.where(np.abs(cropped_alpha-0.5) < BOUNDRY_THRESHOLD/2)
    for row, col in zip(boundry_indices[0],boundry_indices[1]):
        min_row, min_col, max_row, max_col = calc_slice_limits(cropped_img, row, col, row, col, PIXEL_NEIGHBORHOOD_WIDTH//2, PIXEL_NEIGHBORHOOD_WIDTH//2)
        fg_pixel = cropped_img[row, col]
        fg_neighborhood = cropped_img[min_row: max_row, min_col: max_col]
        bg_neighborhood = cropped_bg[min_row: max_row, min_col: max_col]

        matted_box[row, col] = calc_pixel_value(fg_neighborhood, bg_neighborhood, fg_pixel, cropped_alpha[row,col])

    matted_frame[bbox[0]: bbox[2], bbox[1]: bbox[3]] = matted_box
    return matted_frame 
    

def calc_pixel_value(fg_neighborhood, bg_neighborhood, pixel, alpha):
    h, w, d = fg_neighborhood.shape

    fg_pixels_flatten = np.reshape(fg_neighborhood, (h*w, d))
    fg_pixels_grid = np.tile(fg_pixels_flatten, (h*w, 1, 1))

    bg_pixels_flatten = np.reshape(bg_neighborhood, (h*w, d))
    bg_pixels_grid = np.tile(bg_pixels_flatten, (h*w, 1, 1))
    bg_pixels_grid = np.transpose(bg_pixels_grid, (1,0,2))

    equation = fg_pixels_grid*alpha + bg_pixels_grid*(1-alpha) - pixel
    squared_norm = np.sum(equation ** 2, axis=-1)
    min_index = np.unravel_index(np.argmin(squared_norm), squared_norm.shape)
    return equation[min_index] + pixel



def get_bounding_box(trimap):
    rows, cols = np.nonzero(trimap)
    return calc_slice_limits(trimap, np.min(rows), np.min(cols), np.max(rows), np.max(cols), TRIMAP_BBOX_PADDING, TRIMAP_BBOX_PADDING)


def calc_slice_limits(arr, start_row, start_col, end_row, end_col, pad_w, pad_h):
    min_row = max(0, start_row - pad_h)
    max_row = min(arr.shape[0], end_row + pad_h)
    min_col = max(0, start_col - pad_w)
    max_col = min(arr.shape[1], end_col + pad_w)

    return min_row, min_col, max_row, max_col




# video_capture = cv2.VideoCapture('../../../ref_VPproject/FinalProject_300508850_021681283/Outputs/binary.avi')  
# video_capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
# success, frame = video_capture.read()
# cv2.imwrite('../Inputs/bin_img.png', frame)
matting_and_tracking()