import cv2
import numpy as np
import sklearn.neighbors
#import GeodisTK
import skimage.filters
import scipy.ndimage
from tqdm import tqdm
from video_utils import *

def background_substraction():
    stab_vid_cap = cv2.VideoCapture(INPUT_VIDEO_PATH) #TODO
    #extract stabilized video parameters
    params = get_video_parameters(stab_vid_cap)
    stab_vid_frame_count = params["frame_count"]

    #build extracted video writer
    extracted_writer = init_vid_writer(SUBTRACTED_EXTRACTED_VIDEO_PATH, params, True)
    
    #build binary video writer
    binary_writer = init_vid_writer(SUBTRACTED_BINARY_VIDEO_PATH, params, False)

    
    mixtures_num = 5
    threshold_of_var = 4

    #build GMM for background for the frame number
    back_gmm = cv2.createBackgroundSubtractorMOG2(history=int(stab_vid_frame_count))
    back_gmm.setNMixtures(mixtures_num)
    back_gmm.setVarThreshold(threshold_of_var)

    #build GMM for foreground for the frame number
    fore_gmm = cv2.createBackgroundSubtractorMOG2(history=int(stab_vid_frame_count))
    fore_gmm.setNMixtures(mixtures_num)
    fore_gmm.setVarThreshold(threshold_of_var)

    train_iters = 5
    #train the background GMM
    for i in tqdm(range(train_iters)):
        for j in tqdm(range(stab_vid_frame_count)):
            #set the capture to read from the jth frame from the end
            stab_vid_cap.set(1,stab_vid_frame_count-j-1)
            #read the frame
            ret, frame = stab_vid_cap.read()
            if not ret:
                print("failed to read frame")
                exit(1)
            back_gmm.apply(frame,learningRate=None)

    #train the foreground GMM
    for i in tqdm(range(train_iters)):
        #set the capture to read from the first frame in the video
        stab_vid_cap.set(1,0)
        for j in tqdm(range(stab_vid_frame_count)):
            #read the frame
            ret, frame = stab_vid_cap.read()
            if not ret:
                print("failed to read frame")
                exit(1)
            fore_gmm.apply(frame,learningRate=None)

    #build the kernel density estimatotion - kde and the histogram
    #together
    #we start from the first frame
    stab_vid_cap.set(1,0)
    #set number of frames for data
    data_frames_num = 5
    #collect necessary data both kde
    collect_kde_data_frames = np.empty([0,3]) #3 becuase of bgr
    collect_gray_data_for_hist = np.empty([0,1]) #1 because of grayscale
    for i in tqdm(range(data_frames_num)):
        ret, frame = stab_vid_cap.read()
        if not ret:
            print("failed to read frame")
            exit(1)
        foreground_aspect = fore_gmm.apply(frame,learningRate=0)
        #we want to focus only on the 255 values
        foreground_aspect[foreground_aspect<255]=0
        height, width = foreground_aspect.shape
        #TODO
        #we want not to refer to some pixels on top since the person is walking
        #top_factor = 8
        top_factor=10
        foreground_aspect[0:height//top_factor,:] = 0
        #take largest connected component - the foreground object in our case
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(\
            foreground_aspect,connectivity=4)
        component_sizes = stats[:,-1]
        label_max = int(np.argmax(component_sizes[1:]))+1
        new_foreground_aspect = np.zeros_like(labels)
        #make sure that label_max is an integer
        new_foreground_aspect[labels==label_max]=255
        #here we want to clean noise by opening and closing morphology
        open_ker_size = 5
        close_ker_size = 11
        #create kernel - opening
        open_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ker_size,open_ker_size))
        #execute the morphology
        foreground_aspect_after_open = cv2.morphologyEx(new_foreground_aspect.astype(np.uint8),\
                                                        cv2.MORPH_OPEN, open_ker)
        #crete kernel - closing
        close_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ker_size,close_ker_size))
        #execute the morphology
        foreground_aspect_clean = cv2.morphologyEx(foreground_aspect_after_open.astype(np.uint8),\
                                                        cv2.MORPH_CLOSE, close_ker)
        #convert to binary
        foreground_aspect_clean[foreground_aspect_clean>=127] = 255
        foreground_aspect_clean[foreground_aspect_clean<127] = 0

        #add to kde data
        kde_data_frame = frame[foreground_aspect_clean==255]
        collect_kde_data_frames = np.append(collect_kde_data_frames,kde_data_frame,axis=0)

        #add to histogram data
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_data_frame = grayscale_frame[foreground_aspect_clean==255]
        collect_gray_data_for_hist = np.append(collect_gray_data_for_hist, \
                                               np.expand_dims(gray_data_frame,1),axis=0)

    #initialize the kde
    kde = sklearn.neighbors.KernelDensity(bandwidth=0.3, kernel='gaussian',atol=0.00000001)
    #use the data collected
    kde.fit(collect_kde_data_frames)


    #initialize histogram, using the data collected
    ps, bins = np.histogram(collect_gray_data_for_hist.flatten(),bins=25, density=True)
    bins_diff = bins[1]-bins[0]
    #calculate the pdf
    pdf = bins_diff*np.cumsum(ps)

    #threshold acoording to desired low and high pdf thresholds
    #define histogram's edge values
    #low
    #pdf_low_th = 0.001
    pdf_low_th = 0.002
    low_cond = np.where(pdf<pdf_low_th)
    hist_min_val = 0
    if (len(low_cond[0])>0): #need to update histogram minimum value
        #take the maximum out of the bins in the condition
        hist_min_val = bins[np.max(low_cond)]
    
    #high
    #pdf_high_th = 0.999
    pdf_high_th = 0.998
    high_cond = np.where(pdf>pdf_high_th)
    hist_max_val = 255
    if (len(high_cond[0])>0): #need to update histogram minimum value
        #take the minimum out of the bins in the condition
        hist_max_val = bins[np.min(high_cond)]


    #now we run on the video
    #need to start from top
    stab_vid_cap.set(1,0)
    old_frame_applied = None
    frame_last_trained = data_frames_num
    #go over each frame
    for i in tqdm(range(stab_vid_frame_count)):
        ret, frame = stab_vid_cap.read()
        if not ret:
            print("failed to read frame")
            exit(1)
        #for first half, work with foreground gmm, for the second with background gmm
        if i<(stab_vid_frame_count//2):
            frame_applied = fore_gmm.apply(frame,learningRate=0)
        else:
            frame_applied = back_gmm.apply(frame,learningRate=0)

        frame_applied[frame_applied<255]=0
        height, width = frame_applied.shape
        #TODO
        #we want not to refer to some pixels on top since the person is walking
        frame_applied[0:height//top_factor,:] = 0
        #take largest connected component - the foreground object in our case
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(\
            frame_applied,connectivity=4)
        component_sizes = stats[:,-1]
        label_max = int(np.argmax(component_sizes[1:]))+1
        new_frame_applied = np.zeros_like(labels)
        #make sure that label_max is an integer
        new_frame_applied[labels==label_max]=255
        #here we want to clean noise by opening and closing morphology
        open_ker_size = 5
        close_ker_size = 11
        #create kernel - opening
        open_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ker_size,open_ker_size))
        #execute the morphology
        foreground_applied_after_open = cv2.morphologyEx(new_frame_applied.astype(np.uint8),\
                                                        cv2.MORPH_OPEN, open_ker)
        #crete kernel - closing
        close_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ker_size,close_ker_size))
        #execute the morphology
        foreground_applied_clean = cv2.morphologyEx(foreground_applied_after_open.astype(np.uint8),\
                                                        cv2.MORPH_CLOSE, close_ker)
        #convert to binary
        foreground_applied_clean[foreground_aspect_clean>=127] = 255
        foreground_applied_clean[foreground_aspect_clean<127] = 0

        old_foreground_applied_clean = np.copy(foreground_applied_clean)

        #remove values exceeding histogram limits
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        foreground_applied_clean[grayscale_frame<hist_min_val] = 0
        foreground_applied_clean[grayscale_frame>hist_max_val] = 0
        foreground_applied_after_hist_adjust = np.copy(foreground_applied_clean)

        #need to check time derivative change in the movement's direction
        if i>data_frames_num: #frame which we haven't collected data for kde and histogram
            frames_applied_diff = (foreground_applied_clean-old_frame_applied)*\
            (foreground_applied_clean-old_frame_applied>0)
            #check if we need to clear noise
            diff_th_for_clean = 8500
            sum = np.sum(frames_applied_diff)
            if (sum/255)>diff_th_for_clean:
                foreground_applied_clean = kde_refine_clear_noise(foreground_applied_after_hist_adjust,frame,top_factor)
            #check need to retrain the kde
            retrain_th = 7000
            if ((sum/255)<retrain_th) and (i-frame_last_trained>=data_frames_num):
                collect_kde_data_frames = np.append(collect_kde_data_frames,frame[foreground_applied_clean>0],axis=0)
                kde_temp_size = 100000
                #collect_kde_data_frames needs to be in the size of kde_temp_size
                if len(collect_kde_data_frames)>kde_temp_size:
                    collect_kde_data_frames = collect_kde_data_frames[-kde_temp_size:]
                    #initialize the kde
                kde = sklearn.neighbors.KernelDensity(bandwidth=0.3, kernel='gaussian',atol=0.00000001)
                #use the data collected
                kde.fit(collect_kde_data_frames)
                frame_last_trained = i
        
        #fill holes
        foreground_applied_clean = 255*scipy.ndimage.binary_fill_holes(foreground_applied_clean).astype(np.uint8)
        #adjust to version before histogram removal
        foreground_applied_clean[old_foreground_applied_clean<255] = 0
        #save foreground_applied_clean in the old frame variable
        old_frame_applied = np.copy(foreground_applied_clean)

        #write to extracted video
        frame_ext = np.zeros_like(frame)
        #fill in the non zeros indices of foreground_applied_clean=255 the frame in those indices
        #print("max frame: " +str(frame.max()))
        #print("max frame[foreground_applied_clean==255]: " +str(frame[foreground_applied_clean==255].max()))
        frame_ext[foreground_applied_clean==255] = frame[foreground_applied_clean==255]
        extracted_writer.write(frame_ext.astype(np.uint8))
        #print(frame_ext.max())

        #write to binary video
        binary_writer.write(foreground_applied_clean.astype(np.uint8))

    #release writers
    extracted_writer.release()
    binary_writer.release()


def kde_refine_clear_noise(foregroud_elem, frame,kde,top_factor):
    frame_fore_pix = frame[foregroud_elem==255]
    temp_fact = 48
    log_like_res = np.exp(kde.score_samples(frame_fore_pix)/temp_fact)
    foregroud_elem_ps = np.zeros_like(foregroud_elem,dtype=float)
    #apply log likelihood res to probabilities
    foregroud_elem_ps[foregroud_elem==255] = log_like_res

    #build geodesik distance map
    x_cent, y_cent, width, height = cv2.boundingRect(foregroud_elem)
    crop_frame_fore_pix = foregroud_elem_ps[y_cent:y_cent+height, x_cent:x_cent+width]
    high_pix_th = 0.78
    cropped_cond = crop_frame_fore_pix>high_pix_th
    geo_dist_map = np.random.rand(width*height).reshape(crop_frame_fore_pix.shape)
    #geo_dist_map = GeodisTK.geodesic2d_raster_scan(crop_frame_fore_pix.astype(np.float32),\
     #                                             cropped_cond.astype(np.uint8),1.0,2)
    cropped_max_sub_map = geo_dist_map.max()-geo_dist_map
    cropped_th = 0.002
    cropped_max_sub_map[crop_frame_fore_pix<cropped_th] = 0

    #make 3-level otsu thresholding
    threshs = skimage.filters.threshold_multiotsu(cropped_max_sub_map)
    regions_from_threshs = np.digitize(cropped_max_sub_map,bins=threshs)
    region_to_remove = 1
    regions_th = 5000
    if np.sum(regions_from_threshs==1)>regions_th:
        region_to_remove = 0
    #handle cropped ds
    cropped_max_sub_map[cropped_max_sub_map<=threshs[region_to_remove]]=0
    cropped_max_sub_map[cropped_max_sub_map>threshs[region_to_remove]]=255
    #initiate new foregroud_elem
    foregroud_elem = np.zeros_like(foregroud_elem, dtype=np.uint8)
    foregroud_elem[y_cent:y_cent+height, x_cent:x_cent+width] = cropped_max_sub_map.astype(np.uint8)
    #we want to focus only on the 255 values
    foregroud_elem[foregroud_elem<255]=0
    height, width = foregroud_elem.shape
    #TODO
    #we want not to refer to some pixels on top since the person is walking
    foregroud_elem[0:height//top_factor,:] = 0
    #take largest connected component - the foreground object in our case
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(\
        foregroud_elem,connectivity=4)
    component_sizes = stats[:,-1]
    label_max = int(np.argmax(component_sizes[1:]))+1
    new_foreground_elem = np.zeros_like(labels)
    #make sure that label_max is an integer
    new_foreground_elem[labels==label_max]=255
    #here we want to clean noise by only closing morphology
    close_ker_size = 11
    #crete kernel - closing
    close_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ker_size,close_ker_size))
    #execute the morphology
    foreground_elem_clean = cv2.morphologyEx(new_foreground_elem.astype(np.uint8),\
                                                    cv2.MORPH_CLOSE, close_ker)
    #convert to binary
    foreground_elem_clean[foreground_elem_clean>=127] = 255
    foreground_elem_clean[foreground_elem_clean<127] = 0

    return foreground_elem_clean

background_substraction()