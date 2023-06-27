import cv2
import numpy as np
import sklearn.neighbors

def background_substraction(stab_vid_cap):
    #extract stabilized video parameters
    stab_vid_width = int(stab_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    stab_vid_height = int(stab_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stab_vid_frame_count = int(stab_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stab_vid_fps = stab_vid_cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    #build extracted video writer
    extracted_video_path = "extracted.avi"
    extracted_writer = cv2.VideoWriter(extracted_video_path, fourcc, stab_vid_fps, \
                                       (stab_vid_width,stab_vid_height))
    
    #build binary video writer
    binary_video_path = "binary.avi"
    binary_writer = cv2.VideoWriter(binary_video_path, fourcc, stab_vid_fps, \
                                       (stab_vid_width,stab_vid_height),0)
    
    mixtures_num = 5
    threshold_of_var = 4

    #build GMM for background for the frame number
    back_gmm = cv2.createBackgroundSubtractorMOG2(stab_vid_frame_count)
    back_gmm.setNMixtures(mixtures_num)
    back_gmm.setVarThreshold(threshold_of_var)

    #build GMM for foreground for the frame number
    fore_gmm = cv2.createBackgroundSubtractorMOG2(stab_vid_frame_count)
    fore_gmm.setNMixtures(mixtures_num)
    fore_gmm.setVarThreshold(threshold_of_var)

    train_iters = 5
    #train the background GMM
    for i in range(train_iters):
        for j in range(stab_vid_frame_count):
            #set the capture to read from the jth frame from the end
            stab_vid_cap.set(1,stab_vid_frame_count-j-1)
            #read the frame
            ret, frame = stab_vid_cap.read()
            if not ret:
                print("failed to read frame")
                exit(1)
            back_gmm.apply(frame,learningRate=None)

    #train the foreground GMM
    for i in range(train_iters):
        #set the capture to read from the first frame in the video
        stab_vid_cap.set(1,0)
        for j in range(stab_vid_frame_count):
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
    for i in range(data_frames_num):
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
        foreground_aspect[0:height//8,:] = 0
        #take largest connected component - the foreground object in our case
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(\
            foreground_aspect,connectivity=4)
        component_sizes = stats[:,-1]
        label_max = np.floor(np.argmax(component_sizes[1:]))+1
        new_foreground_aspect = np.zeros(labels.shape,dtype=np.int32)
        #make sure that label_max is an integer
        new_foreground_aspect[labels==int(label_max)]=255
        #here we want to clean noise by opening and closing morphology
        open_ker_size = 5
        close_ker_size = 11
        #create kernel - opening
        open_ker = cv2.getStructuringElement(cv2.MORPH_OPEN, (open_ker_size,open_ker_size))
        #execute the morphology
        foreground_aspect_after_open = cv2.morphologyEx(new_foreground_aspect.astype(np.uint8),\
                                                        cv2.MORPH_OPEN, open_ker)
        #crete kernel - closing
        close_ker = cv2.getStructuringElement(cv2.MORPH_CLOSE, (close_ker_size,close_ker_size))
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
        gray_data_frame = frame[grayscale_frame==255]
        collect_gray_data_for_hist = np.append(collect_gray_data_for_hist, \
                                               np.expand_dims(gray_data_frame,1),axis=0)

    #initialize the kde
    kde = sklearn.neighbors.KernelDensity(bandwidth=0.3, kernel='gaussian',atol=0.000000005)
    #use the data collected
    kde.fit(collect_kde_data_frames)

    #initialize histogram, using the data collected
    ps, bins = np.histogram(collect_gray_data_for_hist.flatten(),bins=30, density=True)
    bins_diff = bins[1]-bins[0]
    #calculate the pdf
    pdf = bins_diff*np.cumsum(ps)

    #threshold acoording to desired low and high pdf thresholds
    #define histogram's edge values
    #low
    pdf_low_th = 0.001
    low_cond = np.where(pdf<pdf_low_th)
    hist_min_val = 0
    if (len(low_cond[0])>0): #need to update histogram minimum value
        #take the maximum out of the bins in the condition
        hist_min_val = bins[np.max(low_cond)]
    
    #high
    pdf_high_th = 0.999
    high_cond = np.where(pdf>pdf_high_th)
    hist_max_val = 255
    if (len(high_cond[0])>0): #need to update histogram minimum value
        #take the minimum out of the bins in the condition
        hist_max_val = bins[np.max(high_cond)]


    #now we run of 