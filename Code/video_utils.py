import cv2

INPUT_VIDEO_PATH = "Inputs/INPUT.avi"
STABILIZED_VIDEO_PATH = "Outputs/stabilize_315398875_315328963.avi"
SUBTRACTED_EXTRACTED_VIDEO_PATH = "Outputs/extracted_315398875_315328963.avi"
SUBTRACTED_BINARY_VIDEO_PATH = "Outputs/binary_315398875_315328963.avi"
INPUT_BACKGROUND_PATH = "Inputs/background.jpg"
MATTED_VIDEO_PATH = "Outputs/matted_315398875_315328963.avi"
ALPHA_VIDEO_PATH = "Outputs/alpha_315398875_315328963.avi"
OUTPUT_VIDEO_PATH = "Outputs/OUTPUT_315398875_315328963.avi"




def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def init_vid_writer(output_path: str, params, isColor):
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), params['fps'], (params['width'], params['height']), isColor)
    return out_writer


def release_videos(videos):
    for v in videos:
        v.release()
