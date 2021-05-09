import cv2


def initialize_writer(path, size, frame_rate):
    """
    :param path: Path to output video
    :param size: Video resolution
    :param frame_rate: The number of frames displayed per second
    :return: VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, frame_rate, size)
    return out


def input_video(path):
    """
    :param path: Path to video file
    :return: VideoCapture object
    """
    video = cv2.VideoCapture(path)
    return video


def display_video(path):
    """
    :param path: Path to video file
    """
    video = input_video(path)
    ret0, img0 = video.read()
    if not ret0:
        return
    cv2.namedWindow('Video Preview', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Video Preview', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Video Preview', img0)
    cv2.waitKey()
    while video.isOpened():

        ret, img = video.read()
        if not ret:
            break

        cv2.imshow('Video Preview', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break
    video.release()
    cv2.destroyAllWindows()
