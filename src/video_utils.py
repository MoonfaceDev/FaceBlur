# imports

import cv2


def initialize_writer(path, size, frame_rate):
    """
    :param path: Path to output video
    :param size: Video resolution
    :param frame_rate: The number of frames displayed per second
    :return: VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # output video format
    return cv2.VideoWriter(path, fourcc, frame_rate, size)


def display_video(path):
    """
    :param path: Path to video file
    """
    # get video
    video = cv2.VideoCapture(path)  # output VideoCapture object
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))  # frames per second in input video
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # number of frames in input video
    frame_length = int(1000/frame_rate)  # time for each frame, in milliseconds
    # get first frame
    ret0, img0 = video.read()  # ret0 indicates if first frame was read correctly, img0 is first frame
    if not ret0:
        return
    # initialize display window
    cv2.namedWindow('Video Preview', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Video Preview', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Video Preview', img0)
    cv2.waitKey()

    for i in range(frame_count):
        ret, img = video.read()  # ret indicates if frame was read correctly, img is last read frame
        if not ret:
            break

        cv2.imshow('Video Preview', img)
        k = cv2.waitKey(frame_length) & 0xff  # pressed key
        if k == 27:  # press 'ESC' to quit
            break
    video.release()
    cv2.destroyAllWindows()
