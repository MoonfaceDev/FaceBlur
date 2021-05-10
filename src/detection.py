# imports

import numpy as np
import cv2


def detect_faces(img, net):
    """
    Finds locations of faces in the image
    :param img: Frame in BGR color format
    :param net: Initialized dnn model
    :return: Face locations array
    """
    h, w, _ = img.shape  # image dimensions
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))  # normalized and resized blob object
    net.setInput(blob)
    detections = net.forward()
    face_locations = (detections[0, 0, detections[0, 0, :, 2] > 0.2][:, 3:7] * np.array([w, h, w, h])).astype(int)  # face locations found
    return face_locations
