# imports

from pkg_resources import resource_filename
from cv2.dnn import readNetFromCaffe
from dlib import shape_predictor, face_recognition_model_v1


def detection_dnn_architecture_location():
    return resource_filename(__name__, "models/deploy.prototxt.txt")


def detection_dnn_model_location():
    return resource_filename(__name__, "models/res10_300x300_ssd_iter_140000.caffemodel")


def pose_predictor_five_point_model_location():
    return resource_filename(__name__, "models/shape_predictor_5_face_landmarks.dat")


def face_recognition_model_location():
    return resource_filename(__name__, "models/dlib_face_recognition_resnet_model_v1.dat")


def detection_dnn_model():
    return readNetFromCaffe(detection_dnn_architecture_location(), detection_dnn_model_location())


def pose_predictor():
    return shape_predictor(pose_predictor_five_point_model_location())


def face_encoder():
    return face_recognition_model_v1(face_recognition_model_location())
