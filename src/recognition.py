# imports

import numpy as np
from dlib import rectangle


def get_face_landmarks(img, face_locations, pose_predictor):
    """
    Locates face landmarks for each face in the image
    :param img: Frame in RGB color format
    :param face_locations: The bounding boxes of each face
    :param pose_predictor: Initialized pose predictor model
    :return: List of 5 face landmarks for each face
    """
    face_rects = [rectangle(face[0], face[1], face[2], face[3]) for face in face_locations]  # rectangle object for each face
    return [pose_predictor(img, face_rect) for face_rect in face_rects]


def encode_faces(img, face_locations, pose_predictor, face_encoder, num_jitters=1):
    """
    Computes face encodings for each face in the image
    :param img: Frame in RGB color format
    :param face_locations: The bounding boxes of each face
    :param pose_predictor: Initialized pose predictor model
    :param face_encoder: Initialized face encoder model
    :param num_jitters: How many times to re-sample the face
    :return: An array of 128-dimensional face encodings (one for each face in the image)
    """
    landmarks = get_face_landmarks(img, face_locations, pose_predictor)  # 5 face landmarks for each face
    return np.array([face_encoder.compute_face_descriptor(img, landmark_set, num_jitters) for landmark_set in landmarks])


def face_distance(face_encodings, face_encoding_to_compare):
    """
    Computes euclidean distance between the given face and each of the faces in the given array
    :param face_encodings: List of face encodings to compare
    :param face_encoding_to_compare: A face encoding to compare against
    :return: Face distances array
    """
    if face_encodings.shape[0] == 0:
        return np.empty(0)

    return np.linalg.norm(face_encodings - face_encoding_to_compare, axis=1)


def compare_faces(face_encodings, face_encoding_to_compare, tolerance=0.5):
    """
    Finds matches between the given face and each of the faces in the given array
    :param face_encodings: List of face encodings to compare
    :param face_encoding_to_compare: A face encoding to compare against
    :param tolerance: How much distance between faces to consider it a match
    :return: A list of True/False values indicating which of face_encodings match face_encoding_to_compare
    """
    return list(face_distance(face_encodings, face_encoding_to_compare) <= tolerance)


def match_faces(face_encodings, match_encodings, tolerance=0.5):
    """
    Finds which of the given faces match any of the faces in the given faces to match
    :param face_encodings: List of face encodings to match
    :param match_encodings: List of wanted face encodings
    :param tolerance: How much distance between faces to consider it a match
    :return: A list of indices of matching face_encodings and a list of their encodings
    """
    matched_indices = []  # indices of matched faces
    matched_encodings = []  # encodings of matched faces

    for i, face_encoding in enumerate(face_encodings):
        matches = compare_faces(match_encodings, face_encoding, tolerance)  # flags indicating if the face matches one of given faces to match
        if sum(matches) > 0:  # there are matches
            matched_indices.append(i)
            matched_encodings.append(face_encoding)

    return matched_indices, matched_encodings


def exclude_faces(face_encodings, match_encodings, tolerance=0.5):
    """
    Finds which of the given faces do not match any of the faces in the given faces to match
    :param face_encodings: List of face encodings to match
    :param match_encodings: List of unwanted face encodings
    :param tolerance: How much distance between faces to consider it a match
    :return: A list of indices of not matching face_encodings and a list of their encodings
    """
    unmatched_indices = []  # indices of unmatched faces
    unmatched_encodings = []  # encodings of unmatched faces

    for i, face_encoding in enumerate(face_encodings):
        matches = compare_faces(match_encodings, face_encoding, tolerance)  # flags indicating if the face matches one of given faces to match
        if sum(matches) == 0:  # no matches
            unmatched_indices.append(i)
            unmatched_encodings.append(face_encoding)

    return unmatched_indices, unmatched_encodings
