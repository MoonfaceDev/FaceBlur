import numpy as np
from dlib import rectangle


def get_face_landmarks(face_image, face_locations, pose_predictor):
    """
    :param face_image: The image that contains one or more faces
    :param face_locations: The bounding boxes of each face
    :param pose_predictor: Initialized pose predictor model
    :return: List of 5 face landmarks for each face
    """
    face_rects = [rectangle(face[0], face[1], face[2], face[3]) for face in face_locations]
    return [pose_predictor(face_image, face_rect) for face_rect in face_rects]


def encode_faces(face_image, face_locations, pose_predictor, face_encoder, num_jitters=1):
    """
    :param face_image: The image that contains one or more faces
    :param face_locations: The bounding boxes of each face
    :param pose_predictor: Initialized pose predictor model
    :param face_encoder: Initialized face encoder model
    :param num_jitters: How many times to re-sample the face
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    landmarks = get_face_landmarks(face_image, face_locations, pose_predictor)
    encodings = np.array([face_encoder.compute_face_descriptor(face_image, landmark_set, num_jitters) for
                          landmark_set in landmarks])
    return encodings


def face_distance(face_encodings, face_to_compare):
    """
    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: Face distances array
    """
    if face_encodings.shape[0] == 0:
        return np.empty(0)

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def compare_faces(face_encodings, face_encoding_to_check, tolerance=0.5):
    """
    :param face_encodings: List of known face encodings
    :param face_encoding_to_check: Single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(face_distance(face_encodings, face_encoding_to_check) <= tolerance)


def match_faces(face_encodings, match_encodings, tolerance=0.5):
    matched_indices = []
    matched_encodings = []

    for i, face_encoding in enumerate(face_encodings):
        matches = compare_faces(match_encodings, face_encoding, tolerance)
        if sum(matches) > 0:
            matched_indices.append(i)
            matched_encodings.append(face_encoding)

    return matched_indices, matched_encodings


def exclude_faces(face_encodings, match_encodings, tolerance=0.5):
    matched_indices = []
    matched_encodings = []

    for i, face_encoding in enumerate(face_encodings):
        matches = compare_faces(match_encodings, face_encoding, tolerance)
        if sum(matches) == 0:
            matched_indices.append(i)
            matched_encodings.append(face_encoding)

    return matched_indices, matched_encodings