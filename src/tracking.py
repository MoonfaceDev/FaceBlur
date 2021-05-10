# imports

from dlib import correlation_tracker, rectangle


def start_trackers(img, faces):
    """
    Initializes a tracker for each face in the image
    :param img: Frame in RGB color format
    :param faces: Locations list of faces to blur
    :return: Trackers list
    """
    trackers = []  # list of tracker objects, one for each matched face

    for face in faces:
        tracker = correlation_tracker()  #  face object tracker
        rect = rectangle(face[0], face[1], face[2], face[3])  # face bounding box
        tracker.start_track(img, rect)
        trackers.append(tracker)

    return trackers


def update_locations(trackers, img):
    """
    Updates the trackers with current frame
    :param trackers: Trackers list
    :param img: Frame in RGB color format
    :return: Updated locations list of faces to blur
    """
    face_locations = []  # list of updated face locations
    for tracker in trackers:
        tracker.update(img)
        pos = tracker.get_position()  # updated bounding box

        left, top, right, bottom = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())  # updated face coordinates
        face_locations.append((left, top, right, bottom))
    return face_locations
