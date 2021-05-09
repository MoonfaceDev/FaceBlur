from dlib import correlation_tracker, rectangle


def start_trackers(img, faces):
    """
    :param img: Frame in RGB color format
    :param faces: Locations list of faces to blur
    :return: Trackers list
    """
    trackers = []

    for face in faces:
        tracker = correlation_tracker()
        rect = rectangle(face[0], face[1], face[2], face[3])
        tracker.start_track(img, rect)
        trackers.append(tracker)

    return trackers


def update_locations(trackers, img):
    """
    :param trackers: Trackers list
    :param img: Frame in RGB color format
    :return: Updated locations list of faces to blur
    """
    face_locations = []
    for tracker in trackers:
        tracker.update(img)
        pos = tracker.get_position()

        left, top, right, bottom = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
        face_locations.append((left, top, right, bottom))
    return face_locations
