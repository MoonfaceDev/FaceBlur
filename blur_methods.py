import numpy as np
from cv2 import blur


def blackened(img, faces):
    """
    :param img: Original frame
    :param faces: Face locations list
    :return: Frame with blackened faces
    """
    img_out = img.copy()
    for face in faces:
        img_out[face[1]:face[3], face[0]:face[2]] = np.array([0, 0, 0])
    return img_out


def pixelated(img, faces, d=20):
    """
    :param img: Original frame
    :param faces: Face locations list
    :param d: "Pixel" square's size
    :return: Frame with pixelated faces
    """
    img_out = np.pad(img, ((d, d), (d, d), (0, 0)), mode="edge")
    for face in faces:
        for yy in range(face[1], face[3], d):
            for xx in range(face[0], face[2], d):
                img_out[yy + d:yy + 2 * d, xx + d:xx + 2 * d, :] = img[yy:yy + d, xx:xx + d, :].sum(axis=1).sum(
                    axis=0) / d ** 2
    return img_out[d:-d, d:-d]


def blurred(img, faces, d=20):
    """
    :param img: Original frame
    :param faces: Face locations list
    :param d: Filter's size
    :return: Frame with blurred faces
    """
    img_out = np.copy(img)
    for face in faces:
        h, w, channels = img.shape
        left, top, right, bottom = max(face[0], 0), max(face[1], 0), min(face[2], w), min(face[3], h)
        img_out[top:bottom, left:right] = blur(img[top:bottom, left:right], (d, d))
    return img_out


def focused(img, face, d=20):
    """
    :param img: Original frame
    :param face: Face location
    :param d: Filter's size
    :return: Frame with blurred faces
    """
    h, w, channels = img.shape
    left, top, right, bottom = max(face[0], 0), max(face[1], 0), min(face[2], w), min(face[3], h)
    img_out = blur(img, (d, d))
    img_out[top:bottom, left:right] = img[top:bottom, left:right]
    return img_out
