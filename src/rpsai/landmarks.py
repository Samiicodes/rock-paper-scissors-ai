'''Hand landmarks in a standard position, so that a classifier only sees the hand's shape.

The web game repeats these steps in web/src/gesture.js on the landmarks MediaPipe finds in the
webcam image. Both are checked against the same fixtures.
'''

import numpy as np

WRIST = 0
INDEX_BASE = 5
MIDDLE_BASE = 9
PINKY_BASE = 17
FINGER_TIPS = (8, 12, 16, 20)
FINGER_MIDDLE_JOINTS = (6, 10, 14, 18)
# The wrist is always at the origin after canonicalisation, so it carries no information
FEATURE_COUNT = 40


def canonicalise(points: np.ndarray, aspect: float | np.ndarray = 1.0) -> np.ndarray:
    '''Move hands into a standard position and size.

    Args:
        points: landmarks of shape (..., 21, 2) in image coordinates scaled to [0, 1] by the
            image width and height, as MediaPipe returns them.
        aspect: image height divided by width, one value or one per hand. Scaling y by it makes
            both axes use the same unit.

    Returns:
        Landmarks of the same shape with the wrist at the origin, the base of the middle finger
        at (0, -1), and the index finger on the positive x side. The last step makes left and
        right hands, and palms and backs, look alike.
    '''
    hands = np.array(points, dtype=np.float64)
    hands[..., 1] *= np.asarray(aspect, dtype=np.float64)[..., np.newaxis]
    hands -= hands[..., WRIST : WRIST + 1, :]
    axis = hands[..., MIDDLE_BASE : MIDDLE_BASE + 1, :]
    length = np.maximum(np.linalg.norm(axis, axis=-1), 1e-9)
    # Per hand: the direction from the wrist to the middle finger, as a unit vector (cos, sin)
    cos = axis[..., 0] / length
    sin = axis[..., 1] / length
    x = hands[..., 0]
    y = hands[..., 1]
    # A rotation that takes that direction to (0, -1), followed by scaling it to length 1
    rotated_x = (cos * y - sin * x) / length
    rotated_y = -(cos * x + sin * y) / length
    mirrored = rotated_x[..., INDEX_BASE] < rotated_x[..., PINKY_BASE]
    rotated_x = np.where(mirrored[..., np.newaxis], -rotated_x, rotated_x)
    return np.stack([rotated_x, rotated_y], axis=-1)


def features(points: np.ndarray, aspect: float | np.ndarray = 1.0) -> np.ndarray:
    '''Flatten canonical landmarks into the classifier's input, without the wrist.'''
    canonical = canonicalise(points, aspect)
    return canonical[..., 1:, :].reshape(*canonical.shape[:-2], FEATURE_COUNT)


def random_stretches(count: int, stretch_range: tuple[float, float], seed: int) -> np.ndarray:
    '''Draw vertical stretch factors evenly on a log scale, so 0.5 is as likely as 2.'''
    low, high = stretch_range
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(low), np.log(high), size=count))


def extended_fingers(points: np.ndarray, aspect: float | np.ndarray = 1.0) -> np.ndarray:
    '''Return which of the four fingers are straight.

    A finger counts as straight when its tip is further from the wrist than its middle joint.
    The thumb is ignored because it barely changes between the three moves.
    '''
    canonical = canonicalise(points, aspect)
    tips = np.linalg.norm(canonical[..., FINGER_TIPS, :], axis=-1)
    joints = np.linalg.norm(canonical[..., FINGER_MIDDLE_JOINTS, :], axis=-1)
    return np.asarray(tips > joints)
