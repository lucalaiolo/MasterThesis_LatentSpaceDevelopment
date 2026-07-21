"""COCO-18 / OpenPose skeleton for the YouTube 2D-keypoint dataset.

The dataset (see ``data.py``) stores one row per joint per frame, ordered by
a ``part_idx`` column that runs 0..17 in the OpenPose "COCO" layout:

    0  Nose        6  LElbow      12 LKnee
    1  Neck        7  LWrist      13 LAnkle
    2  RShoulder   8  RHip        14 REye
    3  RElbow      9  RKnee       15 LEye
    4  RWrist      10 RAnkle      16 REar
    5  LShoulder   11 LHip        17 LEar

Everything downstream is generic in the joint count ``J``; these constants
only fix the *meaning* of each index so the limb-masking policy, the
root-centring, and any visualisation refer to the right joints. Coordinates
are 2D (image plane), so ``n_dims = 2`` everywhere.
"""

from __future__ import annotations

# Joint names in ``part_idx`` order (index i == part_idx i).
COCO18_KEYPOINT_NAMES: list[str] = [
    "Nose",       # 0
    "Neck",       # 1
    "RShoulder",  # 2
    "RElbow",     # 3
    "RWrist",     # 4
    "LShoulder",  # 5
    "LElbow",     # 6
    "LWrist",     # 7
    "RHip",       # 8
    "RKnee",      # 9
    "RAnkle",     # 10
    "LHip",       # 11
    "LKnee",      # 12
    "LAnkle",     # 13
    "REye",       # 14
    "LEye",       # 15
    "REar",       # 16
    "LEar",       # 17
]

N_JOINTS: int = len(COCO18_KEYPOINT_NAMES)          # 18
N_DIMS: int = 2                                      # image-plane (x, y)

COCO18_NAME_TO_IDX: dict[str, int] = {
    name: i for i, name in enumerate(COCO18_KEYPOINT_NAMES)
}

# Root joint for per-frame centring ([preprocess]). The Neck is the OpenPose
# body centre and is detected far more reliably than a synthesised mid-hip.
ROOT_JOINT: int = COCO18_NAME_TO_IDX["Neck"]        # 1

# Two joints whose distance is a stable per-clip scale (torso length): Neck
# to the mid-hip. There is no mid-hip keypoint in COCO-18, so RHip is used as
# a robust proxy; ``data.torso_scale`` can also take the true mid-hip.
TORSO_JOINTS: tuple[int, int] = (
    COCO18_NAME_TO_IDX["Neck"], COCO18_NAME_TO_IDX["RHip"],
)

# Limb groups for the ``limb`` masking policy ([MVAE §2.6]): one whole limb is
# hidden per clip. The four limbs plus the head cover the informative joints.
COCO18_LIMBS: dict[str, list[int]] = {
    "right_arm": [2, 3, 4],
    "left_arm":  [5, 6, 7],
    "right_leg": [8, 9, 10],
    "left_leg":  [11, 12, 13],
    "head":      [0, 14, 15, 16, 17],
}

# Bilateral (left, right) joint pairs for the symmetry / laterality analyses
# ([vae_analysis §15]). Midline joints (Nose, Neck) are deliberately omitted.
COCO18_LEFT_RIGHT: list[tuple[int, int]] = [
    (5, 2),    # LShoulder <-> RShoulder
    (6, 3),    # LElbow    <-> RElbow
    (7, 4),    # LWrist    <-> RWrist
    (11, 8),   # LHip      <-> RHip
    (12, 9),   # LKnee     <-> RKnee
    (13, 10),  # LAnkle    <-> RAnkle
    (15, 14),  # LEye      <-> REye
    (17, 16),  # LEar      <-> REar
]

# Bone connectivity (0-indexed joint pairs), the standard OpenPose COCO pose
# graph. Used only for visualisation / structural analysis, never in training.
COCO18_BONES: list[tuple[int, int]] = [
    (1, 2), (1, 5),            # neck -> shoulders
    (2, 3), (3, 4),            # right arm
    (5, 6), (6, 7),            # left arm
    (1, 8), (8, 9), (9, 10),   # right side + leg
    (1, 11), (11, 12), (12, 13),  # left side + leg
    (1, 0),                    # neck -> nose
    (0, 14), (14, 16),         # nose -> right eye -> right ear
    (0, 15), (15, 17),         # nose -> left eye -> left ear
]


def coco18_limbs() -> dict[str, list[int]]:
    """Return a fresh copy of the limb->indices map (safe to mutate)."""
    return {name: list(idx) for name, idx in COCO18_LIMBS.items()}


# ===========================================================================
# BODY-15 layout (the current preprocessing export).
# ===========================================================================
# The newer export drops the four face joints (eyes / ears) and *inserts a
# MidHip* at index 8, shifting the whole lower body down by one relative to
# COCO-18. It is the OpenPose BODY-25 torso+limbs subset:
#
#     0  Nose        5  LShoulder   10 RKnee
#     1  Neck        6  LElbow      11 RAnkle
#     2  RShoulder   7  LWrist      12 LHip
#     3  RElbow      8  MidHip      13 LKnee
#     4  RWrist      9  RHip        14 LAnkle
#
# ``part_idx`` in the CSV matches these indices exactly.
BODY15_KEYPOINT_NAMES: list[str] = [
    "Nose",       # 0
    "Neck",       # 1
    "RShoulder",  # 2
    "RElbow",     # 3
    "RWrist",     # 4
    "LShoulder",  # 5
    "LElbow",     # 6
    "LWrist",     # 7
    "MidHip",     # 8
    "RHip",       # 9
    "RKnee",      # 10
    "RAnkle",     # 11
    "LHip",       # 12
    "LKnee",      # 13
    "LAnkle",     # 14
]

BODY15_NAME_TO_IDX: dict[str, int] = {
    name: i for i, name in enumerate(BODY15_KEYPOINT_NAMES)
}

# Neck is still the most reliable body centre (and still index 1); the MidHip
# now gives a *true* torso length (Neck -> MidHip) rather than the RHip proxy
# COCO-18 had to use.
BODY15_ROOT_JOINT: int = BODY15_NAME_TO_IDX["Neck"]        # 1
BODY15_TORSO_JOINTS: tuple[int, int] = (
    BODY15_NAME_TO_IDX["Neck"], BODY15_NAME_TO_IDX["MidHip"],  # (1, 8)
)

BODY15_LIMBS: dict[str, list[int]] = {
    "right_arm": [2, 3, 4],
    "left_arm":  [5, 6, 7],
    "right_leg": [9, 10, 11],
    "left_leg":  [12, 13, 14],
    "head":      [0],
}

BODY15_LEFT_RIGHT: list[tuple[int, int]] = [
    (5, 2),    # LShoulder <-> RShoulder
    (6, 3),    # LElbow    <-> RElbow
    (7, 4),    # LWrist    <-> RWrist
    (12, 9),   # LHip      <-> RHip
    (13, 10),  # LKnee     <-> RKnee
    (14, 11),  # LAnkle    <-> RAnkle
]

BODY15_BONES: list[tuple[int, int]] = [
    (1, 2), (1, 5),               # neck -> shoulders
    (2, 3), (3, 4),               # right arm
    (5, 6), (6, 7),               # left arm
    (1, 8),                       # neck -> mid-hip (spine)
    (8, 9), (9, 10), (10, 11),    # mid-hip -> right leg
    (8, 12), (12, 13), (13, 14),  # mid-hip -> left leg
    (1, 0),                       # neck -> nose
]


def body15_limbs() -> dict[str, list[int]]:
    """Return a fresh copy of the BODY-15 limb->indices map (safe to mutate)."""
    return {name: list(idx) for name, idx in BODY15_LIMBS.items()}


def skeleton_for(n_joints: int) -> dict:
    """Resolve the skeleton constants for a given joint count.

    Returns a dict with ``names, name_to_idx, root_joint, torso_joints, limbs,
    left_right, bones`` for the 15- or 18-joint layout. Anything else raises,
    so a mismatched export fails loudly instead of silently mis-wiring the
    limbs and bones.
    """
    if n_joints == 15:
        return {
            "names": list(BODY15_KEYPOINT_NAMES),
            "name_to_idx": dict(BODY15_NAME_TO_IDX),
            "root_joint": BODY15_ROOT_JOINT,
            "torso_joints": BODY15_TORSO_JOINTS,
            "limbs": body15_limbs(),
            "left_right": list(BODY15_LEFT_RIGHT),
            "bones": list(BODY15_BONES),
        }
    if n_joints == 18:
        return {
            "names": list(COCO18_KEYPOINT_NAMES),
            "name_to_idx": dict(COCO18_NAME_TO_IDX),
            "root_joint": ROOT_JOINT,
            "torso_joints": TORSO_JOINTS,
            "limbs": coco18_limbs(),
            "left_right": list(COCO18_LEFT_RIGHT),
            "bones": list(COCO18_BONES),
        }
    raise ValueError(
        f"no skeleton defined for J={n_joints}; expected 15 (BODY-15) or "
        f"18 (COCO-18). Add one to youtube_motion/skeleton.py.")
