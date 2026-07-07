"""Pre-built Skeleton definitions for the datasets we use.

`mhr70()` returns a fully populated Skeleton for the MHR-70 layout —
70 whole-body keypoints (face, torso, legs, feet, both hands, elbow
markers, acromia, neck). Bones come from the standard MHR-70 link
list; left-right pairs are derived from the `left_` / `right_` naming
convention; limbs are grouped so the analysis toolkit's kinematic
features produce meaningful laterality and upper-lower balance signals.

If you have a different skeleton, use this module as a template.
"""

from __future__ import annotations

from .interfaces import Skeleton


MHR70_KEYPOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    "right_thumb4", "right_thumb3", "right_thumb2",
    "right_thumb_third_joint",
    "right_forefinger4", "right_forefinger3", "right_forefinger2",
    "right_forefinger_third_joint",
    "right_middle_finger4", "right_middle_finger3", "right_middle_finger2",
    "right_middle_finger_third_joint",
    "right_ring_finger4", "right_ring_finger3", "right_ring_finger2",
    "right_ring_finger_third_joint",
    "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2",
    "right_pinky_finger_third_joint",
    "right_wrist",
    "left_thumb4", "left_thumb3", "left_thumb2",
    "left_thumb_third_joint",
    "left_forefinger4", "left_forefinger3", "left_forefinger2",
    "left_forefinger_third_joint",
    "left_middle_finger4", "left_middle_finger3", "left_middle_finger2",
    "left_middle_finger_third_joint",
    "left_ring_finger4", "left_ring_finger3", "left_ring_finger2",
    "left_ring_finger_third_joint",
    "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2",
    "left_pinky_finger_third_joint",
    "left_wrist",
    "left_olecranon", "right_olecranon",
    "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck",
]

_N2I: dict[str, int] = {n: i for i, n in enumerate(MHR70_KEYPOINT_NAMES)}


_MHR70_SKELETON_LINKS: list[tuple[str, str]] = [
    ("left_ankle", "left_knee"), ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
    ("left_eye", "right_eye"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
    ("left_ankle", "left_big_toe"), ("left_ankle", "left_small_toe"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"),
    ("right_ankle", "right_heel"),
    # Left hand.
    ("left_wrist", "left_thumb_third_joint"),
    ("left_thumb_third_joint", "left_thumb2"),
    ("left_thumb2", "left_thumb3"), ("left_thumb3", "left_thumb4"),
    ("left_wrist", "left_forefinger_third_joint"),
    ("left_forefinger_third_joint", "left_forefinger2"),
    ("left_forefinger2", "left_forefinger3"),
    ("left_forefinger3", "left_forefinger4"),
    ("left_wrist", "left_middle_finger_third_joint"),
    ("left_middle_finger_third_joint", "left_middle_finger2"),
    ("left_middle_finger2", "left_middle_finger3"),
    ("left_middle_finger3", "left_middle_finger4"),
    ("left_wrist", "left_ring_finger_third_joint"),
    ("left_ring_finger_third_joint", "left_ring_finger2"),
    ("left_ring_finger2", "left_ring_finger3"),
    ("left_ring_finger3", "left_ring_finger4"),
    ("left_wrist", "left_pinky_finger_third_joint"),
    ("left_pinky_finger_third_joint", "left_pinky_finger2"),
    ("left_pinky_finger2", "left_pinky_finger3"),
    ("left_pinky_finger3", "left_pinky_finger4"),
    # Right hand.
    ("right_wrist", "right_thumb_third_joint"),
    ("right_thumb_third_joint", "right_thumb2"),
    ("right_thumb2", "right_thumb3"), ("right_thumb3", "right_thumb4"),
    ("right_wrist", "right_forefinger_third_joint"),
    ("right_forefinger_third_joint", "right_forefinger2"),
    ("right_forefinger2", "right_forefinger3"),
    ("right_forefinger3", "right_forefinger4"),
    ("right_wrist", "right_middle_finger_third_joint"),
    ("right_middle_finger_third_joint", "right_middle_finger2"),
    ("right_middle_finger2", "right_middle_finger3"),
    ("right_middle_finger3", "right_middle_finger4"),
    ("right_wrist", "right_ring_finger_third_joint"),
    ("right_ring_finger_third_joint", "right_ring_finger2"),
    ("right_ring_finger2", "right_ring_finger3"),
    ("right_ring_finger3", "right_ring_finger4"),
    ("right_wrist", "right_pinky_finger_third_joint"),
    ("right_pinky_finger_third_joint", "right_pinky_finger2"),
    ("right_pinky_finger2", "right_pinky_finger3"),
    ("right_pinky_finger3", "right_pinky_finger4"),
]

MHR70_EDGES: list[tuple[int, int]] = [
    (_N2I[a], _N2I[b]) for a, b in _MHR70_SKELETON_LINKS
]


def _finger_joints(side: str) -> list[int]:
    """All finger keypoints for one hand — every joint of every finger."""
    fingers = ("thumb", "forefinger", "middle_finger", "ring_finger",
               "pinky_finger")
    tips = ("2", "3", "4", "_third_joint")
    out: list[int] = []
    for f in fingers:
        for tip in tips:
            out.append(_N2I[f"{side}_{f}{tip}"])
    return out


def _left_right_pairs() -> list[tuple[int, int]]:
    """Every `left_X` / `right_X` name pair, as index pairs."""
    pairs: list[tuple[int, int]] = []
    for name, i in _N2I.items():
        if name.startswith("left_"):
            partner = "right_" + name[5:]
            j = _N2I.get(partner)
            if j is not None:
                pairs.append((i, j))
    return pairs


def mhr70() -> Skeleton:
    """Return a fully-populated Skeleton for the MHR-70 keypoint set.

    Bones come from the standard MHR-70 link list. Left-right pairs are
    every keypoint whose name begins with `left_` matched to its
    `right_` twin. Limbs are grouped so `features.kinematic_features`
    produces sensible laterality (arm, leg, hand, foot) and
    upper-lower-balance signals.
    """
    limbs: dict[str, list[int]] = {
        # Arms and legs feed both per-limb speed and, via the "arm"/"leg"
        # substrings, the upper-lower balance feature.
        "left_arm": [
            _N2I["left_shoulder"], _N2I["left_elbow"], _N2I["left_wrist"],
            _N2I["left_olecranon"], _N2I["left_cubital_fossa"],
            _N2I["left_acromion"],
        ],
        "right_arm": [
            _N2I["right_shoulder"], _N2I["right_elbow"], _N2I["right_wrist"],
            _N2I["right_olecranon"], _N2I["right_cubital_fossa"],
            _N2I["right_acromion"],
        ],
        "left_leg": [
            _N2I["left_hip"], _N2I["left_knee"], _N2I["left_ankle"],
        ],
        "right_leg": [
            _N2I["right_hip"], _N2I["right_knee"], _N2I["right_ankle"],
        ],
        # Hands and feet are laterality-only. They don't roll up into
        # the upper-lower balance (no "arm" / "leg" in the name).
        "left_hand":  _finger_joints("left"),
        "right_hand": _finger_joints("right"),
        "left_foot": [
            _N2I["left_big_toe"], _N2I["left_small_toe"], _N2I["left_heel"],
        ],
        "right_foot": [
            _N2I["right_big_toe"], _N2I["right_small_toe"], _N2I["right_heel"],
        ],
    }

    return Skeleton(
        n_joints=len(MHR70_KEYPOINT_NAMES),
        bones=MHR70_EDGES,
        left_right=_left_right_pairs(),
        lateral_axis=0,
        limbs=limbs,
    )
