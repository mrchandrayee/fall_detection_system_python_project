from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math
import numpy as np

class FallState(str, Enum):
    UNKNOWN = "UNKNOWN"
    STANDING = "STANDING"
    FALLING = "FALLING"
    FALLEN = "FALLEN"
    RECOVERING = "RECOVERING"

@dataclass
class FallConfig:
    floor_y_ratio: float = 0.90
    fallen_aspect_ratio: float = 1.15
    torso_angle_threshold: float = 55.0
    fall_confirm_frames: int = 4
    recover_confirm_frames: int = 8
    min_keypoint_confidence: float = 0.20

@dataclass
class PersonTrackState:
    state: FallState = FallState.UNKNOWN
    fall_frames: int = 0
    upright_frames: int = 0
    previous_center: Optional[tuple] = None
    last_metrics: dict = field(default_factory=dict)

class FallStateMachine:
    """Temporal fall-state estimator. Inputs are per-frame pose measurements."""
    def __init__(self, config: FallConfig | None = None):
        self.cfg = config or FallConfig()
        self.tracks: dict[int, PersonTrackState] = {}

    @staticmethod
    def _angle_from_vertical(a, b):
        dx, dy = float(b[0]-a[0]), float(b[1]-a[1])
        return abs(math.degrees(math.atan2(dx, max(1e-6, abs(dy)))))

    def update(self, track_id: int, bbox, keypoints, frame_shape):
        h, w = frame_shape[:2]
        x1,y1,x2,y2 = map(float,bbox)
        bw,bh = max(1.0,x2-x1),max(1.0,y2-y1)
        cx,cy=(x1+x2)/2,(y1+y2)/2
        aspect=bw/bh
        floor_y=h*self.cfg.floor_y_ratio
        state=self.tracks.setdefault(int(track_id),PersonTrackState())

        # COCO keypoint order: nose, eyes, ears, shoulders(5,6), elbows,
        # wrists, hips(11,12), knees, ankles(15,16).
        pts=np.asarray(keypoints,dtype=float)
        def valid(i):
            return pts.ndim==2 and len(pts)>i and pts.shape[1]>=3 and pts[i,2]>=self.cfg.min_keypoint_confidence
        shoulder=None; hip=None; torso_angle=0.0
        if valid(5) and valid(6): shoulder=((pts[5,:2]+pts[6,:2])/2)
        if valid(11) and valid(12): hip=((pts[11,:2]+pts[12,:2])/2)
        if shoulder is not None and hip is not None:
            torso_angle=self._angle_from_vertical(shoulder,hip)
        else:
            # Fallback based on box orientation if pose points are unavailable.
            torso_angle=90.0 if aspect>1.0 else 0.0

        ankle_y=float(np.mean([pts[i,1] for i in (15,16) if valid(i)])) if valid(15) or valid(16) else y2
        near_floor=(cy > floor_y*0.82 or ankle_y > floor_y*0.90)
        horizontal=aspect >= self.cfg.fallen_aspect_ratio
        tilted=torso_angle >= self.cfg.torso_angle_threshold
        # Require more than one cue to reduce false positives.
        fall_candidate=(horizontal and (tilted or near_floor)) or (tilted and near_floor and aspect>0.75)

        old=state.state
        if fall_candidate:
            state.fall_frames += 1
            state.upright_frames = 0
            state.state = FallState.FALLEN if state.fall_frames >= self.cfg.fall_confirm_frames else FallState.FALLING
        else:
            state.upright_frames += 1
            state.fall_frames = 0
            if old in (FallState.FALLEN, FallState.FALLING):
                state.state = FallState.RECOVERING
                if state.upright_frames >= self.cfg.recover_confirm_frames:
                    state.state = FallState.STANDING
            else:
                state.state = FallState.STANDING

        state.previous_center=(cx,cy)
        state.last_metrics={"bbox_aspect":round(aspect,3),"torso_angle":round(torso_angle,1),
                            "near_floor":bool(near_floor),"floor_y":round(floor_y,1)}
        newly_fallen = state.state == FallState.FALLEN and old != FallState.FALLEN
        return state.state, state.last_metrics, newly_fallen
