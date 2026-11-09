from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from .fall_logic import FallStateMachine, FallConfig, FallState

# COCO-17 skeleton edges.
SKELETON = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16)]

class FallDetector:
    def __init__(self, model_path="yolo11n-pose.pt", confidence=.30, iou=.50,
                 imgsz=640, device="cpu", fall_config=None,
                 draw_skeleton=True, draw_floor_line=True):
        if not Path(model_path).exists() and not str(model_path).endswith(".pt"):
            raise FileNotFoundError(f"Pose model not found: {model_path}")
        self.model=YOLO(str(model_path))
        self.confidence=confidence; self.iou=iou; self.imgsz=imgsz; self.device=device
        self.logic=FallStateMachine(fall_config or FallConfig())
        self.draw_skeleton=draw_skeleton; self.draw_floor_line=draw_floor_line
        self.next_id=1
        self.box_to_id={}

    def _track_id(self, idx, boxes):
        if boxes.id is not None:
            return int(boxes.id[idx].item())
        # Fallback ID for models/results without tracker IDs: stable within this frame only.
        return self.next_id + idx

    def process_frame(self, frame):
        result=self.model.predict(frame,conf=self.confidence,iou=self.iou,
                                  imgsz=self.imgsz,device=self.device,verbose=False)[0]
        canvas=frame.copy(); h,w=frame.shape[:2]
        floor_y=int(h*self.logic.cfg.floor_y_ratio)
        if self.draw_floor_line:
            cv2.line(canvas,(0,floor_y),(w,floor_y),(255,180,0),1)
            cv2.putText(canvas,"FLOOR REFERENCE",(8,max(16,floor_y-7)),
                        cv2.FONT_HERSHEY_SIMPLEX,.42,(255,180,0),1)
        people=[]
        boxes=result.boxes
        if boxes is not None and result.keypoints is not None:
            xyxy=boxes.xyxy.cpu().numpy()
            kpts=result.keypoints.data.cpu().numpy()
            for i,box in enumerate(xyxy):
                track_id=self._track_id(i,boxes)
                state,metrics,new_event=self.logic.update(track_id,box,kpts[i],frame.shape)
                x1,y1,x2,y2=map(int,box)
                alert=state==FallState.FALLEN
                color=(0,0,255) if alert else (40,220,70)
                cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                label=f"ID {track_id} | {state.value}"
                cv2.rectangle(canvas,(x1,max(0,y1-24)),(min(w,x1+len(label)*9+8),y1),color,-1)
                cv2.putText(canvas,label,(x1+4,max(16,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.48,(10,10,10),1,cv2.LINE_AA)
                if self.draw_skeleton:
                    points=kpts[i]
                    for a,b in SKELETON:
                        if len(points)>max(a,b) and points[a,2]>.2 and points[b,2]>.2:
                            pa=tuple(points[a,:2].astype(int)); pb=tuple(points[b,:2].astype(int))
                            cv2.line(canvas,pa,pb,(0,255,255),2)
                    for p in points:
                        if p[2]>.2: cv2.circle(canvas,tuple(p[:2].astype(int)),3,(255,80,20),-1)
                if new_event:
                    people.append({"track_id":track_id,"event":"FALL_CONFIRMED","metrics":metrics})
        active_fall=any(s.state==FallState.FALLEN for s in self.logic.tracks.values())
        if active_fall:
            cv2.rectangle(canvas,(0,0),(w,34),(0,0,210),-1)
            cv2.putText(canvas,"FALL ALERT",(12,24),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2)
        return canvas,people,active_fall
