from pathlib import Path
import cv2
from .detector import FallDetector
from .fall_logic import FallConfig

def process_video(source,destination,model_path="yolo11n-pose.pt",confidence=.30,
                  iou=.50,imgsz=640,device="cpu",fall_config=None,draw_skeleton=True):
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise ValueError(f"Cannot open video source: {source}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w<=0 or h<=0:
        cap.release(); raise ValueError("Invalid video dimensions")
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    detector=FallDetector(model_path,confidence,iou,imgsz,device,fall_config,draw_skeleton)
    frame_count=fall_events=alert_frames=0; events=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            annotated,new_events,alert=detector.process_frame(frame)
            if alert: alert_frames+=1
            for event in new_events:
                event["time_seconds"]=round(frame_count/fps,2)
                events.append(event); fall_events+=1
            writer.write(annotated); frame_count+=1
    finally:
        cap.release(); writer.release()
    return {"frames_processed":frame_count,"duration_seconds":round(frame_count/fps,2),
            "fall_events":fall_events,"alert_frames":alert_frames,"events":events}
