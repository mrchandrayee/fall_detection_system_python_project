import argparse
from .processing import process_video
def main():
    p=argparse.ArgumentParser(description="YOLO pose-based fall detection")
    p.add_argument("--source",required=True,help="Video path or webcam index, e.g. 0")
    p.add_argument("--model",default="yolo11n-pose.pt")
    p.add_argument("--confidence",type=float,default=.30)
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",default="outputs/fall_detection.mp4")
    a=p.parse_args()
    src=int(a.source) if a.source.isdigit() else a.source
    print(process_video(src,a.output,a.model,confidence=a.confidence,device=a.device))
if __name__=="__main__": main()
