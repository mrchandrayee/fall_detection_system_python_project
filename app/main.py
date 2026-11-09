from pathlib import Path
import shutil,uuid
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from .processing import process_video
from .fall_logic import FallConfig
BASE=Path(__file__).resolve().parent.parent
UPLOADS=BASE/"uploads"; OUTPUTS=BASE/"outputs"
UPLOADS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
app=FastAPI(title="AI Fall Detection")
app.mount("/static",StaticFiles(directory=str(BASE/"app"/"static")),name="static")
@app.get("/",response_class=HTMLResponse)
def home(): return (BASE/"app"/"static"/"index.html").read_text(encoding="utf-8")
@app.post("/api/analyze")
async def analyze(file:UploadFile=File(...),model_path:str=Form("yolo11n-pose.pt"),
                  confidence:float=Form(.30),device:str=Form("cpu"),
                  floor_y_ratio:float=Form(.90),fall_confirm_frames:int=Form(4)):
    ext=Path(file.filename or "").suffix.lower()
    if ext not in {".mp4",".avi",".mov",".mkv"}: raise HTTPException(400,"Upload MP4, AVI, MOV or MKV.")
    if not (model_path=="yolo11n-pose.pt" or Path(model_path).is_file()):
        raise HTTPException(400,"Model file not found.")
    job=uuid.uuid4().hex[:12]; src=UPLOADS/f"{job}{ext}"; out=OUTPUTS/f"{job}_fall_alert.mp4"
    with src.open("wb") as f: shutil.copyfileobj(file.file,f)
    try:
        cfg=FallConfig(floor_y_ratio=floor_y_ratio,fall_confirm_frames=fall_confirm_frames)
        stats=process_video(src,out,model_path=model_path,confidence=confidence,device=device,fall_config=cfg)
        return {"job_id":job,"stats":stats,"result_url":f"/api/result/{job}_fall_alert.mp4",
                "warning":"Prototype posture heuristic. Verify every alert; not safety-certified."}
    except Exception as e: raise HTTPException(500,str(e)) from e
    finally: src.unlink(missing_ok=True)
@app.get("/api/result/{filename}")
def result(filename:str):
    path=OUTPUTS/filename
    if not path.exists() or path.parent!=OUTPUTS: raise HTTPException(404,"Result not found")
    return FileResponse(path,media_type="video/mp4",filename=filename)
