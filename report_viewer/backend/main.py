import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

# Adjust if your data folder is elsewhere
PRETRAINED_OUTPUTS = "../../outputs/yolo26-pretrained-visdrone/"
FINETUNED_OUTPUTS = "../../outputs/yolo26-finetuned-visdrone/"

PRETRAINED_MODEL_NAME = "Pretrained Model"
FINETUNED_MODEL_NAME = "Fine-tuned Model"

active_directory = FINETUNED_OUTPUTS

def get_dirs():
    return (
        os.path.join(active_directory, "ground_truth"),
        os.path.join(active_directory, "detections"),
        os.path.join(active_directory, "reports")
    )

def fmt(item_id: str):
    return f"{int(item_id):05d}"

# CORS so React can talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/items")
def list_items():
    GT_DIR, _, _ = get_dirs()
    files = os.listdir(GT_DIR)
    raw = [Path(f).stem for f in files]
    return sorted([str(int(r)) for r in raw], key=lambda x: int(x))

@app.post("/toggle")
def toggle_dataset():
    global active_directory
    active_directory = FINETUNED_OUTPUTS if active_directory == PRETRAINED_OUTPUTS else PRETRAINED_OUTPUTS
    return {"active": active_directory}

@app.get("/ground_truth/{item_id}")
def get_ground_truth(item_id: str):
    GT_DIR, _, _ = get_dirs()
    file = os.path.join(GT_DIR, f"{fmt(item_id)}.png")
    if not os.path.exists(file):
        raise HTTPException(404, "Ground truth image not found")
    return FileResponse(file)

@app.get("/predictions/{item_id}")
def get_prediction(item_id: str):
    _, PRED_DIR, _ = get_dirs()
    file = os.path.join(PRED_DIR, f"{fmt(item_id)}.png")
    if not os.path.exists(file):
        raise HTTPException(404, "Prediction image not found")
    return FileResponse(file)

@app.get("/reports/{item_id}")
def get_report(item_id: str):
    _, _, REPORT_DIR = get_dirs()
    file = os.path.join(REPORT_DIR, f"{fmt(item_id)}.json")
    if not os.path.exists(file):
        raise HTTPException(404, "Report not found")
    with open(file, "r") as f:
        text = json.load(f)
    return JSONResponse(text)

@app.get("/model_name")
def model_name():
    return {
        "name": PRETRAINED_MODEL_NAME if active_directory == PRETRAINED_OUTPUTS else FINETUNED_MODEL_NAME
    }

@app.get("/model_metrics")
def get_model_metrics():
    metrics_path = os.path.join(active_directory, "metrics.json")

    if not os.path.exists(metrics_path):
        return {"error": "metrics.json not found"}

    with open(metrics_path, "r") as f:
        return json.load(f)

