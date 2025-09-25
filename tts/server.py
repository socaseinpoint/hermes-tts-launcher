from fastapi import FastAPI, Query
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile
import torch
import uuid

app = FastAPI()
model_id = "sesame/cms-tts-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = BarkModel.from_pretrained(model_id)

@app.post("/tts")
def tts(text: str = Query(...)):
    inputs = processor(text, return_tensors="pt")
    audio = model.generate(**inputs)
    file_path = f"/tmp/{uuid.uuid4().hex}.wav"
    scipy.io.wavfile.write(file_path, model.generation_config.sample_rate, audio[0].numpy())
    return {"file": file_path}
