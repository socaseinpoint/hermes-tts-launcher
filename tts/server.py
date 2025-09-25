from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoProcessor, CsmForConditionalGeneration
from scipy.io.wavfile import write
import torch
import uuid
import os

app = FastAPI()

model_id = "sesame/csm-1b"
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
model = CsmForConditionalGeneration.from_pretrained(model_id, token=hf_token).to("cuda" if torch.cuda.is_available() else "cpu")

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
def tts(req: TTSRequest):
    inputs = processor(req.text, return_tensors="pt").to(model.device)
    audio = model.generate(**inputs, output_audio=True)

    temp_path = f"/tmp/{uuid.uuid4().hex}.wav"
    processor.save_audio(audio, temp_path)

    with open(temp_path, "rb") as f:
        data = f.read()
    os.remove(temp_path)

    return Response(content=data, media_type="audio/wav")
