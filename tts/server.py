from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import torch
import uuid
import os
import io
from TTS.api import TTS

app = FastAPI()

# Initialize Coqui TTS
# You can choose different models from: https://github.com/coqui-ai/TTS#-pretrained-models
model_name = "tts_models/en/vctk/vits"  # Good quality English model
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tts = TTS(model_name=model_name).to(device)
    print(f"TTS model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading TTS model: {e}")
    tts = None

class TTSRequest(BaseModel):
    text: str
    speaker: str = "p225"  # Default speaker for VCTK model

@app.get("/")
def health_check():
    return {"status": "ok", "model": model_name, "device": device}

@app.get("/speakers")
def get_speakers():
    """Get available speakers for the current model"""
    if tts is None:
        raise HTTPException(status_code=500, detail="TTS model not loaded")
    
    try:
        if hasattr(tts.tts, 'speakers'):
            return {"speakers": tts.tts.speakers}
        else:
            return {"message": "This model doesn't support multiple speakers"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
def text_to_speech(req: TTSRequest):
    if tts is None:
        raise HTTPException(status_code=500, detail="TTS model not loaded")
    
    try:
        # Generate audio
        temp_path = f"/tmp/{uuid.uuid4().hex}.wav"
        
        # Check if model supports speakers
        if hasattr(tts.tts, 'speakers') and tts.tts.speakers:
            # Use specified speaker or default
            speaker = req.speaker if req.speaker in tts.tts.speakers else tts.tts.speakers[0]
            tts.tts_to_file(text=req.text, speaker=speaker, file_path=temp_path)
        else:
            # Model doesn't support speakers
            tts.tts_to_file(text=req.text, file_path=temp_path)

        # Read the generated audio file
        with open(temp_path, "rb") as f:
            audio_data = f.read()
        
        # Clean up temporary file
        os.remove(temp_path)

        return Response(content=audio_data, media_type="audio/wav")
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)