FROM python:3.10-slim

RUN apt update && apt install -y ffmpeg

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir torch torchaudio transformers scipy fastapi uvicorn

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8081"]
