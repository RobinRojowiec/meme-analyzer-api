"""

IDE: PyCharm
Project: meme-analyzer-api
Author: Robin
Filename: server.py
Date: 05.06.2020

"""
import io
import os

import requests
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# init webapp
from data_model import AnalysisResponse, AnalysisRequest
from text_model import PretrainedTextModel
from vision_model import PretrainedVisionModel

app = FastAPI()

# allow cors
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init models
vision_model = PretrainedVisionModel()
text_model = PretrainedTextModel()


@app.post("/api/analyze", response_model=AnalysisResponse,
          summary="Analyzes an meme and returns tags/vector from SimSearch")
async def extract_text(request: AnalysisRequest):
    if request is not None:
        response = AnalysisResponse()

        img = download_image(request.image_url)
        tags, vector = vision_model.analyze(img, min_confidence=request.min_confidence)
        response.image_vector = vector.tolist()
        response.tags = [tag[0] for tag in tags]

        if request.text is not None:
            text_vector = text_model.encode(request.text)
            response.text_vector = text_vector.tolist()

        return response
    raise HTTPException(status_code=400, detail="Invalid request")


def download_image(url):
    response = requests.get(url)
    byte_data = io.BytesIO(response.content)
    image_data = Image.open(byte_data)
    image_data = image_data.convert('RGB')
    return image_data

if __name__ == "__main__":
    port = os.getenv("PORT", 8001)
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
