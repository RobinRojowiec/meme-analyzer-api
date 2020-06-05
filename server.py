"""

IDE: PyCharm
Project: meme-analyzer-api
Author: Robin
Filename: server.py
Date: 05.06.2020

"""
import os

import uvicorn
from fastapi import FastAPI, HTTPException

# init webapp
from data_model import AnalysisResponse, AnalysisRequest

app = FastAPI()


# load config and device


# init model


@app.post("/api/analyze", response_model=AnalysisResponse,
          summary="Analyzes an meme and returns tags/vector from SimSearch")
def extract_text(request: AnalysisRequest):
    if request is not None:
        response = AnalysisResponse()

        return response
    raise HTTPException(status_code=400, detail="Invalid request")


if __name__ == "__main__":
    port = os.getenv("PORT", 8000)
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
