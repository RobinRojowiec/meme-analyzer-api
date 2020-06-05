"""

IDE: PyCharm
Project: meme-analyzer-api
Author: Robin
Filename: data_model.py
Date: 05.06.2020

"""
from typing import List

from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    image_file: bytes = None
    image_name: str = ""
    meme_text: str = ""
    comments: List[str] = []


class AnalysisResponse(BaseModel):
    tags: List[str] = []
    vector: List[float] = []
    min_confidence: float = 0.0
