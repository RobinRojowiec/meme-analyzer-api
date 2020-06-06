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
    image_url: bytes = None
    text: str = ""
    comments: List[str] = []
    min_confidence: float = 0.0
    max_tags: int = 10

class AnalysisResponse(BaseModel):
    tags: List[str] = []
    image_vector: List[float] = []
    text_vector: List[float] = []
