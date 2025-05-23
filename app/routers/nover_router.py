import os
import json
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

import google.generativeai as genai
from google.api_core import client_options
from google.cloud import visionai

from fastapi import APIRouter
from pydantic import BaseModel

from app.prompts import create_novel_ua, create_novel_eng, get_image_promt


class Response(BaseModel):
    text: str
    emotion: str
    question: str
    answers: list[str]
    explanation: str
    illustration: str


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


client = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

router = APIRouter()

@router.get("/create_novel")
async def create_novel(lang: str = "Eng"):

    prompt = create_novel_eng if lang == "Eng" else create_novel_ua
    
    try:
        response = client.generate_content(prompt)
        
        try:
            response_data = json.loads(response.text)
        except json.JSONDecodeError:
            response_data = {
                "text": response.text,
                "emotion": "",
                "question": "",
                "answers": [],
                "explanation": "",
                "illustration": "A scene from the story: " + response.text[:100]
            }
        
        response_data["image"] = client.generate_image(
            prompt=get_image_promt(response_data['illustration']),
            size="1024x1024",
            quality="standard"
        )
        return response_data
    
    except Exception as e:
        return {"error": str(e)}