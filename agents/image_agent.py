import os
import requests
import streamlit as st
import io
from PIL import Image

# Using a high-quality, fast model for SDXL via the new HF Router
API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

def generate_image(prompt: str):
    """Generates an image from a text prompt using the Hugging Face API."""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY environment variable is missing.")

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt}

    with st.spinner("Wolf is painting your vision..."):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            # Check if the request was successful
            if response.status_code == 200:
                image_bytes = response.content
                image = Image.open(io.BytesIO(image_bytes))
                return image
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                raise Exception(error_msg)
                
        except Exception as e:
            raise Exception(f"Failed to generate image: {str(e)}")
