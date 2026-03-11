import base64
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import io
from PIL import Image

def perform_ocr(image_bytes: bytes) -> str:
    """Uses Groq Vision to perform OCR and describe the image."""
    # Use Llama 4 Scout for high quality vision tasks (latest supported model)
    vision_llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1
    )
    
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # LangChain structure for multimodal inputs
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract all text from this image. If there is no text, describe what is in the image in detail."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    )
    
    response = vision_llm.invoke([message])
    return response.content
