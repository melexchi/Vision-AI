"""
SmolVLM Server - Vision Language Model
Port: 8002
HuggingFace SmolVLM for image understanding
"""
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import base64
from io import BytesIO

app = FastAPI(title="SmolVLM Service", version="1.0.0")

# Lazy load model
vlm_model = None
vlm_processor = None

class ImageQueryRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    prompt: str = "Describe this image in detail."

def get_model():
    global vlm_model, vlm_processor
    if vlm_model is None:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

        model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

        vlm_processor = AutoProcessor.from_pretrained(model_id)
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="cuda:0"
        )
        vlm_model.eval()
        
    return vlm_model, vlm_processor

def process_image(image_data: bytes):
    """Process image bytes to PIL Image"""
    from PIL import Image
    from io import BytesIO
    return Image.open(BytesIO(image_data)).convert("RGB")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "smolvlm"}

@app.post("/v1/query")
async def query_image(request: ImageQueryRequest):
    """Query image with text prompt"""
    try:
        from PIL import Image
        import httpx
        import torch
        
        model, processor = get_model()
        
        # Get image
        if request.image_base64:
            image_data = base64.b64decode(request.image_base64)
            image = process_image(image_data)
        elif request.image_url:
            async with httpx.AsyncClient() as client:
                resp = await client.get(request.image_url)
                image = process_image(resp.content)
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Build conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": request.prompt}
                ]
            }
        ]
        
        # Process inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
        
        # Decode
        generated_text = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return JSONResponse({
            "success": True,
            "response": generated_text.strip(),
            "prompt": request.prompt
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/v1/query/upload")
async def query_image_upload(
    image: UploadFile = File(...),
    prompt: str = Form(default="Describe this image in detail.")
):
    """Query uploaded image with text prompt"""
    try:
        import torch
        
        model, processor = get_model()
        
        # Read image
        image_data = await image.read()
        pil_image = process_image(image_data)
        
        # Build conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt_text, images=[pil_image], return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
        
        # Decode
        generated_text = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return JSONResponse({
            "success": True,
            "response": generated_text.strip(),
            "prompt": prompt
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/v1/unload")
async def unload_model():
    """Unload model to free VRAM"""
    global vlm_model, vlm_processor
    if vlm_model is not None:
        del vlm_model
        del vlm_processor
        vlm_model = None
        vlm_processor = None
        import torch
        torch.cuda.empty_cache()
    return {"status": "unloaded"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8282)
