# app.py

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PIL import Image
import io

app = FastAPI()

# Load the Mistral model (replace with your local model path if needed)
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.post("/process")
async def process_image_and_text(
    image: UploadFile = File(...),
    modification_prompt: str = Form(...)
):
    # Load image (for now just check it uploads)
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))

    # Format prompt for LLM
    full_prompt = (
        "Instruction: Convert the following user request into a short descriptive caption suitable for CLIP.\n"
        f"Input: {modification_prompt}\nOutput:"
    )

    # Run through Mistral
    llm_output = llm_pipeline(full_prompt, max_new_tokens=50, do_sample=True)[0]['generated_text']
    
    # Extract output (everything after 'Output:')
    caption = llm_output.split("Output:")[-1].strip()

    return JSONResponse({"caption": caption})