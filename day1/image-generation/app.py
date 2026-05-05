from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import io
import os
from PIL import Image

load_dotenv()
client = InferenceClient(token=os.getenv("HF_API_KEY"))
prompt = "doreamon cindrella playing music with guitar"
# llm call
image  = client.text_to_image(
    prompt=prompt,
    model='stabilityai/stable-diffusion-xl-base-1.0'
)
output_path = "guitar.png"
image.save(output_path)
print(f"Image saved successfully ... : {output_path}")