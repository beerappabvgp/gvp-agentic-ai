from dotenv import load_dotenv
from groq import Groq
import os, io
from huggingface_hub import InferenceClient
from PIL import Image
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
hf_client = InferenceClient(os.getenv("HF_API_KEY"))
def groq_stream(system_msg, user_msg, model_name, temperature):
    messages = [
        {
            'role': "system",
            'content': system_msg
        }, 
        {
            'role': 'user',
            'content': user_msg
        }
    ]
    stream = groq_client.chat.completions.create(
        model = model_name,
        messages = messages,
        temperature = 0.5,
        stream = True  
    )
    
    # iterate through the list of tokens in the stream 
    
    for chunk in stream:
        yield chunk.choices[0].delta.content or ''
        
def generate_image(prompt):
    image = hf_client.text_to_image(
        model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt=prompt
    )

    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))

    elif isinstance(image, Image.Image):
        pass  # already good

    else:
        raise ValueError(f"Unexpected output type: {type(image)}")

    return image