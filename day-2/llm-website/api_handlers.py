from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
        
    