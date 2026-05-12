from fastapi import FastAPI
from pydantic import BaseModel
from conversation import conversational_rag

app = FastAPI(
    title='Bank home loan assistant',
    description='Ask questions about bank loan policies.',
    version='1.0.0',
)

class ChatRequest(BaseModel):
    question: str

@app.get('/health')
def health():
    return {'status': 'server is up and running'}

@app.post('/chat')
def chat(body: ChatRequest):
    question = body.question.strip()

    if not question:
        return {'error': 'Question cannot be empty'}

    answer = conversational_rag(question)
    return {
        'question': question,
        'answer': answer,
    }