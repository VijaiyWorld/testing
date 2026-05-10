from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a Personal Computer and Laptop service expert.

Your job:
- Help users fix PC/Laptop issues
- Give step-by-step troubleshooting
- Be simple and friendly
- Only answer questions related to:
  - PC
  - Laptop
  - Hardware
  - Software
  - Networking
  - Windows/Linux/Mac basic troubleshooting

If user asks outside topic, reply exactly:
"I am not well about the topic. Please ask related to PC/Laptop issues or technical help."
"""

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(req: ChatRequest):

    a = ask_ai(req.message)
    return {"reply": a}

def ask_ai(user_input):
    answer = []
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True
    )

    for chunk in completion:
        answer.append(chunk.choices[0].delta.content or "")

    return ''.join(answer)