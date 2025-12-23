from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app import app as langgraph_app
import uvicorn

# FastAPI app
app = FastAPI(title="Confer & MoXi Chatbot API")

# CORS middleware - Allow all origins in production (Coolify handles security)
# For stricter security, replace ["*"] with your specific frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    classification: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Confer & MoXi Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    result = langgraph_app.invoke({"question": request.message})
    
    return ChatResponse(
        response=result["generation"],
        classification=result.get("classification", "general")
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting Confer & MoXi Chatbot API server...")
    print("API will be available at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
