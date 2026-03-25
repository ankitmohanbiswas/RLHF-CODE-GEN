"""
FAST API deployment for the trained model
"""

from fastapi import FastAPI, HTTPException,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from executor import CodeExecutor, load_problems
import torch

app = FastAPI(
    title="RLHF Code-Generator API",
    description="AI-powered code generation trained with RLHF",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
print("Loading model...")
MODEL_PATH = "models/rlhf_trained"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✅ Trained model loaded successfully!")
except:
    print("⚠️ Trained model not found, using base model")
    MODEL_PATH = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✅ Base model loaded successfully!")

executor = CodeExecutor(timeout=5)
problems = load_problems()

# ============ FIXED REQUEST/RESPONSE MODELS ============

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    code: str
    prompt: str  # ← FIXED: Changed from problem_id to prompt

class EvaluateRequest(BaseModel):
    code: str
    problem_id: int

class EvaluateResponse(BaseModel):
    success: bool
    error: str | None = None
    reward: float

# ============ API ENDPOINTS ============

@app.get("/")
def root():
    """API root endpoint"""
    return {
        "message": "RLHF Code Generator API",
        "model": MODEL_PATH,
        "status": "running",
        "endpoints": {
            "GET /": "This help message",
            "GET /problems": "List all coding problems",
            "POST /generate": "Generate code from prompt",
            "POST /evaluate": "Test generated code",
            "GET /health": "Health check"
        }
    }

@app.get("/problems")
def get_problems():
    """Get all coding problems"""
    return {
        "count": len(problems),
        "problems": problems
    }

@app.post("/train")
def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training)
    return {"message": "PPO training started in background"}

def run_training():
    import subprocess
    subprocess.run(["python", "src/train_rlhf.py"])

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """
    Generate code completion from a prompt.
    
    Example request:
    {
        "prompt": "def add(a, b):",
        "max_tokens": 150,
        "temperature": 0.7
    }
    """
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        input_length = inputs['input_ids'].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only new tokens
        new_tokens = outputs[0][input_length:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Combine prompt + generation
        full_code = request.prompt + "\n" + generated
        
        # Clean: extract only the function
        if '\n\n' in full_code:
            full_code = full_code.split('\n\n')[0]
        
        return GenerateResponse(
            code=full_code,
            prompt=request.prompt  # ← FIXED: Return prompt, not problem_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest):
    """
    Evaluate generated code against test cases.
    
    Example request:
    {
        "code": "def add(a, b):\\n    return a + b",
        "problem_id": 1
    }
    """
    try:
        # Find the problem
        problem = next((p for p in problems if p["id"] == request.problem_id), None)
        
        if not problem:
            raise HTTPException(status_code=404, detail=f"Problem {request.problem_id} not found")
        
        # Execute and test
        success, error, reward = executor.execute(request.code, problem["test"])
        
        return EvaluateResponse(
            success=success,
            error=error,
            reward=reward
        )
    
    except StopIteration:
        raise HTTPException(status_code=404, detail="Problem not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "device": str(model.device) if model else None,
        "problems_loaded": len(problems)
    }

# ============ RUN SERVER ============

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 Starting RLHF Code Generator API")
    print("="*70)
    print(f"📍 Docs: http://localhost:8000/docs")
    print(f"📍 API:  http://localhost:8000")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)