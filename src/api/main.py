import json
import asyncio
import aiohttp
import spacy
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import de votre librairie
from detecterreur.orchestrator import Orchestrator

app = FastAPI(title="DetectErreur API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RESSOURCES GLOBALES ---
nlp = spacy.load("fr_core_news_sm")
orchestrator = Orchestrator()

try:
    with open("resources/french_grammar_rules_final.txt", "r", encoding="utf-8") as f:
        GRAMMAR_CONTEXT = f.read()
except FileNotFoundError:
    GRAMMAR_CONTEXT = ""

LLM_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral-nemo"

class TextRequest(BaseModel):
    text: str

# --- HELPERS ---
async def call_llm(prompt: str, temperature: float = 0.1) -> str:
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": 200}
        }
        try:
            async with session.post(LLM_URL, json=payload) as resp:
                if resp.status != 200: return ""
                result = await resp.json()
                return result.get("response", "").strip()
        except: return ""

async def stream_ollama(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 100}
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_URL, json=payload) as resp:
            async for line in resp.content:
                if line:
                    data = json.loads(line.decode("utf-8"))
                    yield data.get("response", "")
                    if data.get("done"): break

# --- ENDPOINTS (ROUTES CORRIGÉES) ---

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    sentences = [sent.text.strip() for sent in nlp(request.text).sents if sent.text.strip()]
    stats = {"FORME": 0, "ORTHOGRAPHE": 0, "GRAMMAIRE": 0, "SYNTAXE": 0, "PONCTUATION": 0}
    for sent in sentences:
        for cat, _, has_err in orchestrator.get_error(sent):
            if has_err and cat in stats: stats[cat] += 1
    total = max(1, len(sentences))
    scores = {k: round(max(0, 10-(v/total*3)), 1) for k, v in stats.items()}
    return {"scores": scores, "raw_counts": stats}

@app.post("/correct")
async def correct_text(request: TextRequest):
    base = orchestrator.correct(request.text)
    prompt = f'Corrige ce texte sans bla-bla, renvoie juste le texte: "{base}"'
    final = await call_llm(prompt)
    return {"correction": final if final else base}

@app.post("/advise")
async def advise_text(request: TextRequest):
    report = orchestrator.get_detailed_report(request.text)
    errs = {name for _, name, is_err in report["errors"] if is_err}
    if not errs:
        async def ok(): yield "Aucune erreur détectée !"
        return StreamingResponse(ok(), media_type="text/plain")
    
    prompt = f"[CONTEXTE]: {GRAMMAR_CONTEXT[:4000]}\n[FAUTE]: {request.text}\nConseil court + exemple (max 50 mots):"
    return StreamingResponse(stream_ollama(prompt), media_type="text/plain")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    return {"text": content.decode("utf-8")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=32768)