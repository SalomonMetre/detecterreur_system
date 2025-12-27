import json
import asyncio
import aiohttp
import spacy
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from functools import lru_cache

# Import de votre librairie
from detecterreur.orchestrator import Orchestrator

app = FastAPI(title="DetectErreur API - Streaming Mode")

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
    GRAMMAR_CONTEXT = "Aucune règle spécifique disponible."

LLM_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral-nemo"

class TextRequest(BaseModel):
    text: str

# --- STREAMING HELPER ---
async def stream_ollama(prompt: str):
    """Générateur pour envoyer les tokens au fur et à mesure"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True, # Indispensable pour le streaming
        "options": {
            "temperature": 0.2,
            "num_ctx": 8192,
            "num_predict": 100, # Limité pour la vitesse
        }
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(LLM_URL, json=payload) as resp:
                async for line in resp.content:
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        yield token
                        if data.get("done"):
                            break
        except Exception as e:
            yield f" [Erreur de connexion: {str(e)}]"

# --- ENDPOINTS ---

@app.post("/api/advise")
async def advise_text(request: TextRequest):
    """
    Endpoint C: Retour pédagogique ultra-rapide en Streaming.
    """
    report = orchestrator.get_detailed_report(request.text)
    errors_set = {name for _, name, is_err in report["errors"] if is_err}
    
    if not errors_set:
        async def success_gen(): yield "Parfait ! Aucune erreur détectée."
        return StreamingResponse(success_gen(), media_type="text/plain")

    prompt = f"""
    [CONTEXTE]: {GRAMMAR_CONTEXT[:5000]}
    [INPUT]: "{request.text}"
    [CODES]: {", ".join(errors_set)}

    TÂCHE: Donne un conseil de prof très court (2 lignes max) + UN exemple. 
    Sois direct. Pas de politesses. Pas de codes techniques.
    """

    return StreamingResponse(stream_ollama(prompt), media_type="text/plain")

# --- GARDER LES AUTRES ENDPOINTS IDENTIQUES (OU ADAPTER) ---
@app.post("/api/analyze")
async def analyze_text(request: TextRequest):
    doc = nlp(request.text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    stats = {"FORME": 0, "ORTHOGRAPHE": 0, "GRAMMAIRE": 0, "SYNTAXE": 0, "PONCTUATION": 0}
    for sent in sentences:
        for cat, _, has_err in orchestrator.get_error(sent):
            if has_err and cat in stats: stats[cat] += 1
    total = max(1, len(sentences))
    return {"scores": {k: round(max(0, 10-(v/total*3)), 1) for k, v in stats.items()}, "raw_counts": stats}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    return {"text": content.decode("utf-8")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=32768)