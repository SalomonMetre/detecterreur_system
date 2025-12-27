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

app = FastAPI(title="DetectErreur API - Hybrid & Streaming")

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
    """Appel standard pour la correction (non-streaming)"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 8192,
                "num_predict": 256  # Multiple de 2
            }
        }
        try:
            async with session.post(LLM_URL, json=payload) as resp:
                if resp.status != 200:
                    return ""
                result = await resp.json()
                return result.get("response", "").strip()
        except Exception:
            return ""

async def stream_ollama(prompt: str):
    """Générateur pour le conseil pédagogique (streaming)"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.2,
            "num_ctx": 8192,
            "num_predict": 128  # Multiple de 2 (plus court pour la vitesse)
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
            yield f" [Erreur de connexion LLM]"

# --- ENDPOINTS ---

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    doc = nlp(request.text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    stats = {"FORME": 0, "ORTHOGRAPHE": 0, "GRAMMAIRE": 0, "SYNTAXE": 0, "PONCTUATION": 0}
    
    for sent in sentences:
        for cat, _, has_err in orchestrator.get_error(sent):
            if has_err and cat in stats:
                stats[cat] += 1
                
    total = max(1, len(sentences))
    scores = {k: round(max(0, 10-(v/total*3)), 1) for k, v in stats.items()}
    return {"scores": scores, "raw_counts": stats}

@app.post("/correct")
async def correct_text(request: TextRequest):
    """Correction hybride : Orchestrateur recyclé par LLM"""
    base_correction = orchestrator.correct(request.text)
    
    prompt = f"""[Instruction]: Agis comme un correcteur professionnel. 
    Améliore la fluidité de cette phrase tout en conservant les corrections déjà apportées.
    - Phrase originale : "{request.text}"
    - Base corrigée : "{base_correction}"
    
    Renvoie UNIQUEMENT la phrase corrigée finale, sans guillemets ni explications."""

    final_correction = await call_llm(prompt, temperature=0.1)
    
    # Fallback de sécurité
    if not final_correction or len(final_correction.strip()) < 2:
        final_correction = base_correction

    return {"correction": final_correction.replace('"', '').strip()}

@app.post("/advise")
async def advise_text(request: TextRequest):
    """Conseil pédagogique en streaming"""
    report = orchestrator.get_detailed_report(request.text)
    errors_set = {name for _, name, is_err in report["errors"] if is_err}
    
    if not errors_set:
        async def success_gen(): yield "Excellent travail ! Aucune erreur détectée."
        return StreamingResponse(success_gen(), media_type="text/plain")

    prompt = f"""
    ### DOCUMENT DE RÉFÉRENCE ###
    {GRAMMAR_CONTEXT[:5000]}
    ##############################
    [INPUT]: "{request.text}"
    [CODES]: {", ".join(errors_set)}

    TÂCHE: Agis en prof de français. Donne un conseil très court (max 50 mots) + UN exemple. 
    Sois direct. Pas de politesses. Pas de codes techniques.
    """
    return StreamingResponse(stream_ollama(prompt), media_type="text/plain")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        return {"text": content.decode("utf-8")}
    except:
        raise HTTPException(status_code=400, detail="Erreur de lecture")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=32768)