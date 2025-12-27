import json
import asyncio
import aiohttp
import spacy
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator

# Import de votre librairie
from detecterreur.orchestrator import Orchestrator

app = FastAPI(title="DetectErreur API - Précision Pédagogique")

# Configuration CORS
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
    """Appel atomique pour la correction textuelle"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": MODEL_NAME, 
            "prompt": prompt, 
            "stream": False,
            "options": {
                "temperature": temperature, 
                "num_ctx": 8192, 
                "num_predict": 256
            }
        }
        try:
            async with session.post(LLM_URL, json=payload) as resp:
                if resp.status != 200: return ""
                result = await resp.json()
                return result.get("response", "").strip()
        except Exception: return ""

async def stream_ollama(prompt: str) -> AsyncGenerator[str, None]:
    """Générateur robuste pour le streaming JSON d'Ollama"""
    payload = {
        "model": MODEL_NAME, 
        "prompt": prompt, 
        "stream": True,
        "options": {
            "temperature": 0.2, 
            "num_ctx": 8192, 
            "num_predict": 256
        }
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_URL, json=payload) as resp:
            # On itère sur les lignes pour éviter de briser les objets JSON
            async for line in resp.content:
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        yield token
                        if data.get("done"): break
                    except json.JSONDecodeError:
                        continue

# --- ENDPOINTS ---

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    """Analyse radar basée sur Spacy et l'Orchestrateur"""
    doc = nlp(request.text)
    stats = {"FORME": 0, "ORTHOGRAPHE": 0, "GRAMMAIRE": 0, "SYNTAXE": 0, "PONCTUATION": 0}
    
    sentence_count = 0
    for sent in doc.sents:
        sentence_count += 1
        # On délègue la détection brute à l'orchestrateur
        for cat, _, has_err in orchestrator.get_error(sent.text):
            if has_err and cat in stats:
                stats[cat] += 1
                
    total = max(1, sentence_count)
    # Calcul de la "santé" du texte (Base 10)
    scores = {k: round(max(0, 10 - (v / total * 3)), 1) for k, v in stats.items()}
    return {"scores": scores, "raw_counts": stats}

@app.post("/correct")
async def correct_text(request: TextRequest):
    """Correction recyclée : Orchestrateur (Règles) + LLM (Fluidité)"""
    report = orchestrator.get_detailed_report(request.text)
    suggestions = report.get("suggestions", [])
    
    # Extraction des modifications indépendantes pour le contexte LLM
    diff_context = "\n".join([f"- Modification : '{request.text}' -> '{sug[3]}'" 
                             for sug in suggestions if sug[2]])
    
    final_cascade = report.get("corrected")

    prompt = f"""[Instruction]: Tu es un correcteur expert en langue française.
    L'analyse locale a suggéré ces modifications atomiques :
    {diff_context}

    [Base de correction]: "{final_cascade}"
    [Texte original]: "{request.text}"

    TÂCHE: Produis une version finale fluide et correcte. 
    Respecte strictement les règles de grammaire française.
    Renvoie UNIQUEMENT le texte final, sans commentaires ni guillemets."""
    
    final_correction = await call_llm(prompt, temperature=0.1)
    return {"correction": final_correction if final_correction else final_cascade}

@app.post("/advise")
async def advise_text(request: TextRequest):
    """Conseil pédagogique en streaming (Masquage des codes techniques)"""
    report = orchestrator.get_detailed_report(request.text)
    categories_set = {cat for cat, _, is_err in report["errors"] if is_err}
    
    if not categories_set:
        async def success(): yield "Excellent travail ! Aucune erreur détectée."
        return StreamingResponse(success(), media_type="text/plain")

    prompt = f"""
    ### RÈGLES DE RÉFÉRENCE ###
    {GRAMMAR_CONTEXT[:5000]}
    ###########################
    [TEXTE ÉTUDIANT]: "{request.text}"
    [DOMAINES D'ERREURS]: {", ".join(categories_set)}

    TÂCHE: Agis en professeur de français bienveillant. Explique la règle.
    
    RÈGLES STRICTES :
    1. NE MENTIONNE JAMAIS de codes techniques (ex: GCON, FMAJ, ORTH).
    2. Parle uniquement des concepts : {", ".join(categories_set)}.
    3. Explique la faute simplement et donne obligatoirement UN exemple correct.
    4. Réponse courte (max 60 mots).
    """
    return StreamingResponse(stream_ollama(prompt), media_type="text/plain")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Gestion de l'import de fichiers texte"""
    try:
        content = await file.read()
        return {"text": content.decode("utf-8")}
    except Exception:
        raise HTTPException(status_code=400, detail="Fichier invalide.")

if __name__ == "__main__":
    # Lancement sur le port configuré pour le tunnel Nginx
    uvicorn.run(app, host="0.0.0.0", port=32768)