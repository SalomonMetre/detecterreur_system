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
    """
    Correction collaborative : Le LLM arbitre entre les corrections par couches.
    """
    # 1. Obtenir les traces indépendantes (original -> corrigé par couche)
    traces = orchestrator.get_independent_trace(request.text)
    
    # 2. Construction d'un contexte de modifications clair pour le LLM
    if traces:
        diff_context = "\n".join([f"- Modification suggérée : '{orig}' -> '{corr}'" for orig, corr in traces])
    else:
        diff_context = "Aucune erreur détectée par l'analyseur local."

    # 3. Récupération de la cascade complète (la meilleure version de l'automate)
    final_cascade = orchestrator.correct(request.text)

    prompt = f"""[Instruction]: Expert en linguistique française.
    
    [ANALYSE LOCALE DES COUCHES]:
    {diff_context}

    [SYNTHÈSE AUTOMATIQUE]: "{final_cascade}"
    [TEXTE SOURCE]: "{request.text}"

    TÂCHE:
    Vérifie la validité des modifications suggérées ci-dessus. 
    Produis une correction finale qui fusionne ces suggestions tout en assurant une fluidité parfaite.
    Si l'analyseur local a fait une erreur (sur-correction), privilégie le sens du texte source.

    RENVOIE UNIQUEMENT LE TEXTE FINAL."""
    
    final_correction = await call_llm(prompt, temperature=0.1)
    
    # Fallback sur la cascade si le LLM échoue
    return {"correction": final_correction if final_correction else final_cascade}

@app.post("/advise")
async def advise_text(request: TextRequest):
    """Conseil pédagogique basé sur les traces de l'orchestrateur."""
    report = orchestrator.get_detailed_report(request.text)
    traces = orchestrator.get_independent_trace(request.text)
    
    if not traces:
        async def success(): yield "Votre texte est déjà correct. Bravo !"
        return StreamingResponse(success(), media_type="text/plain")

    # On transforme les traces en explications visuelles pour le prompt
    explications_locales = "\n".join([f"Erreur détectée: '{orig}' corrigé en '{corr}'" for orig, corr in traces])

    prompt = f"""
    ### CONTEXTE GRAMMATICAL ###
    {GRAMMAR_CONTEXT[:3000]}
    ###########################

    [MODIFICATIONS EFFECTUÉES]:
    {explications_locales}

    TÂCHE: Agis en professeur bienveillant. 
    1. Explique la règle derrière l'une des corrections majeures.
    2. Ne cite aucun code technique (ex: OSUB, OORD).
    3. Donne un exemple concret de la règle appliquée.
    
    Contrainte: Style direct, max 60 mots.
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