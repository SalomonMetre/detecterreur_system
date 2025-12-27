import json
import asyncio
import aiohttp
import spacy
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Set, AsyncGenerator, Optional

from detecterreur.orchestrator import Orchestrator

app = FastAPI(
    title="DetectErreur API - Correction Contextuelle",
    description="API pour la correction avancée de textes en français, utilisant un orchestrateur et un LLM avec contexte grammatical et rapport détaillé.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Ressources globales ---
nlp = spacy.load("fr_core_news_sm")
orchestrator = Orchestrator()

try:
    with open("resources/french_grammar_rules_final.txt", "r", encoding="utf-8") as f:
        GRAMMAR_CONTEXT: str = f.read()
except FileNotFoundError:
    GRAMMAR_CONTEXT: str = ""
    print("Avertissement : Le fichier de règles de grammaire est introuvable.")

LLM_URL: str = "http://localhost:11434/api/generate"
MODEL_NAME: str = "mistral-nemo"

# --- Modèles Pydantic ---
class TextRequest(BaseModel):
    text: str

class AnalysisResult(BaseModel):
    scores: Dict[str, float]
    raw_counts: Dict[str, int]

class CorrectionResult(BaseModel):
    correction: str

# --- Fonctions utilitaires ---
async def call_llm(prompt: str, temperature: float = 0.1) -> str:
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
                if resp.status != 200:
                    print(f"Erreur LLM: Statut HTTP {resp.status}")
                    return ""
                result = await resp.json()
                return result.get("response", "").strip()
        except Exception as e:
            print(f"Erreur lors de l'appel au LLM: {e}")
            return ""

async def stream_ollama(prompt: str) -> AsyncGenerator[str, None]:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.2,
            "num_ctx": 8192,
            "num_predict": 128
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
            yield f" [Erreur de connexion au LLM: {str(e)}]"

# --- Endpoints ---
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_text(request: TextRequest) -> JSONResponse:
    doc = nlp(request.text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    stats: Dict[str, int] = {
        "FORME": 0,
        "ORTHOGRAPHE": 0,
        "GRAMMAIRE": 0,
        "SYNTAXE": 0,
        "PONCTUATION": 0
    }

    for sent in sentences:
        for cat, _, has_err in orchestrator.get_error(sent):
            if has_err and cat in stats:
                stats[cat] += 1

    total = max(1, len(sentences))
    scores = {k: round(max(0, 10 - (v / total * 3)), 1) for k, v in stats.items()}
    return JSONResponse(content={"scores": scores, "raw_counts": stats})

@app.post("/correct", response_model=CorrectionResult)
async def correct_text(request: TextRequest) -> JSONResponse:
    """
    Corrige un texte en utilisant :
    - La correction de base de l'orchestrateur
    - Le rapport détaillé des erreurs
    - Le contexte grammatical
    Le LLM doit améliorer la fluidité et la justesse, en tenant compte de TOUTES ces informations.
    """
    base_correction = orchestrator.correct(request.text)
    report = orchestrator.get_detailed_report(request.text)
    errors_set = {name for _, name, is_err in report["errors"] if is_err}

    prompt = f"""
    ### Contexte ###
    **Règles de grammaire française (extrait)** :
    {GRAMMAR_CONTEXT[:3000]}

    **Rapport détaillé des erreurs** :
    {json.dumps(report, ensure_ascii=False)[:2000]}

    **Types d'erreurs détectées** :
    {", ".join(errors_set) if errors_set else "Aucune"}

    ### Texte à corriger ###
    **Original** : "{request.text}"
    **Correction de base** : "{base_correction}"

    ### Instruction ###
    1. Agis comme un correcteur professionnel.
    2. Utilise la **correction de base** comme point de départ.
    3. Prends en compte le **rapport détaillé** et les **règles de grammaire** pour améliorer la justesse et la fluidité.
    4. Corrige TOUTES les erreurs identifiées dans le rapport, même si elles ne sont pas dans la correction de base.
    5. Renvoie UNIQUEMENT la version finale corrigée, sans guillemets ni explications.
    """

    final_correction = await call_llm(prompt, temperature=0.1)

    if not final_correction or len(final_correction.strip()) < 2:
        final_correction = base_correction

    return JSONResponse(content={"correction": final_correction.replace('"', '').strip()})

@app.post("/advise")
async def advise_text(request: TextRequest) -> StreamingResponse:
    report = orchestrator.get_detailed_report(request.text)
    errors_set: Set[str] = {name for _, name, is_err in report["errors"] if is_err}

    if not errors_set:
        async def success_gen() -> AsyncGenerator[str, None]:
            yield "Excellent travail ! Aucune erreur détectée."
        return StreamingResponse(success_gen(), media_type="text/plain")

    prompt = f"""
    ### Contexte ###
    **Règles de grammaire française** :
    {GRAMMAR_CONTEXT[:3000]}

    **Rapport détaillé** :
    {json.dumps(report, ensure_ascii=False)[:2000]}

    **Types d'erreurs** :
    {", ".join(errors_set)}

    ### Texte ###
    "{request.text}"

    ### Instruction ###
    Agis en professeur de français.
    Donne un conseil **très court** (max 50 mots) + **UN exemple concret** pour corriger les erreurs identifiées.
    Sois direct, clair et pédagogique. Évite les codes techniques.
    """

    return StreamingResponse(stream_ollama(prompt), media_type="text/plain")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    try:
        content = await file.read()
        return JSONResponse(content={"text": content.decode("utf-8")})
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur de lecture du fichier: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=32768,
        log_level="info"
    )
