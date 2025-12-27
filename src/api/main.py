from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import aiohttp
from typing import List, Dict, Any
import uvicorn
import re

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
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    from spacy.cli import download
    download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

orchestrator = Orchestrator()

# --- CHARGEMENT DU CONTEXTE GRAMMATICAL ---
# MODIFICATION: On charge TOUT le fichier. Mistral-Nemo a une grande fenêtre de contexte.
# C'est nécessaire pour que le conseil soit basé sur l'ensemble des règles.
GRAMMAR_CONTEXT = ""
try:
    with open("french_grammar_rules_final.txt", "r", encoding="utf-8") as f:
        GRAMMAR_CONTEXT = f.read() 
except FileNotFoundError:
    print("[WARN] french_grammar_rules_final.txt introuvable. Le contexte sera vide.")

# --- CONFIG ---
LLM_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral-nemo"

class TextRequest(BaseModel):
    text: str

# --- HELPERS ---
def split_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

async def call_llm(prompt: str, temperature: float = 0.3) -> str:
    """Appel asynchrone optimisé vers Ollama"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 8192,   # Augmenté pour contenir tout le fichier de règles
                "num_predict": 256
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

# --- ENDPOINTS ---

@app.post("/api/analyze")
async def analyze_text(request: TextRequest):
    """
    Endpoint A: Analyse statistique pure (très rapide).
    """
    sentences = split_sentences(request.text)
    if not sentences:
        return {"scores": {}, "raw_counts": {}}

    stats = {"FORME": 0, "ORTHOGRAPHE": 0, "GRAMMAIRE": 0, "SYNTAXE": 0, "PONCTUATION": 0}

    for sent in sentences:
        errors = orchestrator.get_error(sent)
        for cat, _, has_err in errors:
            if has_err and cat in stats:
                stats[cat] += 1

    heatmap_scores = {}
    total_sentences = len(sentences)
    
    for cat, count in stats.items():
        avg_err = count / total_sentences
        score = max(0, 10 - (avg_err * 3))
        heatmap_scores[cat] = round(score, 1)

    return {"scores": heatmap_scores, "raw_counts": stats}

@app.post("/api/correct")
async def correct_text(request: TextRequest):
    """
    Endpoint B: Correction hybride.
    """
    # 1. Correction mécanique
    base_correction = orchestrator.correct(request.text)

    # 2. Raffinement Contextuel (LLM)
    # On passe une partie du contexte juste pour aider au style, 
    # mais la priorité ici est la correction fluide.
    prompt = f"""
    [Instruction]: Corrige le texte suivant en français standard.
    - Utilise cette base pré-corrigée : "{base_correction}"
    - Texte original : "{request.text}"
    - Renvoie UNIQUEMENT le texte final corrigé.
    """
    
    final_correction = await call_llm(prompt, temperature=0.1)
    
    if not final_correction or len(final_correction) < 2:
        final_correction = base_correction

    final_correction = final_correction.replace('"', '').strip()

    return {"correction": final_correction}

@app.post("/api/advise")
async def advise_text(request: TextRequest):
    """
    Endpoint C: Conseil Pédagogique STRICTEMENT BASÉ SUR LES RÈGLES.
    """
    report = orchestrator.get_detailed_report(request.text)
    errors_set = {name for _, name, is_err in report["errors"] if is_err}
    
    if not errors_set:
        return {"advice": "Excellent travail ! Aucune erreur majeure détectée selon nos critères."}

    error_summary = ", ".join(errors_set)

    # PROMPT STRICT RAG (Retrieval Augmented Generation)
    prompt = f"""
    ### DOCUMENT DE RÉFÉRENCE (SOURCE UNIQUE DE VÉRITÉ) ###
    {GRAMMAR_CONTEXT}
    #######################################################

    [Situation]
    L'étudiant a écrit : "{request.text}"
    Les erreurs détectées sont : {error_summary}

    [Tâche]
    Agis comme un professeur de français. Donne un conseil pédagogique court (max 50 mots).
    
    [CONTRAINTES STRICTES]
    1. Ton conseil doit être BASÉ sur les règles présentes dans le DOCUMENT DE RÉFÉRENCE ci-dessus.
    2. Si le document ne mentionne pas la règle spécifique pour l'erreur, donne un conseil générique sur la langue française selon l'erreur commise. 
    3. INTERDICTION D'INVENTER des règles ou d'utiliser des connaissances externes non présentes dans le document.
    4. Cite implicitement la règle du document pour expliquer la faute et sers-toi d'un EXEMPLE.
    5. Ne mentionne pas les noms des erreurs donnés ci-dessus dans la liste des erreurs détectées.
    """

    advice = await call_llm(prompt, temperature=0.2) # Température basse pour réduire l'hallucination
    return {"advice": advice}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        return {"text": content.decode("utf-8")}
    except Exception:
        raise HTTPException(status_code=400, detail="Erreur lecture fichier")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=32768)