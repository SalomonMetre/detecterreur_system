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

# Configuration CORS pour autoriser les requêtes venant de votre VPS (DuckDNS)
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
# AUCUNE LIMITE : On charge l'intégralité du fichier pour le RAG
GRAMMAR_CONTEXT = ""
try:
    with open("resources/french_grammar_rules_final.txt", "r", encoding="utf-8") as f:
        GRAMMAR_CONTEXT = f.read() 
except FileNotFoundError:
    print("[WARN] Fichier de règles introuvable. Le LLM n'aura pas de base de référence.")

# --- CONFIG OLLAMA ---
LLM_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral-nemo"

class TextRequest(BaseModel):
    text: str

# --- HELPERS ---
async def call_llm(prompt: str, temperature: float = 0.3) -> str:
    """Appel asynchrone vers Ollama avec fenêtre de contexte optimisée"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 8192,   # 8k est idéal pour 15k-30k caractères + prompt
                "num_predict": 300,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        try:
            async with session.post(LLM_URL, json=payload) as resp:
                if resp.status != 200:
                    return "Désolé, le service de conseil est temporairement indisponible."
                result = await resp.json()
                return result.get("response", "").strip()
        except Exception:
            return "Erreur de connexion au moteur d'I.A."

# --- ENDPOINTS ---

@app.post("/api/analyze")
async def analyze_text(request: TextRequest):
    doc = nlp(request.text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    stats = {"FORME": 0, "ORTHOGRAPHE": 0, "GRAMMAIRE": 0, "SYNTAXE": 0, "PONCTUATION": 0}
    for sent in sentences:
        errors = orchestrator.get_error(sent)
        for cat, _, has_err in errors:
            if has_err and cat in stats:
                stats[cat] += 1

    total_sentences = max(1, len(sentences))
    heatmap_scores = {cat: round(max(0, 10 - (count / total_sentences * 3)), 1) for cat, count in stats.items()}
    
    return {"scores": heatmap_scores, "raw_counts": stats}

@app.post("/api/correct")
async def correct_text(request: TextRequest):
    base_correction = orchestrator.correct(request.text)
    prompt = f"""[Instruction]: Corrige le texte suivant en français standard.
    - Base pré-corrigée : "{base_correction}"
    - Texte original : "{request.text}"
    - Renvoie UNIQUEMENT le texte final corrigé, sans commentaires ni guillemets."""
    
    final_correction = await call_llm(prompt, temperature=0.1)
    if not final_correction or len(final_correction) < 2:
        final_correction = base_correction
    
    return {"correction": final_correction.replace('"', '')}

@app.post("/api/advise")
async def advise_text(request: TextRequest):
    """
    Endpoint C: Conseil Pédagogique STRICTEMENT BASÉ SUR LES RÈGLES.
    """
    report = orchestrator.get_detailed_report(request.text)
    # On récupère les codes d'erreurs pour le contexte interne du LLM
    errors_set = {name for _, name, is_err in report["errors"] if is_err}
    
    if not errors_set:
        return {"advice": "Excellent travail ! Aucune erreur majeure détectée selon nos critères."}

    error_summary = ", ".join(errors_set)

    prompt = f"""
    ### DOCUMENT DE RÉFÉRENCE (SOURCE UNIQUE DE VÉRITÉ) ###
    {GRAMMAR_CONTEXT}
    #######################################################

    [Situation]
    L'étudiant a écrit : "{request.text}"
    Les codes d'erreurs internes détectés sont : {error_summary}

    [Tâche]
    Agis comme un professeur de français bienveillant. Donne un SEUL conseil pédagogique court (max 60 mots).
    
    [CONTRAINTES STRICTES]
    1. Ton conseil doit être BASÉ sur les règles présentes dans le DOCUMENT DE RÉFÉRENCE ci-dessus.
    2. Si le document ne mentionne pas la règle spécifique pour l'erreur, donne un conseil générique sur la langue française selon l'erreur commise. 
    3. INTERDICTION D'INVENTER des règles ou d'utiliser des connaissances externes non présentes dans le document.
    4. Cite implicitement la règle du document pour expliquer la faute et UTILISE obligatoirement UN EXEMPLE concret.
    5. NE MENTIONNE JAMAIS les codes techniques d'erreurs (ex: FAGL, GCON, ORTH) dans ta réponse. Parle de concepts grammaticaux clairs.
    """

    advice = await call_llm(prompt, temperature=0.3)
    return {"advice": advice}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    return {"text": content.decode("utf-8")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=32768)