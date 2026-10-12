"""Poubelle intelligente — backend FastAPI qui interroge un VLM NVIDIA (build.nvidia.com)."""
import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()  # en local : lit .env ; sur Vercel : variables définies dans le dashboard

API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
BASE = Path(__file__).parent

CATEGORIES = ["papier_carton", "plastique", "verre", "metal", "organique", "electronique", "autre"]

PROMPT = f"""Tu es le système de tri d'une poubelle intelligente.
Regarde l'objet principal sur la photo et classe-le dans UNE seule catégorie parmi :
{", ".join(CATEGORIES)}.
- papier_carton : journaux, feuilles, boîtes en carton
- plastique : bouteilles, emballages, sacs plastiques
- verre : bouteilles et pots en verre
- metal : canettes, boîtes de conserve, aluminium
- organique : restes alimentaires, épluchures
- electronique : piles, câbles, petits appareils
- autre : tout le reste ou objet non identifiable

Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour, au format :
{{"categorie": "<une des catégories>", "objet": "<nom court de l'objet>", "confiance": <nombre entre 0 et 1>, "explication": "<une phrase>"}}"""

app = FastAPI(title="Poubelle intelligente")


class ImageIn(BaseModel):
    image: str  # data URL : "data:image/jpeg;base64,...."


def extraire_json(texte: str) -> dict:
    texte = re.sub(r"<think>.*?</think>", "", texte, flags=re.S)  # modèles « raisonneurs »
    texte = texte.replace("```json", "").replace("```", "")
    m = re.search(r"\{.*\}", texte, flags=re.S)
    if not m:
        raise ValueError(f"Pas de JSON dans la réponse : {texte[:200]}")
    data = json.loads(m.group(0))
    if data.get("categorie") not in CATEGORIES:
        data["categorie"] = "autre"
    return data


async def appeler_vlm(client: httpx.AsyncClient, messages: list) -> httpx.Response:
    return await client.post(
        URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"},
        json={"model": MODEL, "messages": messages, "max_tokens": 300, "temperature": 0.2},
    )


@app.post("/api/classify")
async def classify(payload: ImageIn):
    if not API_KEY:
        raise HTTPException(500, "NVIDIA_API_KEY manquante (fichier .env en local, variables d'environnement sur Vercel)")
    if not payload.image.startswith("data:image/"):
        raise HTTPException(400, "Image invalide")

    # Format OpenAI (accepté par la plupart des VLM NVIDIA)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": payload.image}},
        ],
    }]
    async with httpx.AsyncClient(timeout=60) as client:
        r = await appeler_vlm(client, messages)
        if r.status_code == 400:
            # Repli : certains modèles attendent l'image en balise <img> dans le texte
            messages = [{"role": "user", "content": f'{PROMPT}\n<img src="{payload.image}" />'}]
            r = await appeler_vlm(client, messages)

    if r.status_code != 200:
        raise HTTPException(502, f"Erreur NVIDIA {r.status_code} : {r.text[:300]}")

    texte = r.json()["choices"][0]["message"]["content"]
    try:
        return {**extraire_json(texte), "modele": MODEL}
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(502, str(e))


@app.get("/")
def index():
    # En local uniquement : sur Vercel, public/index.html est servi directement par le CDN
    page = BASE / "public" / "index.html"
    if not page.exists():
        raise HTTPException(404, "public/index.html introuvable")
    return FileResponse(page)