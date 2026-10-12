# Poubelle intelligente (VLM NVIDIA)

## Installation
```bash
pip install -r requirements.txt
cp .env.example .env      # puis colle ta clé nvapi-...
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Ouvre http://localhost:8000

## Tester depuis un téléphone
La caméra du navigateur exige HTTPS (sauf sur localhost). Deux options :
- bouton « Importer une photo » : marche en HTTP et ouvre directement l'appareil photo du téléphone ;
- tunnel HTTPS : `cloudflared tunnel --url http://localhost:8000` ou `ngrok http 8000`.

## Changer de modèle
Modifie `NVIDIA_MODEL` dans `.env` (voir la section Vision de build.nvidia.com).
