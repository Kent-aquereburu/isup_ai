# Atelier ISUP — Pipeline d’analyse documentaire avec IA

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![RAG](https://img.shields.io/badge/AI-Retrieval%20Augmented%20Generation-purple)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-green)
![Ollama](https://img.shields.io/badge/Local%20LLM-Ollama-orange)
![Docling](https://img.shields.io/badge/PDF-Docling-red)
![License](https://img.shields.io/badge/Use-Education-lightgrey)

## Notebook — Docling + Audio + RAG (FR)  

Ce dépôt contient un notebook pédagogique utilisé lors d’une intervention à l’ISUP dans le cadre d’une formation de niveau Master.

L’atelier est animé par :

**Kent A.** et **Yolan H.**

L’objectif est de montrer **de manière concrète et progressive** comment construire un pipeline moderne de **Retrieval-Augmented Generation (RAG)** appliqué à de données non structurées (documents, texte ou audios)

Le notebook permet d’explorer un pipeline complet capable de traiter :

- des **documents PDF** (analyse structurée avec Docling)
- des **contenus audio** (transcription automatique)
- une **recherche sémantique basée sur des embeddings**
- un **LLM** capable de répondre à des questions à partir des extraits les plus pertinents.

---

## Contexte pédagogique

Les systèmes RAG sont aujourd’hui au cœur de nombreux systèmes d’IA appliqués :

- assistants documentaires
- moteurs de recherche sémantiques
- analyse de contrats et de rapports
- support client automatisé
- analyse de connaissances d’entreprise

Le principe du RAG est simple :

1. Transformer les documents en **représentations vectorielles (embeddings)**  
2. Utiliser ces vecteurs pour **retrouver les passages pertinents**
3. Fournir ces passages à un **modèle de langage (LLM)** pour générer une réponse fiable.

Cette approche permet de **réduire les hallucinations** et d’améliorer la **traçabilité des réponses**.

---

## ⚙️ Configuration

L’application peut fonctionner avec deux backends pour les modèles de langage et les embeddings :

- **OpenAI (cloud)** : utilisation des modèles via API  
- **Ollama (local)** : exécution de modèles LLM et embeddings localement

Le backend est sélectionné directement dans la **sidebar de l’application Streamlit**.

---

### Configuration des LLM 

#### ☁️ Configuration OpenAI

Il faut une clé API OpenAI est nécessaire. Elle peut être créée sur : [OpenAI api keys](https://platform.openai.com/api-keys)

La clé doit être ajoutée dans le fichier : ``.streamlit/secrets.toml``

Ce fichier ne doit pas être versionné dans le dépôt Git.
Il est ajouté au .gitignore par défaut, ne pas changer cette configuration!

#### 🖥 Configuration Ollama (local)

Ollama permet d’exécuter des modèles de langage localement.

Documentation officielle :

https://ollama.com

Ollama peut être installé sur macOS avec :

brew install ollama

ou aller sur 

https://ollama.com/download

Le serveur Ollama doit être lancé ensuite dans un terminal avec :
``ollama serve``
Le serveur est alors accessible sur :

``http://localhost:11434``

### Installation des packages

```bash
uv venv
uv pip install -e .
```

## Déroulé de l'atelier

### Introduction théorique

- [Slides rappel sur les RAG](isup_ai/support presentation.pptx)
- [Démo streamlit] : 
    - Démarrer l'app streamlit avec ``uv run streamlit run app_rag.py``
    - Uploader ``docs_exemples\Auto_CG_SGRF_DOM_SOGESSUR_DocCliCont.pdf`` > ``Analyser avec Docling``
    - Attendre :)