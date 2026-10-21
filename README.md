# Local AI Studio — Windows 11 starter

A starter for a Venice-style local chat application with your own branding, configurable system instructions, local inference through `llama.cpp`, and local document retrieval through Chroma. It does **not** use Ollama or cloud AI APIs.

## What this starter does

- Serves a custom dark chat interface from FastAPI.
- Sends chat requests to a `llama.cpp` server bound to `127.0.0.1`.
- Saves branding and behavior settings to `config/settings.json`.
- Indexes local PDF, TXT, Markdown, and selected source-code files into a persistent Chroma collection.
- Retrieves relevant passages and places them in the model's context.

## What it does not do yet

- It does not train a foundation model from scratch.
- It does not persist chat history across browser restarts.
- PDF text extraction will not read scanned pages without OCR.
- It is a local development starter, not hardened for public multi-user deployment.
- The first ingestion may download Chroma's default embedding model. Complete setup while online, then test offline.

## Requirements

- Windows 11, 64-bit.
- Python 3.11 or 3.12 recommended.
- Several GB of free disk space for Python dependencies and model weights.
- A GGUF chat/instruct model whose license permits your intended use.
- `llama.cpp` Windows release or a locally built `llama-server.exe`.

Official resources:
- llama.cpp releases/source: https://github.com/ggml-org/llama.cpp
- llama.cpp server instructions: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- Python: https://www.python.org/downloads/windows/
- Chroma docs: https://docs.trychroma.com/
- PyTorch / Transformers (optional future fine-tuning): https://huggingface.co/docs/transformers/installation

## Step 1 — install Python

Install Python 3.11 or 3.12 from python.org. In the installer, select **Add python.exe to PATH**. Open a new PowerShell and verify:

```powershell
py --version
```

## Step 2 — unpack the project

Extract this ZIP to a simple path, for example:

`C:\Users\YOUR_USERNAME\Documents\LocalAIStudio`

Open PowerShell in that folder.

## Step 3 — create the Python environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in that PowerShell window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This changes policy only for the current PowerShell process.

## Step 4 — get llama.cpp (no Ollama)

Download a Windows release from the official llama.cpp GitHub releases page, or build it from source using the project's Windows instructions. Extract the release into a folder such as:

`C:\AI\llama.cpp`

Locate `llama-server.exe`.

## Step 5 — obtain a compatible local model

Download a GGUF chat/instruct model from a reputable model publisher or official model repository. Check its model license and any use restrictions. Choose a quantization that fits your RAM/VRAM; a model file that is too large may be very slow or fail to load.

Create a folder such as `C:\AI\models` and put the model there. Do not use a model file from an unknown source.

## Step 6 — start the local model server

In a separate PowerShell window, run (replace paths and filename):

```powershell
cd C:\AI\llama.cpp
.\llama-server.exe -m "C:\AI\models\YOUR-MODEL.gguf" --host 127.0.0.1 --port 8080 -c 4096
```

Keep that window open. The server should listen only on your own computer. Visit `http://127.0.0.1:8080` to check its built-in interface. The model's actual filename must replace `YOUR-MODEL.gguf`.

If the server rejects the model, check that the file is a compatible GGUF model and review the model's chat-template guidance.

## Step 7 — configure the project

Edit `config\settings.json`. Change `app_name`, `tagline`, `system_prompt`, and model settings as desired. The `model_name` value is sent in the API request; many local llama.cpp builds accept a placeholder name, but if your model/server expects a specific identifier, set it accordingly.

## Step 8 — add your own knowledge

Copy documents into `knowledge\raw`. Supported types in this starter: PDF, TXT, MD, PY, C, CPP, H, JSON, CSV.

With the virtual environment activated, run:

```powershell
python scripts\ingest.py
```

The first run may download a default embedding model for local semantic search. Once downloaded and cached, the embedding model can be used offline. If you require a strictly air-gapped installation, download and verify all packages and model files in advance, then transfer them to the offline machine.

## Step 9 — run your chatbot

From the project root, with `.venv` activated:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

`http://127.0.0.1:8000`

The `llama-server.exe` window and the FastAPI window must both remain running.

## Troubleshooting

- **Could not reach local model server:** start llama-server first and verify port 8080.
- **Model load error:** confirm the path and model format; check model RAM/VRAM requirements.
- **No knowledge retrieved:** add supported files and rerun `python scripts\ingest.py`.
- **Scanned PDF returns no text:** OCR is needed; this starter only extracts embedded text.
- **Slow responses:** use a smaller/less demanding quantization or reduce context length.
- **Port already in use:** choose another port and update `config/settings.json`.

## Privacy and security

- This starter binds the web app and model server to loopback (`127.0.0.1`) so other devices cannot connect by default.
- Do not change the host to `0.0.0.0` or expose ports to your network before adding authentication, authorization, TLS, rate limiting, and appropriate data controls.
- Local processing reduces external disclosure but does not itself encrypt files or protect the computer from malware or other local users.
- Retrieved documents are untrusted content. The application instructs the model to treat them as evidence rather than system instructions, but prompt injection defenses are not perfect.
- You control application configuration; model behavior still depends on model weights, training, and context.
