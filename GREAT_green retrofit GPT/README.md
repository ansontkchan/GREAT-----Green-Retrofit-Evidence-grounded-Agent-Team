# GREAT: Multi-Agent RAG for Standards-Grounded Green Retrofit Planning

**GREAT** stands for **Green Retrofit Evidence-grounded Agent Team**.

This repository contains a research MVP for comparing three systems for strategic green retrofit planning:

1. **Generic LLM** without retrieval
2. **Single-agent RAG** over one combined evidence corpus
3. **GREAT multi-agent RAG** with Energy, BREEAM, WLCA, Cost, and Planner agents

The intended paper framing is:

> **GREAT: Benchmarking Multi-Agent Retrieval-Augmented Generation for Standards-Grounded Green Retrofit Planning**

The current prototype is deliberately a controlled, text-based multi-agent RAG benchmark. It is **not** yet a full enterprise agentic-RAG system with live tool routing, BIM/digital-twin integration, simulation APIs, or GIS/weather querying. Those are future extensions.

---

## 1. Repository structure

```text
.
├── data/                  # RAG evidence corpus: standards and guidance PDFs only
│   ├── breeam/
│   ├── wlca/
│   ├── energy/
│   └── cost/
├── cases/                 # Evaluation-only UKGBC cases; do NOT ingest into RAG
│   └── ukgbc/
├── logs/                  # Generated evaluation outputs
├── src/
│   ├── config.py
│   ├── rag_store.py
│   ├── ingest_pdfs.py
│   ├── agents.py
│   ├── profile_assistant.py
│   ├── great_app.py
│   ├── greenretrofit_app.py  # legacy wrapper
│   ├── case_loader.py
│   ├── metrics.py
│   └── evaluate_cases.py
├── streamlit_app.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 2. Environment setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Create your `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-your-real-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

---

## 3. Add the RAG evidence corpus

Place PDFs into:

```text
data/breeam/   # BREEAM standards and guidance
data/wlca/     # RICS WLCA and carbon assessment guidance
data/energy/   # retrofit and energy-efficiency guidance
data/cost/     # cost, feasibility, and business-case guidance
```

Do **not** place UKGBC case-study JSON files into `data/`. They belong in `cases/ukgbc/` and are used only for evaluation.

---

## 4. Ingest PDFs into the vector database

```bash
python -m src.ingest_pdfs --agent breeam "data/breeam/*.pdf"
python -m src.ingest_pdfs --agent wlca "data/wlca/*.pdf"
python -m src.ingest_pdfs --agent energy "data/energy/*.pdf"
python -m src.ingest_pdfs --agent cost "data/cost/*.pdf"
```

This creates a local Chroma vector store at:

```text
.chroma_great/
```

---

## 5. Run the CLI prototype

```bash
python -m src.great_app
```

The CLI runs:

1. Generic LLM without RAG
2. Single-agent RAG
3. GREAT multi-agent RAG with Planner output

The old command also works:

```bash
python -m src.greenretrofit_app
```

---

## 6. Run the Streamlit prototype

```bash
streamlit run streamlit_app.py
```

The Streamlit app allows a user or expert to set:

- profession / role
- primary concern
- building location
- scope
- timeframe
- BREEAM rating appetite
- budget band
- risk appetite
- extra constraints

Then it compares the three systems.

---

## 7. UKGBC case benchmark

UKGBC case studies should be manually converted into JSON files and stored in:

```text
cases/ukgbc/
```

Example fields:

```json
{
  "id": "tempo",
  "title": "Tempo",
  "source_url": "https://ukgbc.org/resources/tempo/",
  "building_type": "office",
  "location": "London, UK",
  "goals": ["BREEAM Excellent", "net-zero operational carbon"],
  "constraints": ["occupied during works"],
  "measures": ["heat pump", "PV", "LED lighting", "BMS"],
  "lessons": []
}
```

The UKGBC cases are **held-out reference cases**. They should not be indexed into the RAG database for the main benchmark.

Check cases:

```bash
python -m src.case_loader
```

Run evaluation:

```bash
python -m src.evaluate_cases
```

The script outputs logs into `logs/` and computes preliminary alignment metrics:

- measure precision
- measure recall
- F1
- Jaccard similarity

These metrics should be interpreted as **alignment with documented case strategies**, not absolute correctness.

---

## 8. Research positioning

This MVP supports the first-paper claim:

> A domain-specialised multi-agent RAG system can be benchmarked against generic LLMs and single-agent RAG for standards-grounded strategic retrofit planning.

The current prototype does not claim to be a fully autonomous enterprise agentic-RAG platform. It is a controlled research benchmark for:

- standards-grounded retrieval
- domain-specialised multi-agent decomposition
- strategy alignment with unseen UKGBC case studies
- expert validation of professional usefulness

---

## 9. Essential Git workflow

After every meaningful change:

```bash
git status
git add .
git commit -m "Describe what changed"
git push origin main
```

Never commit:

- `.env`
- `.venv/`
- `.chroma_great/`
- proprietary PDFs in `data/`

These are excluded in `.gitignore`.
