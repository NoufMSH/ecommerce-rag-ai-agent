# E-Commerce Product Q&A AI Agent (RAG)

A university course project: a single AI agent that answers customer
questions about products (availability, reviews, common issues, description,
recommendation) using **Retrieval-Augmented Generation (RAG)** over four data
sources — product descriptions, reviews, inventory, and support tickets.

> **Note:** This project uses the free Gemini API. Before running it, create a
> free API key from [Google AI Studio](https://aistudio.google.com/apikey) and
> add it to the `.env` file (see Quick Start below).

## Quick Start (Docker)

The only prerequisite is [Docker Desktop](https://www.docker.com/products/docker-desktop/) — installed and running. No Python, Java, or other dependencies needed on your machine.

```bash
# 1. Clone the repo
git clone https://github.com/NoufMSH/ecommerce-rag-ai-agent.git
cd ecommerce-rag-ai-agent

# 2. Add your own free Gemini API key (get one at https://aistudio.google.com/apikey)
cp .env.example .env
# then edit .env and paste your key into GEMINI_API_KEY

# 3. Build and run
docker compose up --build
```

Open **http://localhost:8501** and ask a question, e.g. *"Is this product available?"*

First run takes a few minutes to build (installs dependencies, and bakes the PySpark ETL + embedding index into the image); after that, `docker compose up` starts in seconds. Stop with `Ctrl+C`, then `docker compose down` to clean up.

Don't want to use Docker? See [Option B](#option-b--run-locally-without-docker) below.

## 1. Project idea

Customers ask natural questions like *"Is this available?"* or *"What do
people think of it?"*. Instead of manually searching separate spreadsheets,
this project builds one small RAG agent that retrieves the relevant facts
about a product and uses a free LLM (Google Gemini) to turn them into a
grounded, natural-language answer.

## 2. How this project meets the requirements

| Requirement | How it's met |
|---|---|
| Build an AI Agent | `src/agent.py` implements `ProductQAAgent`, a single agent that retrieves context and generates answers. |
| One AI Agent is enough | Only one agent class exists; no multi-agent orchestration. |
| Use RAG | The agent embeds the question, retrieves the most relevant chunks by cosine similarity, and feeds them to the LLM as grounding context before generating an answer. |
| Use Python | The entire project (ETL, indexing, agent, UI) is Python. |
| Use PySpark for data processing | `src/etl_pyspark.py` uses Spark DataFrames to join and aggregate the raw CSVs (joins, `groupBy`/`agg`, `when`/`otherwise`) into the knowledge base. |
| Use a free LLM | Google Gemini API (`gemini-2.5-flash`), free tier via an API key from Google AI Studio. |
| Big data handling (bonus) | The ETL is written with genuine Spark transformations, so it would scale to a much larger `reviews.csv`/`support_tickets.csv` unchanged — only the sample data volume is kept small on purpose. |
| Provide code + sample data | All code is in this repo; sample CSVs are in `data/raw/`. |

## 3. Architecture

```
data/raw/*.csv  --(PySpark ETL)-->  data/processed/knowledge_base.jsonl
                                            |
                                  (sentence-transformers embeddings)
                                            v
                          index/embeddings.npy + index/metadata.json
                                            |
                                            v
      app.py (Streamlit)  -->  ProductQAAgent  -->  Gemini API
```

1. **ETL (`src/etl_pyspark.py`)** — loads the 4 raw CSVs with PySpark, joins
   them on `product_id`, aggregates reviews (average rating, review count, a
   Recommended/Mixed/Not recommended label) and support tickets (issue list),
   and writes one JSON-lines file with 4 text "chunks" per product:
   `description`, `reviews`, `inventory`, `support`.
2. **Indexing (`src/build_index.py`)** — embeds every chunk locally with the
   free `sentence-transformers` model `all-MiniLM-L6-v2` and saves the vectors
   + metadata to `index/`.
3. **Agent (`src/agent.py`)** — `ProductQAAgent.answer(question, product_id)`:
   embeds the question, finds the top matching chunks via NumPy cosine
   similarity, builds a grounded prompt, and calls Gemini for the final
   answer.
4. **UI (`app.py`)** — a minimal Streamlit chat app: pick a product, ask a
   question, see the answer and (in an expander) which chunks grounded it.

## 4. Setup & run

Either way, you need your own **free** Gemini API key (there's no way around
this — it's the "free LLM" the assignment requires, and keys can't be shared
publicly in the repo): get one at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey), then:

```bash
cp .env.example .env
# edit .env and paste your key into GEMINI_API_KEY
```

### Option A — Docker (recommended, minimal setup)

See [Quick Start](#quick-start-docker) above — just `docker compose up --build`.
Only re-run with `--build` if you change the code or the raw CSVs.

### Option B — Run locally without Docker

**Prerequisites:** Python 3.9+, and Java 8/11/17 (required by PySpark — see
[Adoptium](https://adoptium.net/) if you don't have a JDK installed).

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline, then the app (steps 1-2 only need to be re-run if the raw data changes)
python src/etl_pyspark.py    # builds data/processed/knowledge_base.jsonl
python src/build_index.py    # builds index/embeddings.npy + index/metadata.json
streamlit run app.py         # launches the chat UI in your browser
```

You can also sanity-check the agent directly from the terminal without the UI:

```bash
python src/agent.py
```

## 5. Example questions and expected answers

Using the sample data, if you select **P001 - Wireless Earbuds X200**:

| Question | Expected grounded answer (from which chunk) |
|---|---|
| Is this product available? | Yes — in stock, 120 units at Warehouse A - New Jersey. *(inventory chunk)* |
| What do customers think about this product? | Average rating 4.0/5 from 5 reviews, mostly positive comments about battery life and sound, one complaint about disconnects. *(reviews chunk)* |
| Does this product have common issues? | Yes — customers report intermittent Bluetooth disconnects after a firmware update (2 support tickets). *(support chunk)* |
| What is the product description? | Compact true-wireless earbuds, 24-hour battery with case, active noise cancellation, touch controls, 3 ear tip sizes. *(description chunk)* |
| Is this product recommended? | Yes — labeled "Recommended" based on a 4.0/5 average rating, though the firmware disconnect issue is worth noting. *(reviews + support chunks)* |

If you instead select **P004 - Portable Bluetooth Speaker Boom Mini** and ask
*"Is this product available?"*, the agent should say it's currently **out of
stock** (0 units) with an expected restock date of 2026-07-20 — showing the
agent adapts to whichever product's data is retrieved.

## 6. Limitations

This is a course project, not a production system:
- No authentication, logging, or monitoring.
- Sample data is small and hand-written for clarity, not scraped from a real store.
- Error handling is minimal (just enough to fail gracefully if the API key is missing or a request fails).
- The Gemini free tier has rate limits; heavy testing may require waiting between requests.
