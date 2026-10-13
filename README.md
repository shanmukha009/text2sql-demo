# Text2SQL — Natural Language to SQL (live demo)

Type a question in plain English; this app generates the SQL and runs it on a sample
sales database, showing both the SQL and the result.

This is a lightweight, deployable demo of my **multi-agent Text2SQL system**. The full
version fine-tunes **Mistral-7B with LoRA** and uses a **CrewAI** multi-agent pipeline
(schema reasoning, SQL optimization, ambiguity resolution) with Flask, BigQuery, and
Docker, reaching **80% exact-match accuracy**.

Full project: https://github.com/shanmukha009/DATA-266-Text2SQL

## How it works
- Zero setup: a built-in schema-aware translator handles common analytical questions
  (top N, totals, group-by category/city/segment/month, counts, averages) so the demo
  works out of the box.
- Optional LLM: add an OpenAI key as a Streamlit secret named `OPENAI_API_KEY` and the
  app will use an LLM to answer open-ended questions over the schema.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech
Streamlit · SQLite · Pandas · (optional OpenAI) · full project: Mistral-7B LoRA + CrewAI
