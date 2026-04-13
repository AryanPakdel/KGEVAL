# KGEVAL — Bidirectional KG Framework for RAG Faithfulness

Phase 1 core pipeline: given a source document and an LLM response, extract
knowledge graphs from both, match triples semantically, classify each response
triple as Grounded / Contradicted / Fabricated / Uncertain, and resolve
uncertain verdicts with an NLI fallback.

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Configure

Copy `.env.example` to `.env` and set your Anthropic API key:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

## Run

```bash
python main.py --source source.txt --response response.txt
```

Output is a JSON report containing the faithfulness score, source-KG
NER coverage, and a per-triple verdict list with supporting evidence.

## Pipeline (6 stages)

1. Extract triples with confidence from source and response via Claude.
2. Check source-KG coverage against spaCy NER entities.
3. Embed triples (Sentence-BERT) and match response→source by cosine similarity.
4. Classify each response triple: grounded / contradicted / fabricated / uncertain.
5. Run NLI fallback (HHEM) on uncertain triples to reclassify.
6. Emit verdicts + faithfulness score.
