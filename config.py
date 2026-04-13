import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

KGGEN_MODEL = os.getenv("KGGEN_MODEL", "anthropic/claude-sonnet-4-5")
KGGEN_TEMPERATURE = float(os.getenv("KGGEN_TEMPERATURE", "0.0"))
KGGEN_CHUNK_SIZE = int(os.getenv("KGGEN_CHUNK_SIZE", "5000"))


def kggen_api_key() -> str | None:
    """Pick the right API key based on the configured kg-gen model prefix."""
    model = (KGGEN_MODEL or "").lower()
    if model.startswith("anthropic/"):
        return ANTHROPIC_API_KEY
    if model.startswith("openai/") or model.startswith("gpt-"):
        return OPENAI_API_KEY
    return OPENAI_API_KEY or ANTHROPIC_API_KEY

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

NLI_MODEL = os.getenv("NLI_MODEL", "vectara/hallucination_evaluation_model")

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
NLI_ENTAILMENT_THRESHOLD = float(os.getenv("NLI_ENTAILMENT_THRESHOLD", "0.7"))

HIGH_CONFIDENCE_LABELS = {"HIGH"}
LOW_CONFIDENCE_LABELS = {"LOW", "MEDIUM"}
