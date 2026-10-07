"""OpenAI client shared by the answer service and agent."""
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def get_client():
    """Return (client, model), or None if no OpenAI key is configured or the package is missing."""
    if not settings.openai_api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed — LLM disabled")
        return None
    return OpenAI(api_key=settings.openai_api_key), settings.openai_model_name
