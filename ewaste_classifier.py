"""
ewaste_classifier.py
--------------------
Stage 1 broad e-waste classification using IBM watsonx.ai
(granite-vision-3-3-2b multimodal model via Chat API).

This module is entirely self-contained and does NOT touch any existing
/predict or /predict-pcb logic.

Confidence note
---------------
granite-vision-3-3-2b is a generative model and does NOT produce
softmax-style probability scores. The confidence values returned by
this module are APPLICATION-LEVEL ESTIMATES derived from response
quality, not model-internal probabilities. This is documented on
every public-facing response via the API spec.
"""

import base64
import json
import logging
import os
import re
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Allowed categories ────────────────────────────────────────────────────────

ALLOWED_CATEGORIES = [
    "PCB / Motherboard",
    "Battery",
    "Mobile Phone",
    "Cable / Wire",
    "Computer Electronics",
    "Metal Scrap",
    "Other E-Waste",
    "Unknown",
]

# Aliases the model might plausibly return → canonical label
_ALIASES: dict[str, str] = {
    # PCB variants
    "pcb": "PCB / Motherboard",
    "pcb/motherboard": "PCB / Motherboard",
    "motherboard": "PCB / Motherboard",
    "circuit board": "PCB / Motherboard",
    "printed circuit board": "PCB / Motherboard",
    # Battery
    "battery": "Battery",
    "batteries": "Battery",
    "lead acid battery": "Battery",
    # Mobile
    "mobile phone": "Mobile Phone",
    "mobile": "Mobile Phone",
    "smartphone": "Mobile Phone",
    "phone": "Mobile Phone",
    "cell phone": "Mobile Phone",
    # Cable
    "cable": "Cable / Wire",
    "wire": "Cable / Wire",
    "cable/wire": "Cable / Wire",
    "cables": "Cable / Wire",
    "wires": "Cable / Wire",
    # Computer electronics
    "computer electronics": "Computer Electronics",
    "electronics": "Computer Electronics",
    "computer": "Computer Electronics",
    "laptop": "Computer Electronics",
    "desktop": "Computer Electronics",
    "hard drive": "Computer Electronics",
    "ram": "Computer Electronics",
    "cpu": "Computer Electronics",
    # Metal
    "metal scrap": "Metal Scrap",
    "metal": "Metal Scrap",
    "scrap metal": "Metal Scrap",
    "iron": "Metal Scrap",
    "copper": "Metal Scrap",
    "aluminium": "Metal Scrap",
    "aluminum": "Metal Scrap",
    # Other e-waste
    "other e-waste": "Other E-Waste",
    "other ewaste": "Other E-Waste",
    "e-waste": "Other E-Waste",
    "ewaste": "Other E-Waste",
    "electronic waste": "Other E-Waste",
    # Unknown
    "unknown": "Unknown",
    "none": "Unknown",
    "unrelated": "Unknown",
    "not e-waste": "Unknown",
}

# ── Confidence thresholds (application-level estimates) ───────────────────────

_CONFIDENCE_EXACT_MATCH = 0.90   # model returned an exact or well-known alias
_CONFIDENCE_FUZZY_MATCH = 0.72   # normalised after stripping punctuation/case
_CONFIDENCE_UNKNOWN     = 0.30   # model said Unknown or gave unrecognised output

_THRESHOLD_DETECTED     = 0.65   # detected=True only above this
_THRESHOLD_NEEDS_REVIEW = 0.82   # needs_review=False only above this

# ── IBM watsonx Chat API constants ────────────────────────────────────────────

_IAM_TOKEN_URL  = "https://iam.cloud.ibm.com/identity/token"
_CHAT_API_PATH = "/ml/v1/text/chat?version=2025-10-25"
_MODEL_ID       = "ibm/granite-vision-3-3-2b"
_MAX_NEW_TOKENS = 64          # category name is short; no need for more
_TIMEOUT_SECS   = 30

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an image classifier for an e-waste recycling application used in India.

Your only job is to look at the image and choose EXACTLY ONE category from the
list below. Do NOT invent categories. Do NOT add explanations.

Allowed categories:
- PCB / Motherboard
- Battery
- Mobile Phone
- Cable / Wire
- Computer Electronics
- Metal Scrap
- Other E-Waste
- Unknown

Rules:
1. If the image clearly shows one of the listed e-waste types, return that category.
2. If the image is ambiguous or shows multiple types, return the most prominent one.
3. If the image is unrelated to e-waste (food, vehicles, people, buildings, animals,
   furniture, documents, etc.), return: Unknown
4. If the image is blurry, dark, or unidentifiable, return: Unknown

Return your answer as valid JSON with exactly this structure and no other text:
{"category": "<exactly one category from the list above>"}
"""

# ── IAM token cache (module-level, simple in-process cache) ───────────────────

_token_cache: dict = {"token": None, "expires_at": 0.0}


def _get_iam_token(api_key: str) -> str:
    """Exchange an IBM Cloud API key for a short-lived IAM bearer token.

    Tokens are cached in memory and refreshed when they expire.
    Token lifetime is ~3600 s; we refresh 5 minutes early.
    """
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]  # type: ignore[return-value]

    resp = requests.post(
        _IAM_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    token: str = data["access_token"]
    expires_in: int = data.get("expires_in", 3600)
    _token_cache["token"] = token
    _token_cache["expires_at"] = now + expires_in - 300  # refresh 5 min early
    return token


# ── Core classification helpers ───────────────────────────────────────────────

def _encode_image(image_bytes: bytes, content_type: str) -> str:
    """Return a data-URI string suitable for the watsonx Chat API image_url field."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    # Normalise MIME type
    mime = content_type if content_type.startswith("image/") else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def normalize_category(raw: str) -> Optional[str]:
    """Map a model-returned string to one of the ALLOWED_CATEGORIES.

    Returns None if no mapping can be found.
    """
    cleaned = raw.strip().strip('"').strip("'")
    # 1. Exact match against allowed list
    if cleaned in ALLOWED_CATEGORIES:
        return cleaned
    # 2. Case-insensitive exact match
    lower = cleaned.lower()
    for cat in ALLOWED_CATEGORIES:
        if cat.lower() == lower:
            return cat
    # 3. Alias lookup (lowercase, strip punctuation)
    normalised = re.sub(r"[^a-z0-9 /]", "", lower).strip()
    if normalised in _ALIASES:
        return _ALIASES[normalised]
    # 4. Partial / fuzzy: does any allowed category appear as a substring?
    for cat in ALLOWED_CATEGORIES:
        if cat.lower() in lower:
            return cat
    return None


def validate_model_response(raw_text: str) -> tuple[str, float]:
    """Parse the model's JSON reply into (category, confidence).

    Returns ("Unknown", _CONFIDENCE_UNKNOWN) on any parse failure.
    """
    text = raw_text.strip()

    # Extract JSON object — model might wrap it in markdown fences
    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not json_match:
        logger.warning("No JSON found in model response: %r", text)
        return "Unknown", _CONFIDENCE_UNKNOWN

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error: %s — raw: %r", exc, text)
        return "Unknown", _CONFIDENCE_UNKNOWN

    raw_category = str(data.get("category", "")).strip()
    if not raw_category:
        return "Unknown", _CONFIDENCE_UNKNOWN

    # Exact match in allowed list
    if raw_category in ALLOWED_CATEGORIES:
        conf = _CONFIDENCE_UNKNOWN if raw_category == "Unknown" else _CONFIDENCE_EXACT_MATCH
        return raw_category, conf

    # Try normalisation
    mapped = normalize_category(raw_category)
    if mapped:
        conf = _CONFIDENCE_UNKNOWN if mapped == "Unknown" else _CONFIDENCE_FUZZY_MATCH
        return mapped, conf

    logger.warning("Model returned unknown category %r; mapping to Unknown", raw_category)
    return "Unknown", _CONFIDENCE_UNKNOWN


def calculate_review_status(category: str, confidence: float) -> tuple[bool, bool]:
    """Return (detected, needs_review) based on category and confidence."""
    if category == "Unknown" or confidence < _THRESHOLD_DETECTED:
        return False, True
    needs_review = confidence < _THRESHOLD_NEEDS_REVIEW
    return True, needs_review


# ── Main public function ──────────────────────────────────────────────────────

def classify_ewaste_image(
    image_bytes: bytes,
    content_type: str = "image/jpeg",
) -> dict:
    """Classify an image using IBM watsonx.ai granite-vision-3-3-2b.

    Returns a dict matching the /classify-ewaste response schema:
        {
            "detected": bool,
            "category": str,
            "confidence": float,
            "needs_review": bool
        }

    Never raises — all exceptions are caught and a safe Unknown response
    is returned instead.
    """
    safe_response = {
        "detected": False,
        "category": "Unknown",
        "confidence": 0.0,
        "needs_review": True,
    }

    # ── Load credentials from environment ────────────────────────────────────
    api_key    = os.environ.get("IBM_API_KEY", "").strip()
    project_id = os.environ.get("IBM_PROJECT_ID", "").strip()
    wx_url     = os.environ.get("IBM_WATSONX_URL", "").strip().rstrip("/")

    if not api_key or not project_id or not wx_url:
        logger.error(
            "IBM credentials not set. Required: IBM_API_KEY, "
            "IBM_PROJECT_ID, IBM_WATSONX_URL"
        )
        return safe_response

    # ── Get bearer token ─────────────────────────────────────────────────────
    try:
        token = _get_iam_token(api_key)
    except Exception as exc:
        logger.error("Failed to obtain IAM token: %s", exc)
        return safe_response

    # ── Build request ─────────────────────────────────────────────────────────
    image_data_uri = _encode_image(image_bytes, content_type)

    payload = {
        "model_id": _MODEL_ID,
        "project_id": project_id,
        "messages": [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Classify this image into exactly one of the allowed "
                            "categories. Return only JSON."
                        ),
                    },
                ],
            },
        ],
        "max_tokens": _MAX_NEW_TOKENS,
    }

    endpoint = wx_url + _CHAT_API_PATH

    # ── Call IBM watsonx Chat API ─────────────────────────────────────────────
    try:
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=_TIMEOUT_SECS,
        )
        resp.raise_for_status()
    except requests.Timeout:
        logger.error("watsonx API timed out after %ds", _TIMEOUT_SECS)
        return safe_response
    except requests.RequestException as exc:
        logger.error("watsonx API request failed: %s", exc)
        return safe_response

    # ── Parse response ────────────────────────────────────────────────────────
    try:
        api_data = resp.json()
        # OpenAI-compatible chat response structure
        raw_text: str = (
            api_data["choices"][0]["message"]["content"]
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.error("Unexpected watsonx response shape: %s — body: %s", exc, resp.text[:500])
        return safe_response

    logger.info("watsonx raw response: %r", raw_text)

    # ── Validate and score ────────────────────────────────────────────────────
    category, confidence = validate_model_response(raw_text)
    detected, needs_review = calculate_review_status(category, confidence)

    return {
        "detected": detected,
        "category": category,
        "confidence": round(confidence, 2),
        "needs_review": needs_review,
    }
