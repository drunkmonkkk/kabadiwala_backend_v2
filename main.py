from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


import os
from dotenv import load_dotenv
import requests
app = FastAPI(
    title="Kabadiwala Connect AI API",
    version="1.0.0"
)
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DISPLAY_NAMES = {
    "Battery_Waste": "Battery",
    "PCB": "PCB / E-Waste",
    "Metal_Waste": "Metal Scrap",
    "Mobile": "Mobile Phone",
    "Keyboard": "Keyboard",
    "Mouse": "Mouse",
    "Light_Bulb": "Light Bulb",
    "Plastic_Waste": "Plastic",
    "Paper_Waste": "Paper",
    "Glass_Waste": "Glass",
    "Medical_Waste": "Medical Waste",
    "Organic_Waste": "Organic Waste",
}

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Kabadiwala Connect AI API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }



@app.post("/predict-pcb")
async def predict_pcb(file: UploadFile = File(...)):
    try:
        api_key = os.environ.get("ROBOFLOW_API_KEY")

        if not api_key:
            return {
                "success": False,
                "error": "ROBOFLOW_API_KEY is not set"
            }

        image_bytes = await file.read()

        url = (
    "https://detect.roboflow.com/"
    "det-pcbs-motherboards-1vzmv/1"
    f"?api_key={api_key}&confidence=10"
)

        response = requests.post(
            url,
            files={
                "file": (
                    file.filename,
                    image_bytes,
                    file.content_type or "image/jpeg",
                )
            },
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()
        predictions = data.get("predictions", [])

        # Keep only somewhat useful predictions
        filtered = [p for p in predictions if p.get("confidence", 0) >= 0.15]

        # Count component classes
        component_counts = {}
        for p in filtered:
            cls = p["class"]
            component_counts[cls] = component_counts.get(cls, 0) + 1

        # Turn into list sorted by count
        components = [
            {"name": k, "count": v}
            for k, v in sorted(component_counts.items(), key=lambda item: item[1], reverse=True)
        ]

        # Decide overall category
        detected_classes = set(component_counts.keys())

        if "motherboard" in detected_classes or "connector" in detected_classes or "capacitor" in detected_classes:
            final_class = "PCB"
            display_name = "PCB / E-Waste"
            detected = True
            confidence = max([p["confidence"] for p in filtered], default=0.0)
        else:
            final_class = None
            display_name = None
            detected = False
            confidence = 0.0
        # Prototype PCB grading logic.
        # Later these rates can come from verified recycler/market data.
        valuable_classes = {
            "cpu",
            "ic",
            "processor",
            "memory",
            "ram",
            "motherboard",
        }

        medium_value_classes = {
            "capacitor",
            "connector",
            "inductor",
            "transformer",
        }

        valuable_found = detected_classes.intersection(valuable_classes)
        medium_found = detected_classes.intersection(medium_value_classes)

        if "motherboard" in detected_classes and valuable_found:
            grade = "High-grade E-Waste"
            rate_min = 260
            rate_max = 340
        elif "motherboard" in detected_classes or len(medium_found) >= 2:
            grade = "Medium-grade E-Waste"
            rate_min = 180
            rate_max = 260
        else:
            grade = "Low-grade E-Waste"
            rate_min = 80
            rate_max = 180

        return {
    "success": True,
    "detected": detected,
    "prediction": {
        "class": final_class,
        "display_name": display_name,
        "confidence": round(confidence, 4),
    } if detected else None,

    "components": components,

    "valuation": {
        "grade": grade,
        "rate_min_per_kg": rate_min,
        "rate_max_per_kg": rate_max,
        "currency": "INR",
        "note": "Prototype estimate; final recycler price may vary.",
    } if detected else None,

    "raw_count": len(predictions),
    "filtered_count": len(filtered),
}

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }



# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Broad e-waste classification (IBM watsonx.ai)
# Isolated from /predict and /predict-pcb.
# To disable: comment out this entire block.
# ═════════════════════════════════════════════════════════════════════════════

import logging as _logging

# Load .env file if present (dev convenience; ignored when env vars already set)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass  # python-dotenv optional; env vars can be set another way

from ewaste_classifier import classify_ewaste_image as _classify_ewaste_image

_ewaste_logger = _logging.getLogger("ewaste")


@app.post("/classify-ewaste")
async def classify_ewaste(file: UploadFile = File(...)):
    """
    Stage 1 broad e-waste classifier powered by IBM watsonx.ai.

    Accepts: multipart/form-data with field 'file' (PNG or JPEG image).

    Returns:
        {
          "detected": bool,
          "category": one of the 8 allowed categories,
          "confidence": float (application-level estimate, not model probability),
          "needs_review": bool
        }

    Confidence note: confidence values are application-level estimates derived
    from response quality, NOT model-internal probabilities. The underlying
    IBM granite-vision-3-3-2b model does not expose softmax scores.

    Thresholds:
        >= 0.82  detected=true,  needs_review=false
        >= 0.65  detected=true,  needs_review=true
        <  0.65  detected=false, needs_review=true
    """
    # ── Validate file ─────────────────────────────────────────────────────────
    if file is None:
        return {
            "detected": False,
            "category": "Unknown",
            "confidence": 0.0,
            "needs_review": True,
        }

    content_type = file.content_type or "image/jpeg"
    if not content_type.startswith("image/"):
        return {
            "detected": False,
            "category": "Unknown",
            "confidence": 0.0,
            "needs_review": True,
        }

    # ── Read image bytes ──────────────────────────────────────────────────────
    try:
        image_bytes = await file.read()
    except Exception as exc:
        _ewaste_logger.error("Failed to read uploaded file: %s", exc)
        return {
            "detected": False,
            "category": "Unknown",
            "confidence": 0.0,
            "needs_review": True,
        }

    if not image_bytes:
        return {
            "detected": False,
            "category": "Unknown",
            "confidence": 0.0,
            "needs_review": True,
        }

    # ── Classify ──────────────────────────────────────────────────────────────
    result = _classify_ewaste_image(image_bytes, content_type)
    _ewaste_logger.info(
        "classify-ewaste result for %s: %s", file.filename, result
    )
    return result


@app.post("/classify-grounding")
async def classify_grounding(file: UploadFile = File(...)):
    try:
        token = os.getenv("DDS_API_TOKEN")

        if not token:
            return {
                "success": False,
                "error": "DDS_API_TOKEN is not configured",
            }

        image_bytes = await file.read()

        import base64
        import time

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        content_type = file.content_type or "image/jpeg"
        image_data = f"data:{content_type};base64,{image_base64}"

        payload = {
            "model": "GroundingDino-1.6-Pro",
            "image": image_data,
            "prompt": {
                "type": "text",
                "text": (
                    "battery."
                    "printed circuit board."
                    "copper wire."
                    "keyboard."
                    "computer mouse."
                    "mobile phone"
                ),
            },
            "targets": ["bbox"],
        }

        headers = {
            "Content-Type": "application/json",
            "Token": token,
        }

        # 1. Submit task
        response = requests.post(
            "https://api.deepdataspace.com/v2/task/grounding_dino/detection",
            headers=headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        submit_data = response.json()
        task_uuid = submit_data.get("data", {}).get("task_uuid")

        if not task_uuid:
            return {
                "success": False,
                "error": "No task_uuid returned",
                "dds_response": submit_data,
            }

        # 2. Poll task
        for _ in range(20):
            status_response = requests.get(
                f"https://api.deepdataspace.com/v2/task_status/{task_uuid}",
                headers={"Token": token},
                timeout=30,
            )

            status_response.raise_for_status()
            status_data = status_response.json()
            data = status_data.get("data", {})
            status = data.get("status")

            if status in ["success", "succeeded", "finished", "completed"]:
                objects = (
                    status_data
                    .get("data", {})
                    .get("result", {})
                    .get("objects", [])
                )

                detections = []
                for obj in objects:
                    detections.append(
                        {
                            "label": obj.get("category"),
                            "confidence": round(float(obj.get("score", 0)), 4),
                            "bbox": obj.get("bbox", []),
                        }
                    )

                CATEGORY_MAP = {
    "copper wire": "Copper Wire",
    "battery": "Battery",
    "printed circuit board": "PCB / Motherboard",
    "keyboard": "Keyboard",
    "computer mouse": "Computer Mouse",
    "mobile phone": "Mobile Phone",
}

                if detections:
                    best = max(detections, key=lambda x: x["confidence"])
                    min_confidence = 0.35
                    mapped_category = CATEGORY_MAP.get(best["label"], "Unknown")
                    if (
                        best["confidence"] >= min_confidence
                        and mapped_category != "Unknown"
                    ):
                        return {
                            "success": True,
                            "detected": True,
                            "category": mapped_category,
                            "confidence": best["confidence"],
                            "needs_review": False,
                            "detections": detections,
                        }

                    return {
                        "success": True,
                        "detected": False,
                        "category": "Unknown",
                        "confidence": best["confidence"],
                        "needs_review": True,
                        "detections": detections,
                    }

                return {
    "success": True,
    "detected": False,
    "category": "Unknown",
    "confidence": 0.0,
    "needs_review": True,
    "detections": [],
}

            if status in ["failed", "error"]:
                return {
                    "success": False,
                    "task_uuid": task_uuid,
                    "dds_response": status_data,
                }

            time.sleep(1)

        return {
            "success": False,
            "error": "Grounding DINO task timed out",
            "task_uuid": task_uuid,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }