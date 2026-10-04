<img src="https://raw.githubusercontent.com/drunkmonkkk/drunkmonkkk/main/assets/backend-cloud.svg" width="100%" alt="Kabadiwala cloud API — e-waste classification and PCB analysis">

# Kabadiwala · Cloud inference API

FastAPI endpoints for e-waste image classification, PCB component analysis, and grounded object detection.

[Flutter app](https://github.com/drunkmonkkk/kabadiwala_connect) · [Local YOLO API](https://github.com/drunkmonkkk/kabadiwala_backend) · [Classifier](ewaste_classifier.py)

## Three inference paths

| Endpoint | Provider / purpose |
| --- | --- |
| `POST /classify-ewaste` | IBM watsonx.ai: broad e-waste classification with a review flag |
| `POST /predict-pcb` | Roboflow: PCB component detection and prototype valuation |
| `POST /classify-grounding` | Grounding DINO through an external service: object detection |

All three routes accept a multipart image in the `file` field. `GET /` and `GET /health` provide service responses.

## Run locally

From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

On Windows, activate with `.venv\\Scripts\\activate`. Configure the IBM settings described in [.env.example](.env.example). The PCB route additionally reads `ROBOFLOW_API_KEY`; the grounding route reads `DDS_API_TOKEN`.

Interactive API documentation: `http://127.0.0.1:8000/docs`.

Example using your own image:

```bash
curl -X POST http://127.0.0.1:8000/classify-ewaste -F "file=@scrap.jpg"
```

## Integration notes

- This variant does **not** expose `POST /predict`. The Flutter app's general prediction service currently expects that route from the [local YOLO backend](https://github.com/drunkmonkkk/kabadiwala_backend).
- PCB valuations are prototype estimates; they are not verified market quotes.
- The e-waste confidence value is an application-level estimate, not a model probability.
- The health route returns a static response and does not verify external provider availability.

## Repository map

[main.py](main.py) contains the routes. [ewaste_classifier.py](ewaste_classifier.py) implements the IBM classification path. [requirements.txt](requirements.txt) lists dependencies.
