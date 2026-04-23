import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.config import DEFAULT_BACKEND, DEFAULT_POLICY_PATH, DEFAULT_RUBRIC_PATH, resolve_model
from src.evaluator import evaluate_document
from src.llm import setup_llm

app = FastAPI()

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    async def event_generator():
        try:
            policy = DEFAULT_POLICY_PATH
            if not policy.exists() and not policy.is_absolute():
                policy = Path.cwd() / DEFAULT_POLICY_PATH

            rubric = DEFAULT_RUBRIC_PATH
            if not rubric.exists() and not rubric.is_absolute():
                rubric = Path.cwd() / DEFAULT_RUBRIC_PATH

            llm = setup_llm(resolve_model(None, DEFAULT_BACKEND), DEFAULT_BACKEND)
            for event in evaluate_document(tmp_path, policy, rubric, llm):
                yield json.dumps(event) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
