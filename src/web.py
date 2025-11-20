import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.config import DEFAULT_MODEL_NAME, DEFAULT_POLICY_PATH, DEFAULT_RUBRIC_PATH
from src.evaluator import evaluate_document

app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    # Create a temporary file to save the uploaded content
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Ensure policy and rubric exist relative to CWD if not absolute
        policy = DEFAULT_POLICY_PATH
        if not policy.exists() and not policy.is_absolute():
            policy = Path.cwd() / DEFAULT_POLICY_PATH

        rubric = DEFAULT_RUBRIC_PATH
        if not rubric.exists() and not rubric.is_absolute():
            rubric = Path.cwd() / DEFAULT_RUBRIC_PATH

        # Run evaluation
        result = evaluate_document(
            target_path=tmp_path,
            policy_path=policy,
            rubric_path=rubric,
            model_name=DEFAULT_MODEL_NAME,
            verbose=True,
        )
        return {"result": result}

    except Exception as e:
        return {"result": f"Error during evaluation: {str(e)}"}

    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()
