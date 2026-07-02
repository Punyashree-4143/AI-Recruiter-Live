from threading import Lock
import traceback

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from search_candidates import run_recruitment_pipeline


pipeline_lock = Lock()


class CandidateSearchRequest(BaseModel):
    job_description: str = Field(
        ...,
        min_length=20,
        description="The complete job description to analyze.",
    )


app = FastAPI(
    title="AI Recruiter API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://ai-recruiter-live.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_pipeline(job_description):
    with pipeline_lock:
        return run_recruitment_pipeline(
            job_description
        )


@app.get("/")
def root():
    return {
        "message": "AI Recruiter API Running"
    }


@app.post("/search-candidates")
async def search_candidates(
    request: CandidateSearchRequest,
):
    try:
        result = await run_in_threadpool(
            _run_pipeline,
            request.job_description.strip(),
        )

        return jsonable_encoder(result)

    except Exception as error:
        print("\n========== PIPELINE ERROR ==========")
        traceback.print_exc()
        print("ERROR:", repr(error))
        print("====================================\n")

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )