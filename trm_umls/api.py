"""
Local API for TRM-UMLS.

This is designed for running on your own machine (often with PHI in memory).
It does not persist uploaded text or results.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import TRMUMLSPipeline


def _env_list(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default).strip()
    return [p.strip() for p in raw.split(",") if p.strip()]


class ExtractOptions(BaseModel):
    threshold: float = Field(0.55, ge=0.0, le=1.0)
    top_k: int = Field(10, ge=1, le=50)
    dedupe: bool = True

    rerank: bool = True
    lexical_weight: float = Field(0.30, ge=0.0, le=5.0)
    rerank_margin: float = Field(0.04, ge=-1.0, le=1.0)

    clinical_rerank: bool = True
    relation_rerank: bool = False
    relation_weight: float = Field(0.05, ge=0.0, le=5.0)
    relation_max_degree: int = Field(2000, ge=1)

    include_candidates: bool = False
    use_model_labels: bool = False
    extract_all: bool = True

    group_bias: Optional[Dict[str, float]] = None
    tui_bias: Optional[Dict[str, float]] = None


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)
    options: ExtractOptions = Field(default_factory=ExtractOptions)


class NoteIn(BaseModel):
    id: str
    name: str
    text: str


class ExtractBatchRequest(BaseModel):
    notes: List[NoteIn] = Field(..., min_length=1)
    options: ExtractOptions = Field(default_factory=ExtractOptions)


class ExtractResponse(BaseModel):
    extractions: List[Dict[str, Any]]
    meta: Dict[str, Any]


class ExtractBatchItem(BaseModel):
    id: str
    name: str
    extractions: List[Dict[str, Any]]
    meta: Dict[str, Any]


class ExtractBatchResponse(BaseModel):
    results: List[ExtractBatchItem]
    meta: Dict[str, Any]


app = FastAPI(title="trm-umls api", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_env_list("TRM_UI_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"),
    allow_methods=["*"],
    allow_headers=["*"],
)

_PIPE: Optional[TRMUMLSPipeline] = None


@app.on_event("startup")
def _startup() -> None:
    global _PIPE
    checkpoint = os.getenv("TRM_CHECKPOINT", "trm_umls/checkpoints/model.pt")
    device = os.getenv("TRM_DEVICE", "cuda")
    _PIPE = TRMUMLSPipeline.load(checkpoint, device=device)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "loaded": _PIPE is not None}


def _require_pipe() -> TRMUMLSPipeline:
    if _PIPE is None:
        raise RuntimeError("Pipeline not loaded yet.")
    return _PIPE


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    pipe = _require_pipe()
    t0 = time.time()
    exts = pipe.extract(
        req.text,
        threshold=req.options.threshold,
        extract_all=req.options.extract_all,
        use_model_labels=req.options.use_model_labels,
        dedupe=req.options.dedupe,
        top_k=req.options.top_k,
        rerank=req.options.rerank,
        lexical_weight=req.options.lexical_weight,
        rerank_margin=req.options.rerank_margin,
        relation_rerank=req.options.relation_rerank,
        relation_weight=req.options.relation_weight,
        relation_max_degree=req.options.relation_max_degree,
        include_candidates=req.options.include_candidates,
        clinical_rerank=req.options.clinical_rerank,
        group_bias=req.options.group_bias,
        tui_bias=req.options.tui_bias,
    )
    dt_ms = int((time.time() - t0) * 1000.0)
    return ExtractResponse(
        extractions=[asdict(e) for e in exts],
        meta={"ms": dt_ms, "count": len(exts)},
    )


@app.post("/extract_batch", response_model=ExtractBatchResponse)
def extract_batch(req: ExtractBatchRequest) -> ExtractBatchResponse:
    pipe = _require_pipe()
    t0 = time.time()
    out: List[ExtractBatchItem] = []
    for note in req.notes:
        t_note = time.time()
        exts = pipe.extract(
            note.text,
            threshold=req.options.threshold,
            extract_all=req.options.extract_all,
            use_model_labels=req.options.use_model_labels,
            dedupe=req.options.dedupe,
            top_k=req.options.top_k,
            rerank=req.options.rerank,
            lexical_weight=req.options.lexical_weight,
            rerank_margin=req.options.rerank_margin,
            relation_rerank=req.options.relation_rerank,
            relation_weight=req.options.relation_weight,
            relation_max_degree=req.options.relation_max_degree,
            include_candidates=req.options.include_candidates,
            clinical_rerank=req.options.clinical_rerank,
            group_bias=req.options.group_bias,
            tui_bias=req.options.tui_bias,
        )
        out.append(
            ExtractBatchItem(
                id=note.id,
                name=note.name,
                extractions=[asdict(e) for e in exts],
                meta={"ms": int((time.time() - t_note) * 1000.0), "count": len(exts)},
            )
        )
    return ExtractBatchResponse(
        results=out,
        meta={"ms": int((time.time() - t0) * 1000.0), "notes": len(out)},
    )


def main() -> None:
    import uvicorn

    host = os.getenv("TRM_HOST", "127.0.0.1")
    port = int(os.getenv("TRM_PORT", "8000"))
    uvicorn.run("trm_umls.api:app", host=host, port=port)


if __name__ == "__main__":
    main()

