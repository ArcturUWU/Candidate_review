from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..services.rag import search_documents

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/corpora", response_model=schemas.RagCorpusOut, status_code=status.HTTP_201_CREATED)
@router.post("/corpora/", response_model=schemas.RagCorpusOut, status_code=status.HTTP_201_CREATED)
def create_corpus(payload: schemas.RagCorpusCreate, db: Session = Depends(get_db)):
    corpus = models.RagCorpus(**payload.model_dump())
    db.add(corpus)
    db.commit()
    db.refresh(corpus)
    return corpus


@router.get("/corpora", response_model=list[schemas.RagCorpusOut])
@router.get("/corpora/", response_model=list[schemas.RagCorpusOut])
def list_corpora(db: Session = Depends(get_db)):
    return db.query(models.RagCorpus).all()


@router.get("/corpora/{corpus_id}", response_model=schemas.RagCorpusOut)
def get_corpus(corpus_id: int, db: Session = Depends(get_db)):
    corpus = db.get(models.RagCorpus, corpus_id)
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    return corpus


@router.post(
    "/corpora/{corpus_id}/documents",
    response_model=schemas.DocumentOut,
    status_code=status.HTTP_201_CREATED,
)
def add_document(corpus_id: int, payload: schemas.DocumentCreate, db: Session = Depends(get_db)):
    corpus = db.get(models.RagCorpus, corpus_id)
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    doc = models.Document(
        rag_corpus_id=corpus_id,
        filename=payload.filename,
        content=payload.content,
        meta=payload.metadata,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


@router.get("/corpora/{corpus_id}/documents", response_model=list[schemas.DocumentOut])
def list_documents(corpus_id: int, db: Session = Depends(get_db)):
    corpus = db.get(models.RagCorpus, corpus_id)
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    return db.query(models.Document).filter_by(rag_corpus_id=corpus_id).all()


@router.post("/search", response_model=list[schemas.RagSearchResult])
def rag_search(payload: schemas.RagSearchRequest, db: Session = Depends(get_db)):
    corpus = db.get(models.RagCorpus, payload.corpus_id)
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    docs = db.query(models.Document).filter_by(rag_corpus_id=payload.corpus_id).all()
    doc_dicts = [{"id": d.id, "filename": d.filename, "content": d.content} for d in docs]
    return search_documents(doc_dicts, payload.query, payload.top_k)
