from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/sql-scenarios", tags=["sql-scenarios"])


@router.post("/", response_model=schemas.SqlScenarioOut, status_code=status.HTTP_201_CREATED)
@router.post("", response_model=schemas.SqlScenarioOut, status_code=status.HTTP_201_CREATED)
def create_sql_scenario(payload: schemas.SqlScenarioCreate, db: Session = Depends(get_db)):
    sql_scenario = models.SqlScenario(**payload.model_dump())
    db.add(sql_scenario)
    db.commit()
    db.refresh(sql_scenario)
    return sql_scenario


@router.get("/", response_model=list[schemas.SqlScenarioOut])
@router.get("", response_model=list[schemas.SqlScenarioOut])
def list_sql_scenarios(db: Session = Depends(get_db)):
    return db.query(models.SqlScenario).all()


@router.get("/{sql_scenario_id}", response_model=schemas.SqlScenarioOut)
def get_sql_scenario(sql_scenario_id: int, db: Session = Depends(get_db)):
    sql_scenario = db.get(models.SqlScenario, sql_scenario_id)
    if not sql_scenario:
        raise HTTPException(status_code=404, detail="SQL scenario not found")
    return sql_scenario
