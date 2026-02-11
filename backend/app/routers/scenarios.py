from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


@router.get("/", response_model=list[schemas.ScenarioOut])
@router.get("", response_model=list[schemas.ScenarioOut])
def list_scenarios(db: Session = Depends(get_db)):
    return db.query(models.Scenario).all()


@router.post("/", response_model=schemas.ScenarioOut, status_code=status.HTTP_201_CREATED)
@router.post("", response_model=schemas.ScenarioOut, status_code=status.HTTP_201_CREATED)
def create_scenario(payload: schemas.ScenarioCreate, db: Session = Depends(get_db)):
    role = db.get(models.Role, payload.role_id)
    if not role:
        raise HTTPException(status_code=400, detail="Role does not exist")
    existing = (
        db.query(models.Scenario)
        .filter(models.Scenario.role_id == payload.role_id, models.Scenario.slug == payload.slug)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Scenario slug already used for this role")
    scenario = models.Scenario(**payload.model_dump())
    db.add(scenario)
    db.commit()
    db.refresh(scenario)
    return scenario


@router.get("/{scenario_id}", response_model=schemas.ScenarioOut)
def get_scenario(scenario_id: int, db: Session = Depends(get_db)):
    scenario = db.get(models.Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario


@router.put("/{scenario_id}", response_model=schemas.ScenarioOut)
def update_scenario(scenario_id: int, payload: schemas.ScenarioUpdate, db: Session = Depends(get_db)):
    scenario = db.get(models.Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    data = payload.model_dump(exclude_unset=True)
    if "role_id" in data:
        new_role = db.get(models.Role, data["role_id"])
        if not new_role:
            raise HTTPException(status_code=400, detail="Role does not exist")
    for key, value in data.items():
        setattr(scenario, key, value)
    db.commit()
    db.refresh(scenario)
    return scenario


@router.delete("/{scenario_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scenario(scenario_id: int, db: Session = Depends(get_db)):
    scenario = db.get(models.Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    db.delete(scenario)
    db.commit()
    return None
