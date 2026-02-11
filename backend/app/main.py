from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import models
from .config import settings
from .database import Base, SessionLocal, engine
from .routers import rag, roles, scenarios, sessions, sql_scenarios

# Create DB tables for a simple demo; for production prefer migrations
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Chat-Review for candidates", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.allow_origins.split(",") if origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(roles.router)
app.include_router(scenarios.router)
app.include_router(rag.router)
app.include_router(sql_scenarios.router)
app.include_router(sessions.router)


@app.get("/health")
def health():
    return {"status": "ok"}

frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
else:
    @app.get("/")
    def root_fallback():
        return {"message": "Frontend build not found. Run `npm run build` in frontend/."}


def seed_defaults() -> None:
    """Insert demo roles/scenarios if DB is empty to avoid 404 on first run."""
    db = SessionLocal()
    try:
        if db.query(models.Role).count() > 0:
            return

        ds = models.Role(name="Data Scientist", slug="ds", description="ML, эксперименты, метрики")
        be = models.Role(name="Backend", slug="backend", description="API, очереди, устойчивость")
        de = models.Role(name="Data Engineer", slug="de", description="ETL, SQL, пайплайны")
        db.add_all([ds, be, de])
        db.flush()

        scenarios_payload = [
            models.Scenario(
                role_id=ds.id,
                name="DS — Junior ML",
                slug="ds-junior-ml",
                description="Регрессия, классификация, SQL основы",
                difficulty="junior",
                tasks=[
                    {
                        "id": "T1",
                        "type": "theory",
                        "title": "Основы регрессии",
                        "description": "Различия L1/L2: sparsity, устойчивость, геометрия штрафа.",
                        "max_points": 5,
                        "hints_allowed": True,
                        "evaluation_criteria": {
                            "full_answer": "Геометрия, sparsity, устойчивость",
                            "partial_answer": "Общее различие без деталей",
                        },
                        "related_topics": ["regularization", "linear_models"],
                    },
                    {
                        "id": "C1",
                        "type": "coding",
                        "language": "python",
                        "title": "Логистическая регрессия",
                        "description_for_candidate": "Реализуйте логистическую регрессию без sklearn.",
                        "max_points": 10,
                        "tests_id": "logreg_basic",
                        "related_topics": ["gradient_descent", "classification"],
                    },
                    {
                        "id": "SQL1",
                        "type": "sql",
                        "title": "Агрегация заказов",
                        "description_for_candidate": "По таблицам orders и customers посчитайте сумму заказов по городам.",
                        "max_points": 8,
                        "sql_scenario_id": "ecommerce_basic",
                        "related_topics": ["joins", "aggregation"],
                    },
                ],
            ),
            models.Scenario(
                role_id=ds.id,
                name="DS — Product ML",
                slug="ds-product-ml",
                description="A/B, метрики продукта, рекомендации",
                difficulty="middle",
                tasks=[
                    {
                        "id": "T-metrics",
                        "type": "theory",
                        "title": "Метрики A/B",
                        "max_points": 5,
                        "hints_allowed": True,
                        "related_topics": ["experimentation", "metrics"],
                    },
                    {
                        "id": "SQL-ab",
                        "type": "sql",
                        "title": "Конверсия по когорте",
                        "description_for_candidate": "Напишите запрос конверсии по дню регистрации.",
                        "sql_scenario_id": "ab_product",
                        "max_points": 8,
                        "related_topics": ["joins", "aggregation"],
                    },
                ],
            ),
            models.Scenario(
                role_id=be.id,
                name="Backend — REST",
                slug="be-rest",
                description="API дизайн, идемпотентность, очереди",
                difficulty="middle",
                tasks=[
                    {"id": "T-REST", "type": "theory", "title": "PUT vs PATCH идемпотентность", "max_points": 5},
                    {
                        "id": "C-BE",
                        "type": "coding",
                        "language": "python",
                        "title": "Очередь задач",
                        "description_for_candidate": "Реализуйте очередь с ack/nack.",
                        "tests_id": "queue_basic",
                        "max_points": 8,
                    },
                ],
            ),
            models.Scenario(
                role_id=be.id,
                name="Backend — Resilience",
                slug="be-resilience",
                description="Ретраи, троттлинг, circuit breaker",
                difficulty="senior",
                tasks=[
                    {"id": "T-circuit", "type": "theory", "title": "Circuit breaker", "max_points": 6, "hints_allowed": True},
                    {
                        "id": "C-rate",
                        "type": "coding",
                        "language": "python",
                        "title": "Rate limiter",
                        "description_for_candidate": "Сделайте токен-бакет.",
                        "tests_id": "rate_limiter",
                        "max_points": 9,
                    },
                ],
            ),
            models.Scenario(
                role_id=de.id,
                name="DE — Pipelines",
                slug="de-pipelines",
                description="Инкрементальные пайплайны, буферизация, SLA",
                difficulty="middle",
                tasks=[
                    {"id": "T-de-incr", "type": "theory", "title": "Инкрементальные загрузки", "max_points": 5, "hints_allowed": True},
                    {
                        "id": "SQL-de-agg",
                        "type": "sql",
                        "title": "Агрегация событий",
                        "description_for_candidate": "Посчитайте DAU по регионам из таблицы events.",
                        "sql_scenario_id": "events_basic",
                        "max_points": 8,
                    },
                ],
            ),
            models.Scenario(
                role_id=de.id,
                name="DE — Warehousing",
                slug="de-warehousing",
                description="Моделирование данных, SCD, оркестрация",
                difficulty="senior",
                tasks=[
                    {"id": "T-scd", "type": "theory", "title": "SCD типы", "max_points": 6},
                    {
                        "id": "SQL-scd",
                        "type": "sql",
                        "title": "SCD Type 2 обновление",
                        "description_for_candidate": "Напишите SQL, который добавляет новую версию записи клиента.",
                        "sql_scenario_id": "scd_customers",
                        "max_points": 9,
                    },
                ],
            ),
        ]
        db.add_all(scenarios_payload)
        db.commit()
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    seed_defaults()


@app.get("/lm/ping")
def lm_ping():
    """Check connectivity to LM Studio."""
    try:
        resp = lm_client.ping()
        return {"status": "ok", "model": resp.get("model", settings.lm_model)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"LM Studio not reachable: {exc}") from exc
