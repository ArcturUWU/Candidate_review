import json
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import SessionLocal, get_db
from ..services import sandbox, web_search
from ..services.lm_client import lm_client
from ..services.rag import search_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])

# Tools exposed to the model (LM Studio)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Поиск по загруженной документации сценария.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Поиск в интернете для валидации ответа кандидата.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_task",
            "description": "Поставить баллы за задание кандидату.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "points": {"type": "number"},
                    "comment": {"type": "string"},
                },
                "required": ["task_id", "points"],
            },
        },
    },
]


def _get_task_by_id(scenario: models.Scenario, task_id: str) -> Optional[dict[str, Any]]:
    for task in scenario.tasks or []:
        if task.get("id") == task_id:
            return task
    return None


def _build_system_prompt(session: models.Session, rag_available: bool) -> str:
    """Construct a strict instruction block for the model."""
    scenario = session.scenario
    role = session.role
    tasks_descr = "\n".join(
        [
            f"- {t.get('id')}: {t.get('type')} {t.get('title')} (max {t.get('max_points', 'n/a')})"
            for t in scenario.tasks or []
        ]
    )
    tool_hint = (
        "rag_search для материалов сценария, web_search для общих фактов."
        if rag_available
        else "документов нет — НЕ вызывай rag_search; для валидации используй знания и web_search."
    )
    return (
    "<SYSTEM>\n"
    "Ты — AI-интервьюер/оркестратор. Тебе поручено вести собеседование с кандидатом на определенную роль и по определенному сценарию. Также есть и сложность сценария. "
    "Работай только в рамках переданных ролей, задач и контекста.\n"
    f"Контекст: роль {role.name} ({role.slug}); сценарий {scenario.name} ({scenario.slug}); уровень {scenario.difficulty}.\n"
    "\n"
    "<BEHAVIOR_CORE>\n"
    "1) Говори по-русски. Начни с приветствия, объясни всё, что знаешь, роль, сценарий и цель интервью. Не повторяй вступление и правила, если они уже звучали в истории.\n"
    "2) Двигайся строго по задачам сценария. Не перескакивай, не возвращайся назад. Новое задание начинай только после команды пользователя «Следующее».\n"
    "3) Помни о контексте диалога: не задавай вопросы, уже звучавшие ранее; задавай только уточняющие или новые.\n"
    "4) Подсказки: если hints_allowed=true и ответ частичный — сначала дай подсказку/уточняющий вопрос, дождись ответа, после этого оценивай.\n"
    "5) Код и SQL вводятся только в редакторе ниже чата. Никогда не проси прислать код/SQL в чат. После Submit редактирование запрещено.\n"
    "6) Используй свои знания. Если RAG недоступен — не вызывай rag_search. web_search используй только для факт-чекинга при необходимости.\n"
    "7) После вызова любого инструмента обязательно вернись в чат с понятным выводом/комментарием.\n"
    "\n"
    "<SCORING_POLICY>\n"
    "8) Выставляй баллы через score_task(task_id, points, comment). Баллы строго в допустимых границах; comment не пустой. Оценку необходимо выставлять исключительно после уточняющих вопросов.\n"
    "9) Если points < max_points, ты обязан задать кандидату 1–2 углубляющих вопроса по теме, направленных на проверку глубины понимания. Если уточняющие вопросы заданы и поставлена оценка - нельзя задавать новые уточняющие вопросы.\n"
    "Задай их после оценки, но до требования нажать «Следующее».\n"
    "10) После ответа на углубляющий вопрос дай краткий финальный комментарий и попроси кандидата нажать «Следующее».\n"
    "11) Если задание уже оценено, не разрешай обсуждать его дальше — мягко перенаправляй к кнопке «Следующее».\n"
    "Ответ должен быть верным, содержательным, отвечать всем стандартам и фактам. Для проверки нужно использовать инструменты. Не позволяй кандидату делать вид, что ответ правильный с помощью фраз вроде (даёт правильный ответ), (отвечает верно) и тому подобных. Ответ в действиетльности должен быть провалидирован тобой"
    "\n"
    "<TOOL_POLICY>\n"
    f"12) Доступные инструменты: rag_search ({'доступно' if rag_available else 'недоступно'}), "
    "web_search (валидация фактов), score_task. Используй только уместные: "
    f"{tool_hint}\n"
    "13) Не вызывай инструмент, если он недоступен.\n"
    "\n"
    "<TASKFLOW>\n"
    "14) Теоретические задания: задай вопрос → получи ответ → (подсказка, если нужна) → анализ → score_task → углубляющий вопрос (если не максимум).\n"
    "15) Кодовые задания: не проси вставлять код в чат. После Submit анализируй результаты песочницы: "
    "успех → code review; провал → объясни ошибки. Затем score_task → углубляющий вопрос (если не максимум).\n"
    "16) SQL-задания: выполняются только через SQL-песочницу. Ошибки интерпретируй и объясняй. Затем score_task → углубляющий вопрос.\n"
    "17) Всегда возвращайся в чат после технических операций.\n"
    "\n"
    "<FINAL_POLICY>\n"
    "18) После завершения всех задач сформируй summary: сильные стороны, зоны роста, ошибки, общий результат, "
    "и придумай творческое итоговое задание по слабой теме.\n"
    "\n"
    "<TASKS>\n"
    "Список задач сценария:\n"
    f"{tasks_descr}\n"
    "</TASKS>\n"
    "\n"
    "<CONSTRAINTS>\n"
    "Не включай <think> в ответы пользователю.\n"
    "Если вступление уже было — не повторяй.\n"
    "</CONSTRAINTS>\n"
    "</SYSTEM>"
)


def _strip_think(content: Optional[str]) -> str:
    if not content:
        return ""
    if "</think>" in content:
        return content.split("</think>", 1)[1].strip()
    return content.replace("<think>", "").strip()


def _strip_intro(text: str, intro_done: bool) -> str:
    """Cut repetitive greetings when intro already done."""
    if not intro_done or not text:
        return text
    intro_patterns = [
        "привет", "добрый день", "здравствуйте", "я проведу собеседование", "формат состоит",
        "сегодня мы пройдём", "мы проведём", "давайте приступим", "начнём с теории",
    ]
    lowered = text.lower()
    for pat in intro_patterns:
        if lowered.startswith(pat):
            # Remove first sentence
            parts = text.split("\n", 1)
            return parts[1] if len(parts) > 1 else ""
    return text


def _analyze_candidate_message(text: str) -> list[str]:
    """Detect placeholder/meta/roleplay/offtopic/empty/code_in_chat/sql_in_chat flags."""
    flags: list[str] = []
    t = text.strip().lower()
    if not t:
        flags.append("empty")
    placeholder_phrases = [
        "(отвечает правильно)",
        "(правильный ответ)",
        "код верный",
        "решение корректное",
        "ответ правильный",
        "(пишет правильный код)",
        "(solution)",
    ]
    if any(p in t for p in placeholder_phrases):
        flags.append("placeholder")
    if len(t) < 40 and ("регресс" not in t and "join" not in t and "select" not in t):
        flags.append("too_short")
    roleplay = ["представим", "я бот", "как модель", "как ассистент", "роль"]
    if any(p in t for p in roleplay):
        flags.append("roleplay")
    if "def " in t or "print(" in t or "import " in t:
        flags.append("code_in_chat")
    if "select " in t or "from " in t:
        flags.append("sql_in_chat")
    return flags


def _control_state(session: models.Session, history: list[models.Message]) -> dict[str, Any]:
    intro_done = any(m.sender == "model" for m in history)
    scores = session.scores or {}
    task_status = {tid: "scored" for tid in scores.keys()}
    current_task = session.current_task_id or (session.scenario.tasks[0]["id"] if session.scenario.tasks else "нет")
    awaiting_next = current_task in task_status
    return {
        "intro_done": intro_done,
        "current_task_id": current_task,
        "task_status": task_status,
        "hint_count": {},
        "awaiting_next_click": awaiting_next,
        "code_submitted": {},
        "sql_submitted": {},
    }


def _semantic_memory(session: models.Session) -> dict[str, Any]:
    """Derive simple strengths/weaknesses from scores."""
    strengths: set[str] = set()
    weaknesses: set[str] = set()
    issues: list[dict[str, str]] = []
    scores = session.scores or {}
    for task in session.scenario.tasks or []:
        tid = task.get("id")
        if not tid or tid not in scores:
            continue
        pts = scores[tid]
        max_pts = task.get("max_points") or 1
        ratio = float(pts) / float(max_pts)
        topics = task.get("related_topics") or []
        if ratio >= 0.8:
            strengths.update(topics)
        elif ratio <= 0.5:
            weaknesses.update(topics)
            for t in topics:
                issues.append({"key": f"weak_{t}", "text": f"Низкий балл по теме {t}"})
    return {
        "strengths": list(strengths),
        "weaknesses": list(weaknesses),
        "issues": issues,
    }


def _episodic_memory(history: list[models.Message]) -> list[str]:
    events: list[str] = []
    for m in history[-60:]:
        if m.sender == "tool":
            events.append(f"tool:{m.text[:120]}")
        elif m.sender == "system" and "result" in m.text:
            events.append(f"system:{m.text[:120]}")
    return events[-30:]


def _convert_history(messages: list[models.Message]) -> list[dict[str, Any]]:
    converted = []
    for msg in messages:
        if msg.sender == "candidate":
            role = "user"
        elif msg.sender == "model":
            role = "assistant"
        else:
            role = "system"
        converted.append({"role": role, "content": msg.text})
    return converted


def _conversation_snapshot(session: models.Session, history: list[models.Message]) -> str:
    """Short, explicit state for the model to avoid repetition."""
    control = _control_state(session, history)
    sem = _semantic_memory(session)
    episodic = _episodic_memory(history)
    last_user = next((m for m in reversed(history) if m.sender == "candidate"), None)
    last_user_text = (last_user.text if last_user else "нет последних вопросов")[:200]
    last_model = next((m for m in reversed(history) if m.sender == "model"), None)
    last_model_text = (last_model.text if last_model else "нет")[:200]
    return (
        "<CONTROL_STATE>"
        f"<INTRO_DONE>{control['intro_done']}</INTRO_DONE>"
        f"<CURRENT_TASK_ID>{control['current_task_id']}</CURRENT_TASK_ID>"
        f"<AWAITING_NEXT_CLICK>{control['awaiting_next_click']}</AWAITING_NEXT_CLICK>"
        f"<TASK_STATUS>{json.dumps(control['task_status'], ensure_ascii=False)}</TASK_STATUS>"
        f"<HINT_COUNT>{json.dumps(control['hint_count'], ensure_ascii=False)}</HINT_COUNT>"
        f"<CODE_SUBMITTED>{json.dumps(control['code_submitted'], ensure_ascii=False)}</CODE_SUBMITTED>"
        f"<SQL_SUBMITTED>{json.dumps(control['sql_submitted'], ensure_ascii=False)}</SQL_SUBMITTED>"
        "</CONTROL_STATE>"
        "<SEMANTIC_MEMORY>"
        f"<STRENGTHS>{', '.join(sem.get('strengths', []))}</STRENGTHS>"
        f"<WEAKNESSES>{', '.join(sem.get('weaknesses', []))}</WEAKNESSES>"
        f"<ISSUES>{json.dumps(sem.get('issues', []), ensure_ascii=False)}</ISSUES>"
        "</SEMANTIC_MEMORY>"
        "<EPISODIC_MEMORY>"
        f"{json.dumps(episodic, ensure_ascii=False)}"
        "</EPISODIC_MEMORY>"
        f"<LAST_USER>{last_user_text}</LAST_USER>"
        f"<LAST_MODEL>{last_model_text}</LAST_MODEL>"
        "Не повторяй уже сказанное; продолжай диалог логично и не начинай новую задачу без явного перехода."
    )


def _apply_score(session: models.Session, args: dict[str, Any], db: Session) -> dict[str, Any]:
    task_id = args.get("task_id")
    points = float(args.get("points", 0))
    comment = args.get("comment")
    task = _get_task_by_id(session.scenario, task_id)
    if not task:
        return {"error": f"Task {task_id} not found in scenario"}
    max_points = task.get("max_points", 0)
    if points < 0 or points > max_points:
        return {"error": f"Points should be within [0, {max_points}]"}
    score = models.Score(session_id=session.id, task_id=task_id, points=points, comment=comment)
    current_scores = session.scores or {}
    session.scores = {**current_scores, task_id: points}
    db.add(score)
    db.commit()
    db.refresh(score)
    return {"ok": True, "task_id": task_id, "points": points, "comment": comment}


def _dispatch_tool_call(session: models.Session, tool_call: dict[str, Any], db: Session) -> dict[str, Any]:
    name = tool_call["function"]["name"]
    try:
        args = json.loads(tool_call["function"].get("arguments", "{}"))
    except json.JSONDecodeError:
        args = {}
    if name == "rag_search":
        if not session.scenario.rag_corpus_id:
            return {"error": "No RAG corpus configured for this scenario. Use web_search instead."}
        docs = db.query(models.Document).filter_by(rag_corpus_id=session.scenario.rag_corpus_id).all()
        if not docs:
            return {"error": "No RAG documents available. Use web_search instead."}
        doc_dicts = [{"id": d.id, "filename": d.filename, "content": d.content} for d in docs]
        results = search_documents(doc_dicts, args.get("query", ""), args.get("top_k", 3))
        return {"results": [r.model_dump() for r in results]}
    if name == "web_search":
        return {"results": web_search.web_search(args.get("query", ""), args.get("top_k", 3))}
    if name == "score_task":
        return _apply_score(session, args, db)
    return {"error": f"Unsupported tool {name}"}


def _score_feedback(result: dict[str, Any]) -> str:
    task_id = result.get("task_id") or ""
    pts = result.get("points")
    comment = result.get("comment") or ""
    pts_txt = f"{pts} балл(ов)" if pts is not None else "оценка выставлена"
    return f"Оценка сохранена: {pts_txt} за {task_id}. Комментарий: {comment}. Нажмите «Следующее», чтобы перейти далее."


@router.post("/", response_model=schemas.SessionOut, status_code=status.HTTP_201_CREATED)
@router.post("", response_model=schemas.SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(payload: schemas.SessionCreate, db: Session = Depends(get_db)):
    scenario = db.get(models.Scenario, payload.scenario_id)
    role = db.get(models.Role, payload.role_id)
    if not scenario or not role:
        raise HTTPException(status_code=400, detail="Scenario or role not found")
    if scenario.role_id != role.id:
        raise HTTPException(status_code=400, detail="Scenario does not belong to the selected role")
    session = models.Session(
        scenario_id=payload.scenario_id,
        role_id=payload.role_id,
        candidate_id=payload.candidate_id,
        state="active",
        current_task_id=None,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/{session_id}", response_model=schemas.SessionOut)
def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.get(models.Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/{session_id}/messages", response_model=list[schemas.MessageOut])
def list_messages(session_id: str, db: Session = Depends(get_db)):
    session = db.get(models.Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return db.query(models.Message).filter_by(session_id=session_id).order_by(models.Message.created_at).all()


@router.post("/{session_id}/messages", response_model=schemas.MessageOut)
def post_message(session_id: str, payload: schemas.MessageCreate, db: Session = Depends(get_db)):
    session = db.get(models.Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    message = models.Message(session_id=session_id, **payload.model_dump())
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


@router.post("/{session_id}/score", response_model=schemas.ScoreOut)
def score_task(session_id: str, payload: schemas.ScoreCreate, db: Session = Depends(get_db)):
    session = db.get(models.Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    scenario = session.scenario
    task = _get_task_by_id(scenario, payload.task_id)
    if not task:
        raise HTTPException(status_code=400, detail="Task not found in scenario")
    max_points = task.get("max_points", 0)
    if payload.points < 0 or payload.points > max_points:
        raise HTTPException(
            status_code=400,
            detail=f"Points should be within [0, {max_points}]",
        )
    score = models.Score(session_id=session_id, **payload.model_dump())
    current_scores = session.scores or {}
    session.scores = {**current_scores, payload.task_id: payload.points}
    db.add(score)
    db.commit()
    db.refresh(score)
    return score


@router.post("/{session_id}/tasks/{task_id}/submit_code")
def submit_code(session_id: str, task_id: str, payload: schemas.CodeSubmission, db: Session = Depends(get_db)):
    session = db.get(models.Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    task = _get_task_by_id(session.scenario, task_id)
    if not task or task.get("type") != "coding":
        raise HTTPException(status_code=400, detail="Task is not a coding task")
    result = sandbox.run_code(payload.language, payload.code, payload.tests_id)
    system_msg = models.Message(
        session_id=session_id,
        sender="system",
        text=f"Code execution result for {task_id}: {result}",
        task_id=task_id,
    )
    db.add(system_msg)
    db.commit()
    return {"task_id": task_id, "result": result}


@router.post("/{session_id}/tasks/{task_id}/submit_sql")
def submit_sql(session_id: str, task_id: str, payload: schemas.SqlSubmission, db: Session = Depends(get_db)):
    session = db.get(models.Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    task = _get_task_by_id(session.scenario, task_id)
    if not task or task.get("type") != "sql":
        raise HTTPException(status_code=400, detail="Task is not a SQL task")
    result = sandbox.run_sql(payload.sql_scenario_id, payload.query)
    system_msg = models.Message(
        session_id=session_id,
        sender="system",
        text=f"SQL execution result for {task_id}: {result}",
        task_id=task_id,
    )
    db.add(system_msg)
    db.commit()
    return {"task_id": task_id, "result": result}


@router.post("/{session_id}/complete")
def complete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.get(models.Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.state = "completed"
    session.finished_at = datetime.utcnow()
    db.commit()
    return {"status": "ok"}


@router.post("/{session_id}/web-search")
def run_web_search(session_id: str, payload: schemas.WebSearchRequest, db: Session = Depends(get_db)):
    if not db.get(models.Session, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    results = web_search.web_search(payload.query, payload.top_k)
    return {"results": results}


@router.post("/{session_id}/lm/chat")
def call_model(session_id: str, db: Session = Depends(get_db)):
    """Non-streaming call (fallback)."""
    session = db.get(models.Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    history_db = (
        db.query(models.Message)
        .filter_by(session_id=session_id)
        .order_by(models.Message.created_at)
        .all()
    )
    rag_available = False
    if session.scenario.rag_corpus_id:
        rag_available = db.query(models.Document).filter_by(rag_corpus_id=session.scenario.rag_corpus_id).count() > 0
    system_prompt = _build_system_prompt(session, rag_available)
    snapshot = _conversation_snapshot(session, history_db)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": snapshot},
    ]
    messages.extend(_convert_history(history_db))

    try:
        first_resp = lm_client.chat(messages, tools=TOOLS)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"LM request failed: {exc}") from exc

    assistant_msg = first_resp["choices"][0]["message"]
    tool_calls = assistant_msg.get("tool_calls")
    messages.append(assistant_msg)

    tool_results_db: list[models.Message] = []
    last_score_result: dict[str, Any] | None = None
    final_msg = assistant_msg
    if tool_calls:
        tool_messages = []
        for tc in tool_calls:
            result = _dispatch_tool_call(session, tc, db)
            if tc["function"]["name"] == "score_task":
                last_score_result = result
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            tool_results_db.append(
                models.Message(
                    session_id=session_id,
                    sender="tool",
                    text=f"{tc['function']['name']} -> {result}",
                    task_id=tc["function"].get("arguments", None),
                )
            )
        messages.extend(tool_messages)
        try:
            second_resp = lm_client.chat(messages, tools=TOOLS)
            final_msg = second_resp["choices"][0]["message"]
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"LM request failed after tool calls: {exc}") from exc
        if (not final_msg.get("content")) and last_score_result:
            final_msg["content"] = _score_feedback(last_score_result)

    for tm in tool_results_db:
        db.add(tm)
    db.add(
        models.Message(
            session_id=session_id,
            sender="model",
            text=final_msg.get("content") or "",
        )
    )
    db.commit()

    return {"message": final_msg}


@router.get("/{session_id}/lm/chat-stream")
def stream_model(session_id: str):
    """Stream tokens from LM Studio. Runs tool calls first, then streams/returns final answer."""
    base_db = SessionLocal()
    session = base_db.get(models.Session, session_id)
    if not session:
        base_db.close()
        raise HTTPException(status_code=404, detail="Session not found")
    history_db = (
        base_db.query(models.Message)
        .filter_by(session_id=session_id)
        .order_by(models.Message.created_at)
        .all()
    )
    # Pre-validate last candidate message for placeholders/offtopic
    last_msg = history_db[-1] if history_db else None
    if last_msg and last_msg.sender == "candidate":
        flags = _analyze_candidate_message(last_msg.text)
        if flags:
            warn = "Ответ не принят: дайте содержательный ответ по сути вопроса."
            if "code_in_chat" in flags or "sql_in_chat" in flags:
                warn = "Не вставляйте код/SQL в чат. Введите решение в редактор ниже и нажмите Submit."
            base_db.add(models.Message(session_id=session_id, sender="system", text=warn))
            base_db.commit()
            base_db.close()

            def reject_stream():
                yield "data: " + json.dumps({"type": "token", "content": warn}, ensure_ascii=False) + "\n\n"
                yield "data: " + json.dumps({"type": "done", "content": warn}, ensure_ascii=False) + "\n\n"

            return StreamingResponse(reject_stream(), media_type="text/event-stream")

    rag_available = False
    if session.scenario.rag_corpus_id:
        rag_available = (
            base_db.query(models.Document).filter_by(rag_corpus_id=session.scenario.rag_corpus_id).count() > 0
        )
    system_prompt = _build_system_prompt(session, rag_available)
    snapshot = _conversation_snapshot(session, history_db)
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": snapshot},
    ]
    base_messages.extend(_convert_history(history_db))

    try:
        first_resp = lm_client.chat(base_messages, tools=TOOLS)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LM request failed before streaming")
        base_db.close()
        raise HTTPException(status_code=500, detail=f"LM request failed: {exc}") from exc

    assistant_msg = first_resp["choices"][0]["message"]
    tool_calls = assistant_msg.get("tool_calls")

    stream_messages = list(base_messages)
    tool_results_payload: list[dict[str, Any]] = []
    status_events: list[str] = []

    score_result_payload: dict[str, Any] | None = None
    if tool_calls:
        stream_messages.append(assistant_msg)
        for tc in tool_calls:
            fname = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            if fname == "web_search":
                status_text = f"Ищем в интернете: {args.get('query', '')}"
                base_db.add(models.Message(session_id=session_id, sender="system", text=status_text))
                base_db.commit()
                status_events.append(status_text)

            result = _dispatch_tool_call(session, tc, base_db)
            if fname == "score_task":
                score_result_payload = result
            tool_results_payload.append(
                {
                    "sender": "tool",
                    "text": f"{tc['function']['name']} -> {result}",
                    "task_id": tc["function"].get("arguments", None),
                    "name": tc["function"]["name"],
                    "result": result,
                }
            )
            stream_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
    else:
        stream_messages = base_messages

    base_db.close()

    def event_stream():
        local_db = SessionLocal()
        control_state = _control_state(session, history_db)
        final_chunks: list[str] = []
        hidden_buffer = ""
        revealed = False
        saw_think = False
        fallback_text = _strip_think(assistant_msg.get("content"))
        # If the model only called score_task and stayed silent, prepare a minimal feedback
        if not fallback_text:
            score_calls = [t for t in tool_results_payload if t.get("name") == "score_task"]
            if score_calls:
                res = score_calls[-1].get("result", {})
                fallback_text = _score_feedback(res)
        received_tokens = False
        final_text = ""
        try:
            for status_text in status_events:
                yield "data: " + json.dumps({"type": "token", "content": status_text}, ensure_ascii=False) + "\n\n"

            if tool_calls:
                try:
                    sync_resp = lm_client.chat(stream_messages, tools=[])
                    final_text = _strip_think(sync_resp["choices"][0]["message"].get("content"))
                except Exception:
                    final_text = fallback_text or ""
                if score_result_payload and (not final_text or final_text.strip() == fallback_text.strip()):
                    final_text = _score_feedback(score_result_payload)
                chunk_size = 120
                for i in range(0, len(final_text), chunk_size):
                    piece = final_text[i : i + chunk_size]
                    yield "data: " + json.dumps({"type": "token", "content": piece}, ensure_ascii=False) + "\n\n"
                    final_chunks.append(piece)
                final_text = "".join(final_chunks)
            else:
                for chunk in lm_client.stream_chat(stream_messages, tools=TOOLS):
                    if "<think>" in chunk:
                        saw_think = True
                    if not saw_think and not revealed:
                        revealed = True  # нет блока размышлений – стримим сразу
                    if saw_think and not revealed:
                        hidden_buffer += chunk
                        if "</think>" in hidden_buffer:
                            revealed = True
                            after = hidden_buffer.split("</think>", 1)[1]
                            hidden_buffer = ""
                            if after:
                                final_chunks.append(after)
                                yield "data: " + json.dumps({"type": "token", "content": after}, ensure_ascii=False) + "\n\n"
                                received_tokens = True
                        continue
                    final_chunks.append(chunk)
                    yield "data: " + json.dumps({"type": "token", "content": chunk}, ensure_ascii=False) + "\n\n"
                    received_tokens = True
                final_text = "".join(final_chunks)
                if not received_tokens and not final_text:
                    try:
                        sync_resp = lm_client.chat(stream_messages, tools=[])
                        final_text = _strip_think(sync_resp["choices"][0]["message"].get("content"))
                    except Exception:
                        final_text = fallback_text or ""

            for payload in tool_results_payload:
                msg = models.Message(
                    session_id=session_id,
                    sender=payload["sender"],
                    text=payload["text"],
                    task_id=payload.get("task_id"),
                )
                local_db.add(msg)

            if final_text:
                trimmed = _strip_intro(final_text, control_state.get("intro_done", False))
                local_db.add(models.Message(session_id=session_id, sender="model", text=trimmed))
            elif fallback_text:
                trimmed = _strip_intro(fallback_text, control_state.get("intro_done", False))
                local_db.add(models.Message(session_id=session_id, sender="model", text=trimmed))
                final_text = trimmed
            local_db.commit()
            yield "data: " + json.dumps({"type": "done", "content": final_text}, ensure_ascii=False) + "\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("LM streaming failed")
            local_db.add(
                models.Message(
                    session_id=session_id,
                    sender="system",
                    text=f"Ошибка сервиса LM Studio: {exc}",
                )
            )
            local_db.commit()
            yield "data: " + json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n\n"
            if fallback_text:
                yield "data: " + json.dumps({"type": "done", "content": fallback_text}, ensure_ascii=False) + "\n\n"
        finally:
            local_db.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
