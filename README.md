# Chat-Review для кандидатов

Минимальный монорепозиторий с backend (FastAPI) и frontend (React + Vite) для проведения собеседований в формате диалога с моделью, кодовыми и SQL‑заданиями, RAG и системой оценок.

## Быстрый старт

### Backend
1. `cd backend`
2. `python -m venv .venv && .venv\\Scripts\\activate` (Windows) или `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `uvicorn app.main:app --reload --port 8000`

Переменные окружения: `.env` (опционально)
```
DATABASE_URL=sqlite:///./reviewer.db
LM_STUDIO_URL=http://localhost:1234/v1/chat/completions
LM_MODEL=qwen/qwen3-8b
SANDBOX_CODE_URL=http://localhost:8001/run_code
SANDBOX_SQL_URL=http://localhost:8002/run_sql
WEB_SEARCH_URL=http://localhost:8003/search
ALLOW_ORIGINS=*
```

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run build`
4. (опционально dev) `npm run dev -- --host 127.0.0.1 --port 5173`

Собранный фронт (`frontend/dist`) автоматически раздаётся FastAPI: интерфейс доступен на `http://127.0.0.1:8000/`. `VITE_API_URL` можно указать, если backend запущен не на том же хосте.

## Что реализовано
- **Backend (FastAPI, SQLite через SQLAlchemy)**: сущности Role, Scenario, Session, Message, Score, RagCorpus/Document, SqlScenario; CRUD для ролей, сценариев, RAG‑корпусов и документов; эндпоинты сессии/чат, сабмиты кода и SQL, валидация score_task; заглушки для `rag_search`, `web_search`, LM Studio и песочниц.
- **Интеграция LM Studio**: прямые вызовы HTTP API LM Studio с моделью `qwen/qwen3-8b`, поддержка tool calls (`rag_search`, `web_search`, `score_task`) и запись результатов в БД.
- **Стриминг ответа модели**: `/sessions/{id}/lm/chat-stream` отдает `text/event-stream` с токенами; фронт отображает поток.
- **Песочницы**: вызовы `run_code`/`run_sql` вынесены в сервисы и ждут реальный executor на `SANDBOX_CODE_URL`/`SANDBOX_SQL_URL`.
- **Frontend (React + TS)**: экран выбора роли и сценария, чат, редакторы кода и SQL, базовая админка для ролей/сценариев. Поддерживается работа с API backend или локальные демо-данные; готов для отдачи через FastAPI на `:8000/`.
- **Архитектура RAG**: загрузка документов в корпус, поиск по простому косинусному сходству токенов.

## Основные эндпоинты
- `POST /sessions` — создать сессию; `GET /sessions/{id}` — состояние; `GET/POST /sessions/{id}/messages` — история/новое сообщение.
- `POST /sessions/{id}/tasks/{task_id}/submit_code|submit_sql` — отправка решений в песочницу.
- `POST /sessions/{id}/score` — эквивалент tool `score_task` с проверкой max_points из сценария.
- `POST /rag/corpora` + `POST /rag/corpora/{id}/documents` + `POST /rag/search` — загрузка и поиск документов.
- CRUD: `/roles`, `/scenarios`, `/sql-scenarios`.

## Следующие шаги
- Подключить реальные песочницы и LM Studio, заменить заглушки.
- Добавить сохранение итоговых отчётов и творческих заданий.
- Расширить фронт: переключение задач, блокировка редакторов после submit, WebSocket для живых обновлений.
