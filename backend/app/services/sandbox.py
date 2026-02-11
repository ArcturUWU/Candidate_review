from typing import Any, Dict

import httpx

from ..config import settings


def run_code(language: str, code: str, tests_id: str) -> Dict[str, Any]:
    payload = {"language": language, "code": code, "tests_id": tests_id}
    try:
        resp = httpx.post(settings.sandbox_code_url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "details": f"sandbox run_code failed: {exc}"}


def run_sql(sql_scenario_id: str, query: str) -> Dict[str, Any]:
    payload = {"sql_scenario_id": sql_scenario_id, "query": query}
    try:
        resp = httpx.post(settings.sandbox_sql_url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"sandbox run_sql failed: {exc}"}
