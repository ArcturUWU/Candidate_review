from typing import Any, Dict, List

import httpx


def web_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Simple web search via DuckDuckGo instant answer; falls back to stub."""
    try:
        url = f"https://api.duckduckgo.com/?q={httpx.utils.quote(query)}&format=json&no_redirect=1&no_html=1"
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results: List[Dict[str, Any]] = []
        for topic in data.get("RelatedTopics", []):
            if "Text" in topic and "FirstURL" in topic:
                results.append(
                    {
                        "title": topic.get("Text", "")[:120],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                    }
                )
            if len(results) >= top_k:
                break
        if not results:
            results.append(
                {
                    "title": "Результаты не найдены",
                    "url": "",
                    "snippet": f"По запросу нет точных совпадений; опираемся на знания модели. Запрос: {query}",
                }
            )
        return results
    except Exception:
        return [
            {
                "title": "stub result",
                "url": "https://example.com",
                "snippet": f"No external search, query='{query}'",
            }
        ]
