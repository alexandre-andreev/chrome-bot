import os
import logging
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from dotenv import load_dotenv

# --- Environment Loading ---
# Explicitly load .env.local from the project root, which is 2 levels up
# from this file (backend/app/main.py)
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / '.env.local'

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # Fallback to default .env search if .env.local is not found
    load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai
from exa_py import Exa

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# --- API Clients Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")

if not GEMINI_API_KEY:
    logger.critical("CRITICAL: GEMINI_API_KEY not found. Please check your .env.local file.")
else:
    logger.info("GEMINI_API_KEY loaded successfully.")

if not EXA_API_KEY:
    logger.critical("CRITICAL: EXA_API_KEY not found. Please check your .env.local file.")
else:
    logger.info("EXA_API_KEY loaded successfully.")

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    
    if EXA_API_KEY:
        exa_client = Exa(api_key=EXA_API_KEY)
    else:
        exa_client = None
except Exception as e:
    logger.critical(f"Failed to configure API clients: {e}")
    exa_client = None

# --- Exa Search Tool Definition for Gemini ---
def search_exa(query: str) -> List[Dict[str, Any]]:
    """
    Performs a web search using the Exa API to find relevant information.
    Args:
        query: The search query.
    Returns:
        A list of search results, each containing a title, URL, and snippet.
    """
    if not exa_client:
        logger.error("Exa client not configured or API key is missing.")
        return [{"error": "Exa client not configured"}]
    try:
        results = exa_client.search_and_contents(
            query,
            num_results=3,
            text={"max_characters": 1000},
        )
        return [
            {
                "title": res.title,
                "url": res.url,
                "snippet": res.text,
            }
            for res in results.results
        ]
    except Exception as e:
        logging.error(f"Exa API search failed: {e}")
        return [{"error": f"Search failed with exception: {e}"}]

# --- Gemini Model with Tool ---
# Properly declare the `search_exa` function as a tool for the Gemini SDK
SEARCH_EXA_TOOL = {
    "function_declarations": [
        {
            "name": "search_exa",
            "description": "Выполняет веб‑поиск по запросу и возвращает список релевантных результатов с заголовками, URL и сниппетами.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "Строка поискового запроса на любом языке."
                    }
                },
                "required": ["query"]
            },
        }
    ]
}

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
FAST_SEARCH_ONLY = os.getenv("FAST_SEARCH_ONLY", "0").strip() in {"1", "true", "yes", "on"}
STRICT_ONPAGE = os.getenv("STRICT_ONPAGE", "0").strip() in {"1", "true", "yes", "on"}

model = genai.GenerativeModel(
    model_name=DEFAULT_GEMINI_MODEL,
    tools=[SEARCH_EXA_TOOL],
) if GEMINI_API_KEY else None

# --- FastAPI App & Pydantic Models ---
app = FastAPI(title="Chrome-bot Backend", version="1.3.0")

# CORS configuration via environment variable (comma-separated origins)
raw_origins = os.getenv("CORS_ALLOW_ORIGINS")
allow_origins = [o.strip() for o in raw_origins.split(",") if o.strip()] if raw_origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class ChatRequest(BaseModel):
    message: str
    page_url: Optional[str] = None
    page_title: Optional[str] = None
    page_text: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    used_external: bool = False

# --- Heuristics to decide when to force external search ---
FALLBACK_NEGATIVE_PATTERNS = [
    r"не могу предоставить",
    r"на странице (?:нет|не представлена|не указана)",
    r"нет (?:таких )?сведени",
    r"нет информации",
    r"insufficient",
    r"i cannot provide",
    r"not available on the page",
]

FALLBACK_QUERY_PATTERNS = [
    r"где купить",
    r"купить",
    r"цена",
    r"стоимост",
    r"адрес",
    r"доставк",
    r"where to buy",
    r"price",
    r"availability",
]

def _should_fallback_search(user_message: str, response_text: str) -> bool:
    text = (response_text or "").lower()
    msg = (user_message or "").lower()
    for pat in FALLBACK_NEGATIVE_PATTERNS:
        if re.search(pat, text):
            return True
    for pat in FALLBACK_QUERY_PATTERNS:
        if re.search(pat, msg):
            return True
    return False

def _build_search_query(user_message: str, page_title: Optional[str], page_text: Optional[str]) -> str:
    title_part = (page_title or "").strip()
    text_part = ((page_text or "").strip().replace("\n", " "))[:200]
    parts: List[str] = []
    if title_part:
        parts.append(title_part)
    if text_part:
        parts.append(text_part)
    parts.append(user_message.strip())
    query = " ".join(p for p in parts if p)
    return query[:300]

def _should_allow_external(user_message: str, page_text: Optional[str]) -> bool:
    msg = (user_message or "").lower()
    text_len = len(page_text or "")
    # Explicit shopping/price/availability queries – allow external
    for pat in FALLBACK_QUERY_PATTERNS:
        if re.search(pat, msg):
            return True
    # If page has very little content, allow external
    if text_len < 400:
        return True
    # If STRICT_ONPAGE set and text is long enough, prefer on-page
    if STRICT_ONPAGE and text_len >= 800:
        return False
    # For common factoid intents with long text, prefer on-page
    if text_len >= 1200 and re.search(r"\b(что такое|когда|кто|где|каког|истори|описани)\b", msg):
        return False
    # Otherwise allow tools
    return True

def _extract_relevant_sentences(user_message: str, page_text: Optional[str], max_sentences: int = 3) -> str:
    if not page_text:
        return ""
    text = page_text.replace("\n", " ")
    # naive sentence split
    sentences = re.split(r"(?<=[\.!?…])\s+", text)
    umsg = (user_message or "").lower()
    # keywords: non-trivial tokens > 3 chars
    tokens = [t for t in re.findall(r"[a-zа-яё0-9-]+", umsg, re.IGNORECASE) if len(t) >= 4]
    scores: List[tuple[int, str]] = []
    for s in sentences:
        sl = s.lower()
        score = 0
        for t in tokens:
            if t in sl:
                score += 2
        # boost for year mention if asking 'когда'
        if "когда" in umsg and re.search(r"\b(1[89]\d{2}|20\d{2})\b", sl):
            score += 3
        if score:
            scores.append((score, s.strip()))
    scores.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scores[:max_sentences]]
    return "\n".join(top)

# --- API Endpoints ---
@app.get("/")
async def root() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    if not model:
        raise HTTPException(status_code=500, detail="Gemini model is not configured. Check API key.")

    user_message = (payload.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")

    # Use a ChatSession to handle the conversation history and tool calls correctly.
    chat_session = model.start_chat()

    # A more direct and forceful prompt to encourage tool use.
    system_prompt = (
        "Ты — умный ассистент в браузере. Ты даешь содержательные ответы, но не более 15 предложений в одном ответе."
        "Твоя главная задача — отвечать на вопросы пользователя, основываясь на контексте предоставленной веб-страницы."
        "Если информации на текущей странице недостаточно для ответа на вопрос (например, пользователь "
        "просит информацию, которой нет на странице), "
        "ты ОБЯЗАН использовать инструмент `search_exa` для поиска в интернете. "
        "Это твой ЕДИНСТВЕННЫЙ способ получить внешнюю информацию. "
        "Не извиняйся, что не можешь искать, а просто используй инструмент."
    )
    
    prompt_parts = [system_prompt]
    ctx_lines: List[str] = []
    if payload.page_url:
        ctx_lines.append(f"URL: {payload.page_url}")
    if payload.page_title:
        ctx_lines.append(f"TITLE: {payload.page_title}")
    if payload.page_text:
        # allow larger context
        snippet = payload.page_text.strip().replace("\n", " ")[:6000]
        if snippet:
            ctx_lines.append(f"TEXT: {snippet}")
    
    if ctx_lines:
        prompt_parts.append("КОНТЕКСТ СТРАНИЦЫ:\n" + "\n".join(ctx_lines))
    
    prompt_parts.append("ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n" + user_message)
    
    # Add relevant sentences extracted from the page to guide the model
    relevant = _extract_relevant_sentences(user_message, payload.page_text)
    if relevant:
        prompt_parts.append("ПОДСКАЗКИ ИЗ ТЕКСТА СТРАНИЦЫ (используй для ответа):\n" + relevant)
    final_prompt = "\n\n".join(prompt_parts)

    try:
        # Decide whether to allow external tools for this request
        allow_external = _should_allow_external(user_message, payload.page_text)
        tool_cfg = {"function_calling_config": {"mode": "AUTO"}} if allow_external else {"function_calling_config": {"mode": "NONE"}}
        response = chat_session.send_message(final_prompt, tool_config=tool_cfg)
        candidate = response.candidates[0]
        parts = getattr(candidate.content, "parts", []) or []

        # Find function call part if present
        function_call = None
        for p in parts:
            if getattr(p, "function_call", None):
                function_call = p.function_call
                break

        if function_call and allow_external:
            if function_call.name == "search_exa":
                query = function_call.args["query"]
                logger.info(f"Gemini requested search with query: '{query}'")
                search_results = search_exa(query)

                # Fast path: return immediately without second model pass if enabled
                if FAST_SEARCH_ONLY:
                    lines: List[str] = []
                    for s in search_results:
                        url = s.get("url")
                        if url:
                            title = s.get("title") or url
                            lines.append(f"- {title}: {url}")
                    fast_answer = "Нашёл внешние источники по запросу:\n" + ("\n".join(lines) if lines else "Источники не найдены.")
                    return ChatResponse(answer=fast_answer, sources=search_results, used_external=True)

                # Try to hand results back to the model. If it fails, return a direct answer with sources.
                try:
                    second_response = chat_session.send_message({
                        "function_response": {
                            "name": "search_exa",
                            "response": {"results": search_results},
                        }
                    })
                    final_answer = second_response.text
                    if final_answer:
                        return ChatResponse(answer=final_answer, sources=search_results, used_external=True)
                except Exception as tool_err:
                    logger.warning(f"Tool handoff failed, returning direct sources. Error: {tool_err}")

                # Direct concise answer using sources
                lines: List[str] = []
                for s in search_results:
                    url = s.get("url")
                    if url:
                        title = s.get("title") or url
                        lines.append(f"- {title}: {url}")
                direct_answer = "Нашёл внешние источники по запросу.\n" + ("\n".join(lines) if lines else "Источники не найдены.")
                return ChatResponse(answer=direct_answer, sources=search_results, used_external=True)
            else:
                 return ChatResponse(answer=f"Model requested unknown function: {function_call.name}")
        
        # If no function call, collect direct text and maybe fallback
        text_parts = [getattr(p, "text", None) for p in parts]
        text_parts = [t for t in text_parts if t]
        direct_text = "\n\n".join(text_parts) if text_parts else ""

        # Fallback: if allowed and model refused or info missing, run Exa search
        if allow_external and _should_fallback_search(user_message, direct_text):
            query = _build_search_query(user_message, payload.page_title, payload.page_text)
            logger.info(f"Fallback search triggered. Query: '{query}'")
            search_results = search_exa(query)
            # Build a concise answer summarizing available sources
            sources_lines = []
            for s in search_results:
                if s.get("url"):
                    title = s.get("title") or s["url"]
                    sources_lines.append(f"- {title}: {s['url']}")
            answer = (direct_text or "Недостаточно данных на странице.") + "\n\n" + (
                "Нашёл внешние источники:\n" + "\n".join(sources_lines) if sources_lines else "Не удалось найти внешние источники."
            )
            return ChatResponse(answer=answer, sources=search_results, used_external=True)

        if direct_text:
            return ChatResponse(answer=direct_text, used_external=False)
        logger.error(f"Unexpected empty response from Gemini: {response}")
        return ChatResponse(answer="Получен пустой ответ от модели.")

    except Exception as e:
        logger.exception(f"Error during Gemini chat: {e}")
        if "API key not valid" in str(e):
             raise HTTPException(status_code=500, detail="Ключ API для Gemini недействителен. Проверьте .env файл.")
        raise HTTPException(status_code=500, detail="Произошла внутренняя ошибка сервера при общении с моделью.")


