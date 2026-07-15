import os
import json
import asyncio
import shutil
import tempfile
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from datetime import datetime
import uuid

import chromadb
from chromadb.utils import embedding_functions
from pymongo import MongoClient

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import auth
import ingest as ingest_module

# ---------------------------------------------------------------------------
# Environment / LangSmith
# ---------------------------------------------------------------------------
# Load secrets from .env — never hardcode API keys in source.
# See .env.example for the full list of variables this app reads.
load_dotenv()

# LangSmith is enabled purely by environment variables; LangChain's runnables
# pick these up automatically (no code-level SDK calls needed). We just make
# sure sane defaults exist so tracing "just works" once LANGCHAIN_API_KEY is set.
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.environ.get("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_PROJECT", os.environ.get("LANGCHAIN_PROJECT", "ollamachat"))
if os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true" and not os.environ.get("LANGCHAIN_API_KEY"):
    print("[langsmith] WARNING: LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is not set — tracing will fail silently.")


app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

PROMPTS_DIR = Path(__file__).parent / "prompts"
CONFIG_FILE = PROMPTS_DIR / "config.yaml"

UPLOAD_MAX_BYTES = 20 * 1024 * 1024  # 20 MB per file

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
mongo_client = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))
mongo_db = mongo_client["chatbot"]
mongo_col = mongo_db["messages"]      # now includes a "user_id" field per document
users_col = mongo_db["users"]
documents_col = mongo_db["documents"]  # tracks each user's uploaded files

auth_service = auth.AuthService(users_col)

# ---------------------------------------------------------------------------
# Chroma — two collections
# 1. conversations : in-memory, stores chat history embeddings per session
# 2. knowledge_base: persistent, stores uploaded-document chunks (per user)
# ---------------------------------------------------------------------------
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_collection = chroma_client.get_or_create_collection(
    name="conversations", embedding_function=embedding_fn
)

# knowledge_base collection now lives inside ingest.py so both app.py and the
# CLI ingester share exactly one PersistentClient instance.
kb_collection = ingest_module.get_kb_collection()

# ---------------------------------------------------------------------------
# Guardrail LLM  (unchanged from original)
# ---------------------------------------------------------------------------
_GUARDRAIL_MODEL = "qwen2.5:1.5b"
# num_predict=80 - enough for "GREETING", "SAFE", or "UNSAFE: <reason>"
_classifier_llm = ChatOllama(model=_GUARDRAIL_MODEL, temperature=0, num_predict=80)

_classifier_chain = None


def _load_guardrail_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
    else:
        data = {}
    return data.get("guardrails", {})


def _get_guardrail_rules() -> str:
    rules = _load_guardrail_config().get("rules", [])
    if rules:
        return "\n".join(f"- {r}" for r in rules)
    fallback = _load_guardrail_config().get("fallback_rules", "")
    if not fallback:
        raise KeyError("No rules or fallback_rules found under 'guardrails' in config.yaml.")
    return f"- {fallback}"


def _build_guardrail_chains():
    global _classifier_chain
    guard_cfg = _load_guardrail_config()
    classifier_template = guard_cfg.get("classifier_prompt")

    if not classifier_template:
        raise KeyError("Missing 'classifier_prompt' under 'guardrails' in config.yaml.")

    _classifier_chain = PromptTemplate.from_template(classifier_template) | _classifier_llm | StrOutputParser()


async def check_guard(message: str) -> tuple[bool, str]:
    """
    Single-call guardrail. The classifier prompt returns exactly one of:
      GREETING            -> trivial message, no request/question -> allow
      SAFE                -> normal request, no violation -> allow
      UNSAFE: <reason>     -> violates a rule -> block, reason surfaced to the refusal prompt
    """
    clean_message = message.strip().strip('"\'')
    rules = _get_guardrail_rules()

    try:
        raw = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _classifier_chain.invoke({"rules": rules, "message": clean_message}).strip(),
        )
    except Exception as e:
        print(f"[guardrail] error: {e} - failing closed")
        return False, "guardrail check failed"

    print(f"[guardrail] input={clean_message!r}  output={raw!r}")
    upper = raw.upper()

    if upper.startswith("GREETING"):
        return True, ""

    # Check UNSAFE before SAFE — "SAFE" is a substring of "UNSAFE" so order matters
    if "UNSAFE" in upper:
        unsafe_pos = upper.find("UNSAFE")
        reason_raw = raw[unsafe_pos + len("UNSAFE"):].lstrip(": ").strip()
        reason = reason_raw.splitlines()[0].strip() if reason_raw else "flagged by guardrail"
        return False, reason or "flagged by guardrail"

    if upper.startswith("SAFE"):
        return True, ""

    print(f"[guardrail] unexpected response: {raw!r} - failing closed")
    return False, f"Message blocked: {raw[:80]}"


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
PERSONAS: dict = {}
DEFAULT_PERSONA_ID = "default"


def load_config():
    """
    Single-config mode: config.yaml has top-level `system_prompt` / `max_history`
    instead of a `personas:` list. We wrap that single config into the same
    PERSONAS dict shape the rest of the app expects (keyed by DEFAULT_PERSONA_ID),
    so build_chain / get_persona / the /personas endpoint / index.html all keep
    working unchanged — there's just exactly one persona, always "default".
    """
    global PERSONAS
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
    else:
        data = {}

    system_prompt = data.get("system_prompt")
    if not system_prompt:
        raise KeyError("Missing 'system_prompt' in config.yaml.")

    PERSONAS = {
        DEFAULT_PERSONA_ID: {
            "id": DEFAULT_PERSONA_ID,
            "name": data.get("name", "Assistant"),
            "system_prompt": system_prompt,
            "max_history": data.get("max_history", 20),
        }
    }
    # Rebuild guardrail chains so any prompt edits in YAML take effect immediately
    _build_guardrail_chains()


load_config()


def get_persona(persona_id: str) -> dict:
    return PERSONAS.get(persona_id, PERSONAS.get(DEFAULT_PERSONA_ID, {}))


# ---------------------------------------------------------------------------
# Session / History — now scoped per (user_id, session_id)
# ---------------------------------------------------------------------------
_chat_histories: dict[str, InMemoryChatMessageHistory] = {}
_session_personas: dict[str, str] = {}


def _history_key(user_id: str, session_id: str) -> str:
    return f"{user_id}:{session_id}"


def _persist_exchange(user_id: str, session_id: str, user_msg: str, ai_msg: str) -> None:
    now = datetime.utcnow()
    exchange_id = str(uuid.uuid4())

    mongo_col.insert_one({
        "user_id": user_id,
        "session_id": session_id,
        "user": user_msg,
        "assistant": ai_msg,
        "timestamp": now,
    })

    combined_text = f"User: {user_msg}\nAssistant: {ai_msg}"
    chroma_collection.add(
        documents=[combined_text],
        metadatas=[{"user_id": user_id, "session_id": session_id, "timestamp": now.isoformat()}],
        ids=[exchange_id],
    )


def get_history(user_id: str, session_id: str) -> InMemoryChatMessageHistory:
    key = _history_key(user_id, session_id)
    if key not in _chat_histories:
        hist = InMemoryChatMessageHistory()
        # Scoped to this user — a session_id from another account can never
        # leak history, even if guessed or reused.
        for doc in mongo_col.find({"user_id": user_id, "session_id": session_id}).sort("timestamp", 1):
            user_msg = doc.get("user", "")
            ai_msg = doc.get("assistant", "")
            if user_msg:
                hist.add_user_message(user_msg)
            if ai_msg:
                hist.add_ai_message(ai_msg)
        _chat_histories[key] = hist
    return _chat_histories[key]


_chat_llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.7,
    streaming=True,
)


# ---------------------------------------------------------------------------
# Knowledge base retrieval — now filtered to the requesting user's documents
# ---------------------------------------------------------------------------
def retrieve_context(query: str, user_id: str, n_results: int = 5) -> str:
    total = kb_collection.count()
    if total == 0:
        return ""

    results = kb_collection.query(
        query_texts=[query],
        n_results=min(n_results, total),
        where={"user_id": user_id},
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        print(f"[RAG] no chunks retrieved for user={user_id!r} query={query!r}")
        return ""

    print(f"[RAG] user={user_id!r} query={query!r} kb_total={total} retrieved={len(docs)}")
    parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        relevance = round((1 - dist) * 100, 1)
        snippet = doc[:80].replace("\n", " ")
        print(f"  chunk {i+1}: [{source} p.{page}] score={relevance}%  snippet={snippet!r}")
        if relevance >= 35:
            parts.append(f"[Source: {source}, Page {page}]\n{doc.strip()}")
        else:
            print(f"  chunk {i+1}: skipped (score too low)")

    if not parts:
        return ""

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chain builder — unchanged except get_history now needs a user_id, handled
# via a per-request closure passed into RunnableWithMessageHistory below.
# ---------------------------------------------------------------------------
def build_chain(persona: dict, user_id: str, rag_context: str = "", refusal_context: str = ""):
    guard_cfg = _load_guardrail_config()
    system_prompt = persona["system_prompt"]

    if refusal_context:
        system_prompt += f"\n\n{refusal_context}"
        human_template = "{input}"

    elif rag_context:
        rag_header = guard_cfg.get("rag_context_header")
        if not rag_header:
            raise KeyError("Missing 'rag_context_header' under 'guardrails' in config.yaml.")
        human_template = (
            f"{rag_header}\n\n"
            f"{rag_context}\n\n"
            "---\n"
            "Question: {input}\n"
            "Answer strictly from the excerpts above. "
            "If the answer is not in the excerpts, say you don't have that information."
        )
    else:
        human_template = "{input}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", human_template),
    ])

    # session_id passed through the `configurable` dict at .stream() time is
    # just the raw session_id, so we close over user_id here to scope history
    # lookups correctly without changing RunnableWithMessageHistory's API.
    def _get_history_for_session(session_id: str) -> InMemoryChatMessageHistory:
        return get_history(user_id, session_id)

    return RunnableWithMessageHistory(
        prompt | _chat_llm,
        _get_history_for_session,
        input_messages_key="input",
        history_messages_key="history",
    )


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.get("/login")
async def login_page(request: Request):
    if auth.get_optional_user_id(request):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/auth/register")
async def register(request: Request):
    data = await request.json()
    try:
        user = auth_service.register(
            username=data.get("username", ""),
            password=data.get("password", ""),
            email=data.get("email", ""),
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    resp = JSONResponse({"status": "registered", "username": user["username"]})
    auth.set_session_cookie(resp, str(user["_id"]))
    return resp


@app.post("/auth/login")
async def login(request: Request):
    data = await request.json()
    user = auth_service.authenticate(data.get("username", ""), data.get("password", ""))
    if not user:
        return JSONResponse({"error": "Invalid username or password"}, status_code=401)

    resp = JSONResponse({"status": "logged_in", "username": user["username"]})
    auth.set_session_cookie(resp, str(user["_id"]))
    return resp


@app.post("/auth/logout")
async def logout():
    resp = JSONResponse({"status": "logged_out"})
    auth.clear_session_cookie(resp)
    return resp


@app.get("/auth/me")
async def me(user_id: str = Depends(auth.get_current_user_id)):
    user = auth_service.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return JSONResponse({"user_id": user_id, "username": user["username"], "email": user.get("email", "")})


# ---------------------------------------------------------------------------
# Page / persona routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index(request: Request):
    user_id = auth.get_optional_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)
    user = auth_service.get_by_id(user_id)
    if not user:
        # Cookie referenced a user that no longer exists — force re-login.
        resp = RedirectResponse(url="/login", status_code=302)
        auth.clear_session_cookie(resp)
        return resp

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "personas": list(PERSONAS.values()),
            "default_persona": PERSONAS.get(DEFAULT_PERSONA_ID, {}),
            "username": user["username"],
        },
    )


@app.get("/personas")
async def personas_list():
    return JSONResponse(list(PERSONAS.values()))


@app.post("/personas/reload")
async def reload_personas():
    load_config()
    return JSONResponse({"status": "reloaded", "count": len(PERSONAS)})


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: Request, user_id: str = Depends(auth.get_current_user_id)):
    data = await request.json()
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    persona_id = data.get("persona_id", DEFAULT_PERSONA_ID)

    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    ok, reason = await check_guard(user_message)

    context = await asyncio.get_event_loop().run_in_executor(
        None, retrieve_context, user_message, user_id
    )

    hist_key = _history_key(user_id, session_id)
    prev_persona = _session_personas.get(hist_key)
    if prev_persona and prev_persona != persona_id:
        _chat_histories.pop(hist_key, None)

    _session_personas[hist_key] = persona_id
    persona = get_persona(persona_id)

    max_history = persona.get("max_history", 20)
    history = get_history(user_id, session_id)
    if len(history.messages) > max_history * 2:
        history.messages = history.messages[-(max_history * 2):]

    if not ok:
        guard_cfg = _load_guardrail_config()
        refusal_template = guard_cfg.get("refusal_instruction")
        if not refusal_template:
            raise KeyError("Missing 'refusal_instruction' under 'guardrails' in config.yaml.")
        refusal_context = refusal_template.replace("{reason}", reason)
        chain = build_chain(persona, user_id, refusal_context=refusal_context)
    else:
        chain = build_chain(persona, user_id, rag_context=context)

    # LangSmith: tag every run with user/session/persona so traces are
    # filterable in the LangSmith UI without any extra instrumentation code.
    run_config = {
        "configurable": {"session_id": session_id},
        "tags": [f"persona:{persona_id}", "guardrail:blocked" if not ok else "guardrail:passed"],
        "metadata": {
            "user_id": user_id,
            "session_id": session_id,
            "persona_id": persona_id,
            "rag_used": bool(context),
        },
        "run_name": f"chat:{persona_id}",
    }

    def generate():
        ai_accum = ""
        try:
            for chunk in chain.stream({"input": user_message}, config=run_config):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if not token:
                    continue
                ai_accum += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            _persist_exchange(user_id, session_id, user_message, ai_accum)
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/history/{session_id}")
async def get_session_history(session_id: str, user_id: str = Depends(auth.get_current_user_id)):
    history = get_history(user_id, session_id)
    messages = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            messages.append({"role": "system", "content": msg.content})
    return JSONResponse({"session_id": session_id, "messages": messages})


@app.post("/clear")
async def clear(request: Request, user_id: str = Depends(auth.get_current_user_id)):
    data = await request.json() or {}
    session_id = data.get("session_id", "default")
    hist_key = _history_key(user_id, session_id)
    _chat_histories.pop(hist_key, None)
    _session_personas.pop(hist_key, None)
    return JSONResponse({"status": "cleared", "session_id": session_id})


# ---------------------------------------------------------------------------
# Document upload / RAG management
# ---------------------------------------------------------------------------
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(auth.get_current_user_id),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ingest_module.SUPPORTED_EXTENSIONS:
        return JSONResponse(
            {"error": f"Unsupported file type '{suffix}'. Allowed: pdf, txt, md"},
            status_code=400,
        )

    # Stream to a temp file with a hard size cap instead of reading the
    # whole upload into memory at once.
    tmp_dir = Path(tempfile.mkdtemp(prefix="ollamachat_upload_"))
    tmp_path = tmp_dir / file.filename
    size = 0
    try:
        with open(tmp_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > UPLOAD_MAX_BYTES:
                    raise ValueError("File too large (max 20MB)")
                out.write(chunk)

        chunk_count = ingest_module.ingest_file(tmp_path, user_id, clear_existing=True)

        documents_col.update_one(
            {"user_id": user_id, "filename": file.filename},
            {"$set": {
                "user_id": user_id,
                "filename": file.filename,
                "chunk_count": chunk_count,
                "uploaded_at": datetime.utcnow(),
                "size_bytes": size,
            }},
            upsert=True,
        )

        return JSONResponse({
            "status": "ingested",
            "filename": file.filename,
            "chunks": chunk_count,
        })

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Failed to process file: {e}"}, status_code=500)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/documents")
async def list_documents(user_id: str = Depends(auth.get_current_user_id)):
    docs = list(documents_col.find({"user_id": user_id}, {"_id": 0}).sort("uploaded_at", -1))
    for d in docs:
        d["uploaded_at"] = d["uploaded_at"].isoformat()
    return JSONResponse({"documents": docs})


@app.delete("/documents/{filename}")
async def delete_document(filename: str, user_id: str = Depends(auth.get_current_user_id)):
    ingest_module.delete_file(filename, user_id)
    result = documents_col.delete_one({"user_id": user_id, "filename": filename})
    if result.deleted_count == 0:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    return JSONResponse({"status": "deleted", "filename": filename})


if __name__ == "__main__":
    print("OllamaChat - LangChain + YAML Prompts (FastAPI)")
    print("  Make sure Ollama is running: ollama serve")
    print("  Model: ollama pull qwen2.5:1.5b")
    print("  Personas loaded:", ", ".join(PERSONAS.keys()))
    print("  LangSmith tracing:", os.environ.get("LANGCHAIN_TRACING_V2", "false"))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)