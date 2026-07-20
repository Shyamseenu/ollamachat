# Chatbot — LangChain + YAML Prompts + Auth + RAG Uploads

An AI chatbot with **LangChain conversation memory**, a **YAML-driven persona/prompt system**, **per-user authentication**, **LangSmith tracing**, and a **drag-and-drop RAG document pipeline** so each user can chat with their own uploaded files. Powered by **Google Gemini** via `langchain-google-genai`.

## Setup

**1. Get a Gemini API key**

Create one at https://aistudio.google.com/apikey — free tier available (Flash models, rate-limited; see [Google's pricing page](https://ai.google.dev/pricing) for current limits).

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"   # paste into SESSION_SECRET_KEY in .env
```

Add your Gemini key to `.env` as `GOOGLE_API_KEY`, and set `GEMINI_MODEL` (defaults to `gemini-3.5-flash` if unset).

Get a LangSmith API key at https://smith.langchain.com/ → **Settings → API Keys** and add it to `.env` as `LANGCHAIN_API_KEY` (or set `LANGCHAIN_TRACING_V2=false` to disable tracing). See [Environment Variables](#environment-variables) below for the full list.

**4. Set up MongoDB Atlas**

1. Create a free cluster at https://www.mongodb.com/cloud/atlas/register (M0 free tier)
2. Create a database user (username/password) under **Database Access**
3. Under **Network Access**, add an IP Access List entry — for local dev or a deployment without a fixed IP, use `0.0.0.0/0` (allow all)
4. Under **Database → Connect → Drivers**, copy the connection string and set it as `MONGO_URI` in `.env`:
   ```
   MONGO_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
   (URL-encode the password if it contains special characters like `@` or `/`)

**5. Run**

```bash
uvicorn app:app --reload   # starts FastAPI on port 8000
```

**6. Open** → http://localhost:8000 — you'll land on the sign-in page; create an account to get started.

---

## Project Structure

```
chatbot/
├── app.py                   # FastAPI app + LangChain chains + auth + document routes
├── auth.py                  # Password hashing, signed session cookies, user store
├── ingest.py                # Document ingestion (PDF/TXT/MD) — CLI + importable functions
├── check_mongo.py           # Quick script to sanity-check the Mongo connection
├── requirements.txt
├── .env.example             # Copy to .env and fill in secrets
├── prompts/                 # ← YAML persona + guardrail config
│   ├── config.yaml
│   ├── default.yaml
│   ├── coder.yaml
│   └── tutor.yaml
├── templates/
│   ├── index.html            # Main chat UI (auth-gated, includes document sidebar)
│   └── login.html            # Sign-in / registration page
└── chroma_db/                # Persistent Chroma store for uploaded-document vectors
```

---

## Authentication

- Visiting `/` while logged out redirects to `/login`, which doubles as both the sign-in and registration form.
- Sessions are stateless, signed, HttpOnly cookies (no JWTs, no server-side session table) — see `auth.py`. They last 7 days by default.
- Every account's chat history and uploaded documents are private to that account; nothing is shared across users.
- `SESSION_SECRET_KEY` must be set in `.env` for sessions to survive a server restart. If it's missing, a random one is generated at boot (with a printed warning) and all existing sessions are invalidated.
- Set `COOKIE_SECURE=true` in `.env` once you're serving over HTTPS.

> This is a minimal auth system meant for local/personal or small internal use — there's no email verification, password reset, or login rate-limiting out of the box. Add those before exposing it publicly.

---

## LangSmith Tracing

Tracing is enabled purely through environment variables (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` in `.env`) — LangChain's runnables pick these up automatically, no extra code required. Every chat run is additionally tagged with the persona, session, user, and whether RAG context or the guardrail fired, so you can filter and debug runs in the LangSmith dashboard at https://smith.langchain.com/.

Set `LANGCHAIN_TRACING_V2=false` to turn tracing off entirely (e.g. for offline development).

---

## RAG Document Upload

- Drag a `.pdf`, `.txt`, or `.md` file (up to 20MB) onto the sidebar in the chat UI, or click to browse.
- The file is parsed, chunked (`RecursiveCharacterTextSplitter`, 500 chars / 50 overlap by default), embedded locally with `all-MiniLM-L6-v2` (via `sentence-transformers` — no external API call), and stored in the persistent `knowledge_base` Chroma collection — scoped to your account via metadata, so you'll only ever retrieve chunks from your own uploads.
- Once a file is indexed, just ask a question about it in the chat — relevant chunks (≥35% similarity) are automatically retrieved and injected into the prompt.
- Re-uploading a file with the same name replaces its old vectors; the ✕ button in the sidebar removes a document entirely.
- Bulk/manual ingestion is still available from the command line:
  ```bash
  python ingest.py --file handbook.pdf --user <user_id>
  ```

---

## Adding a New Persona

Create a new `.yaml` file in `prompts/` — no restart needed, just click **⟳ RELOAD** in the UI:

```yaml
id: my_persona # unique key (must match filename)
name: My Persona # display name in UI
description: One-line description
model: gemini-3.5-flash # any Gemini model — see GEMINI_MODEL env var
temperature: 0.7 # 0.0 (focused) → 1.0 (creative)
max_history: 20 # how many past exchanges to keep

system_prompt: |
  You are ...
  Your rules:
  - ...

welcome_message: "Hello! What can I help with?"
```

---

## API Endpoints

| Method | Route                   |        Auth required        | Description                                      |
| ------ | ----------------------- | :-------------------------: | ------------------------------------------------ |
| GET    | `/`                     | — (redirects if logged out) | Web UI                                           |
| GET    | `/login`                |              —              | Sign-in / registration page                      |
| POST   | `/auth/register`        |              —              | Create an account, sets session cookie           |
| POST   | `/auth/login`           |              —              | Authenticate, sets session cookie                |
| POST   | `/auth/logout`          |              —              | Clears session cookie                            |
| GET    | `/auth/me`              |              ✔              | Current user info                                |
| POST   | `/chat`                 |              ✔              | Stream a chat response                           |
| GET    | `/personas`             |              —              | List all loaded personas                         |
| POST   | `/personas/reload`      |              —              | Hot-reload configuration (personas + guardrails) |
| GET    | `/history/{session_id}` |              ✔              | Get conversation history                         |
| POST   | `/clear`                |              ✔              | Clear a session's history                        |
| POST   | `/documents/upload`     |              ✔              | Upload + ingest a PDF/TXT/MD file                |
| GET    | `/documents`            |              ✔              | List your uploaded documents                     |
| DELETE | `/documents/{filename}` |              ✔              | Remove a document and its vectors                |

### `/chat` request body

```json
{
  "message": "Your question here",
  "session_id": "unique-session-id",
  "persona_id": "coder"
}
```

### `/documents/upload` request

`multipart/form-data` with a single `file` field (`.pdf`, `.txt`, or `.md`, ≤20MB).

> 🔐 **Guardrails**
> The server filters user input using a small LLM-based safety check (Gemini, with thinking disabled for fast single-word classification), configured entirely in `prompts/config.yaml`.
>
> - Rules are plain-English descriptions, not regex — the safety-check model reads them directly.
> - Refusal wording, the RAG context header, and both guardrail prompts are all editable in the YAML — no code changes needed.
> - Edit `prompts/config.yaml` and hit `/personas/reload` to apply changes without restarting the server.

---

## Environment Variables

All read from `.env` (see `.env.example`):

| Variable               | Purpose                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| `GOOGLE_API_KEY`       | Gemini API key from Google AI Studio.                               |
| `GEMINI_MODEL`         | Gemini model to use (defaults to `gemini-3.5-flash`).               |
| `SESSION_SECRET_KEY`   | Signs session cookies. Set explicitly so sessions survive restarts. |
| `COOKIE_SECURE`        | `true`/`false` — send the session cookie only over HTTPS.           |
| `MONGO_URI`            | MongoDB Atlas connection string.                                    |
| `LANGCHAIN_TRACING_V2` | `true`/`false` — enable LangSmith tracing.                          |
| `LANGCHAIN_API_KEY`    | Your LangSmith API key.                                             |
| `LANGCHAIN_PROJECT`    | LangSmith project name traces are grouped under.                    |
| `LANGCHAIN_ENDPOINT`   | Optional — only for self-hosted LangSmith.                          |

---

## How It Works

- **LangChain** manages conversation memory via `InMemoryChatMessageHistory` and `RunnableWithMessageHistory`, backed by MongoDB Atlas for persistence across restarts.
- Each user has their own isolated chat sessions and their own isolated RAG document store — enforced at the query level, not just the UI.
- Switching personas automatically resets history for that session.
- YAML files define the full prompt/guardrail config — edit and reload without restarting.
- Responses stream token-by-token via Server-Sent Events (SSE).
- Every LLM call is traced to LangSmith (when enabled) with user/session/persona metadata for debugging and monitoring.
