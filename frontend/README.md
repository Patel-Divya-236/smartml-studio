# SmartML Studio — React workspace

React single-page front end for SmartML Studio, over a FastAPI backend that wraps the
existing `src/` package. The eleven-step pipeline is unchanged; the interface is not.

## Running

Two processes. From the repository root:

```bash
# once
cd frontend && npm install && cd ..

# every time
python run_dev.py
```

Or start them separately:

```bash
python -m uvicorn backend.main:app --reload --port 8000
cd frontend && npm run dev
```

The app is at <http://localhost:5173>, the API docs at <http://127.0.0.1:8000/docs>.
Vite proxies `/api` (including the training WebSocket) to the backend, so the browser
sees one origin.

## Layout

```
src/
├── api/client.ts        Typed API client, session handling, WebSocket URL
├── store/pipeline.ts    Step order, lock rules, completion, auto-advance
├── theme/               Design tokens (light + dark) and the theme provider
├── components/          Shell, primitives, charts, RecommendationCard
└── steps/               One screen per pipeline module
```

### Design system

Every colour, space and type size is a CSS custom property in `theme/tokens.css`. No
component declares a raw hex value, which is what keeps eleven screens looking like one
application, and what makes the light/dark switch a change of eleven variables rather
than a second stylesheet.

The theme follows the OS by default and remembers an explicit choice. Charts read their
palette from the same tokens, so they re-colour with the toggle.

### Navigation

The sidebar groups the modules as Dataset → Explore → Prepare → Build → Predict →
Explain → Export. Each step declares its prerequisites in `store/pipeline.ts`; a step
whose prerequisites are unmet is disabled with the blocking step named in a tooltip.

Completing a step advances to the next one automatically. The toggle at the bottom of
the sidebar turns that off, and the sidebar remains navigable either way.

### Session state

The backend holds pipeline state in memory, keyed by a session id the client stores in
`sessionStorage`. Changing an upstream step invalidates everything downstream — the same
cascade the Streamlit app enforced through `reset_downstream`.

Because sessions are process-local and expire, a tab is the right lifetime for the id: a
second tab gets its own pipeline, and a restored id pointing at a dead session would only
produce confusing errors.
