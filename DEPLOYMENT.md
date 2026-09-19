# Deploying SmartML Studio

The frontend is a static bundle on Vercel. The backend is a single long-lived Python
process. That split is deliberate, and the backend's shape constrains where it can run.

## Why the backend is not serverless

Between requests the backend holds the uploaded DataFrame, the fitted transformers and the
trained estimators in memory, runs training on a background thread, and streams training
progress over a WebSocket. It needs:

- **One process that stays up.** Sleep it and every in-flight pipeline dies.
- **Room to work.** A 50 MB upload becomes a DataFrame, a train/test split, and one fitted
  model per algorithm selected. 512 MB is not enough; 2 GB is comfortable.
- **A writable volume that outlives the container.** Session checkpoints
  (`backend/core/persistence.py`) are what let a restart resume instead of sending the user
  back to the upload step.

A free tier that spins down after 15 minutes of inactivity and caps memory at 512 MB fails
all three, which is what produced the slow uploads, missing visualizations, "Failed to
fetch", and spurious "Complete the upload step first" errors in production.

## Backend on one small VM (recommended)

An `t3.small` (2 GB, 2 vCPU) or equivalent hosts this API — and a second project's API
alongside it — behind Caddy, which handles TLS automatically.

### 1. Point DNS at the machine

Create an `A` record for the API hostname (for example `smartml-api.example.com`)
targeting the instance's public IP. Caddy cannot issue a certificate until this resolves.
Open inbound ports 80 and 443 in the security group.

### 2. Install Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"   # log out and back in for this to take effect
```

### 3. Configure

```bash
git clone <this repository> smartml-studio
cd smartml-studio
cp .env.example .env
```

Fill in `.env`:

| Variable | Purpose |
| --- | --- |
| `API_DOMAIN` | Hostname Caddy serves the API on. Must already resolve to this machine. |
| `SMARTML_ALLOWED_ORIGINS` | The frontend's origin, e.g. `https://smartml-studio.vercel.app`. This project's Vercel preview URLs are already matched by pattern in `backend/main.py`. |
| `LLM_API_KEY` | Optional. Without it, explanations fall back to the advisors' own text. |
| `LLM_MODEL` | Leave as the default. If narration starts failing, the id has probably been retired — run `python scripts_list_models.py` to see what the provider currently serves. |

### 4. Start

```bash
docker compose up -d --build
docker compose logs -f          # watch the first certificate issuance
curl https://$API_DOMAIN/api/health
```

### 5. Point the frontend at it

Set `VITE_API_BASE` to `https://<API_DOMAIN>/api` in the Vercel project's environment
variables, then **trigger a redeploy**. Vite inlines `import.meta.env` at build time, so
changing the variable without rebuilding has no effect.

### Operations

```bash
docker compose ps                        # health of both services
docker compose logs -f smartml-api       # application logs
docker compose up -d --build             # deploy a new version
docker compose restart smartml-api       # sessions survive this, by design
docker volume ls | grep smartml          # the session and log volumes
```

Session checkpoints live in the `smartml-sessions` volume and are pruned automatically
after four hours of inactivity (`SESSION_TTL_SECONDS` in `backend/core/session.py`). The
container is capped at 1400 MB so a runaway training job cannot take the proxy down with
it; if training a large dataset gets OOM-killed, raise that limit in `docker-compose.yml`
and use an instance with more memory.

## Keeping the existing free-tier deployment

`render.yaml` is still in the repository and still works, with two caveats worth knowing:

- Checkpoints survive a process restart but **not** a spin-down, because the container's
  filesystem is discarded with it. Sessions are still lost after an idle period.
- 512 MB constrains dataset size regardless of the 50 MB upload cap.

The frontend handles both cases better than it used to — it warms the backend on load and
says so, retries idempotent requests, and explains a lost session rather than silently
resetting — but it cannot manufacture memory or stop a spin-down.

## Verifying a deployment

1. `curl https://$API_DOMAIN/api/health` returns `{"status":"ok"}`.
2. Load the frontend; the browser console shows no CORS errors. If it does,
   `SMARTML_ALLOWED_ORIGINS` does not match the origin the browser is actually sending.
3. Upload a dataset, then `docker compose restart smartml-api`, then reload the page. The
   pipeline rail should still show the completed steps rather than demanding a new upload.
4. Run the pipeline through to Download without a 400 or 409.
