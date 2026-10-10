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

The project first ran on a free tier that spun down after 15 minutes of inactivity and
capped memory at 512 MB. That failed all three, and produced slow uploads, missing
visualizations, "Failed to fetch", and spurious "Complete the upload step first" errors.
It now runs on one EC2 instance, set up as described below.

## Backend on one small VM

A 2 GB / 2 vCPU machine hosts this API — and a second project's API alongside it —
behind Caddy, which handles TLS automatically. On AWS that is the $12/month Lightsail plan
(public IPv4, 60 GB disk and transfer included) or an EC2 `t3.small` (about $21/month once
the IPv4 address and disk are added). Do not use a 512 MB plan: that is the ceiling this
move exists to escape.

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
| `LLM_API_KEY` | Optional. Powers the recommendation planner and the plain-English explanations. Without it the rule-based advisors recommend and explanations fall back to their own text. |
| `LLM_MODEL` | Leave as the default. If narration starts failing, the id has probably been retired — run `python scripts_list_models.py` to see what the provider currently serves. |
| `SMARTML_MEMORY_BUDGET_MB` | Resident memory at which training jobs are refused rather than risking an out-of-memory kill. `1300` for a 2 GB machine; keep it below the container limit in `docker-compose.yml`. The code default is 512, a cautious value for a small host. |
| `SMARTML_TRAINING_THREADS` | Threads each model may train on. `2` for a 2 vCPU machine. The code default is 1. |

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

### Running it only when needed (EC2 stop and start)

An EC2 instance that is stopped is not billed for compute, only for its disk (about $2 a
month for 20 GB). That suits a project that is demonstrated occasionally rather than used
all day. Lightsail does not work this way: its plans are billed while stopped.

The one thing that breaks is the address. A stopped instance gives up its public IP and
gets a different one when it starts, while the frontend is built with one fixed API
address. The `duckdns` service in `docker-compose.yml` closes that gap for free:

1. Create a subdomain at <https://www.duckdns.org> and copy the token shown there.
2. In `.env`, set `DUCKDNS_SUBDOMAIN` and `DUCKDNS_TOKEN`, and set `API_DOMAIN` to
   `<subdomain>.duckdns.org`.
3. `docker compose up -d --build` as usual.

From then on the instance re-registers its own IP within seconds of booting, and again
every five minutes. Every service has `restart: unless-stopped` and Docker starts at boot,
so starting the instance is the whole procedure:

- **To use it:** start the instance in the EC2 console and wait about two minutes.
- **When finished:** stop the instance. Do not terminate it; that deletes the disk.

The TLS certificate and the session checkpoints live in Docker volumes on the disk, so they
survive a stop. While the instance is off, the frontend reports that it cannot reach the
server, which is accurate.

An Elastic IP is the alternative to DuckDNS, but AWS bills it by the hour even while the
instance is stopped (about $3.60 a month), which costs more than the stopped instance.

## Verifying a deployment

1. `curl https://$API_DOMAIN/api/health` returns `{"status":"ok"}`.
2. Load the frontend; the browser console shows no CORS errors. If it does,
   `SMARTML_ALLOWED_ORIGINS` does not match the origin the browser is actually sending.
3. Upload a dataset, then `docker compose restart smartml-api`, then reload the page. The
   pipeline rail should still show the completed steps rather than demanding a new upload.
4. Run the pipeline through to Download without a 400 or 409.
