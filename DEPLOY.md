# Deploying Violin to AWS EC2

## Prerequisites

- An AWS account
- A Together AI API key
- A domain name (optional but recommended for HTTPS)

## 1. Launch an EC2 instance

1. Open the [EC2 console](https://console.aws.amazon.com/ec2/) and click **Launch Instance**.
2. Choose **Ubuntu 24.04 LTS** (or Amazon Linux 2023).
3. Instance type: **t3.medium** or larger (translation jobs are IO-bound but concurrent workers need RAM).
4. Storage: **30 GB+** gp3 (job files — uploaded videos, outputs — need disk space).
5. Security group — allow inbound on:
   - **22** (SSH)
   - **80** (HTTP)
   - **443** (HTTPS)
6. Launch and note the public IP. Allocate an **Elastic IP** and associate it so the address persists across reboots.

## 2. Install Docker on the instance

SSH into the instance and run:

```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable docker
sudo usermod -aG docker $USER
# Log out and back in for group membership to take effect
```

## 3. Clone and configure

```bash
git clone https://github.com/shang-zhu/Violin.git
cd Violin

# Create your .env file
cp .env.example .env
# Edit .env and fill in:
#   TOGETHER_API_KEY=your_key_here
#   CORS_ORIGINS=https://yourdomain.com   (optional, defaults to *)
```

## 4. Configure your domain

Edit `Caddyfile` and replace `yourdomain.com` with your actual domain:

```
yourdomain.com {
    reverse_proxy violin:8000
}
```

Then point your domain's **DNS A record** to the EC2 Elastic IP. Caddy will automatically provision a Let's Encrypt TLS certificate once DNS propagates.

**No domain yet?** Use this Caddyfile for plain HTTP:

```
:80 {
    reverse_proxy violin:8000
}
```

## 5. Deploy

```bash
docker compose up -d --build
```

Check logs:

```bash
docker compose logs -f
```

The app is now available at `https://yourdomain.com` (or `http://<ec2-ip>` if using plain HTTP).

## 6. Maintenance

```bash
# Pull latest code and rebuild
git pull
docker compose up -d --build

# View logs
docker compose logs -f violin
docker compose logs -f caddy

# Stop
docker compose down

# Stop and remove job data volume
docker compose down -v
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TOGETHER_API_KEY` | Yes | Together AI API key |
| `OPENAI_API_KEY` | Optional | Only needed when translation provider is OpenAI |
| `ELEVENLABS_API_KEY` | Optional | Only needed when TTS provider is ElevenLabs |
| `CORS_ORIGINS` | No | Comma-separated allowed origins (default: `*`) |
| `TOGETHER_TTS_BASE_URL` | No | Custom base URL for Together AI dedicated endpoints |

## Production overrides (`config/prod.yaml`)

The Dockerfile starts the API with `--config config/prod.yaml`. Override anything from `config/default.yaml` there — common tweaks:

```yaml
merge_video:
  workers: 4                 # match the host's vCPU count; each ffmpeg ~300MB RAM
api:
  max_workers: 1             # serialize jobs so the multiplier doesn't blow RAM
  url_upload: false          # disable yt-dlp URL ingest if running a public demo
  max_duration_seconds: 1800 # reject videos longer than 30 min
  max_file_size_mb: 500      # reject uploads larger than 500 MB
```

Memory headroom: `api.max_workers × merge_video.workers × ~300MB` should comfortably fit RAM. On `c7g.xlarge` (8 GB), `1 × 4 = 1.2 GB` is safe.

## Architecture

```
Internet → Caddy (:80/:443, auto-HTTPS) → Violin API (:8000) → Together AI / OpenAI / ElevenLabs
                                            ↕
                                       jobs/ volume (persistent — uploads, outputs, stats.sqlite)
```

- **Caddy** handles TLS termination and reverse proxying.
- **Violin** runs as a single uvicorn process with a thread pool for concurrent jobs.
- **jobs/** is a Docker named volume — uploads, outputs, and the stats database all live here. Survives container rebuilds.

## Stats / cost tracking

Every finished job writes one row to `jobs/stats.sqlite` (job id, language, providers used, tokens, characters, audio seconds, estimated cost, wall time). No video content, IPs, or API keys are stored.

To inspect:

```bash
# Quick aggregates
docker compose exec violin sqlite3 jobs/stats.sqlite \
  "SELECT target_language, COUNT(*) AS jobs,
          printf('\$%.2f', SUM(total_cost_usd)) AS cost
   FROM jobs GROUP BY 1 ORDER BY jobs DESC;"

# Last 10 jobs
docker compose exec violin sqlite3 jobs/stats.sqlite \
  "SELECT datetime(created_at, 'unixepoch'), target_language, status,
          printf('%.1fmin', audio_seconds/60.0),
          printf('\$%.4f', total_cost_usd)
   FROM jobs ORDER BY created_at DESC LIMIT 10;"

# Pull the file to your laptop and explore with a GUI (e.g. DB Browser for SQLite)
docker compose cp violin:/app/jobs/stats.sqlite ./stats.sqlite
```

The file is **persistent** — it survives `docker compose down && up`, container rebuilds, and EC2 reboots. It is only deleted by `docker compose down -v` or `docker volume rm jobs_data`.

## Migrating to a different EC2 instance

Common when scaling up (e.g. `t3.medium` → `c7g.xlarge`):

1. **Launch the new instance** in the same region; reuse the existing security group and key pair. For ARM-based families (`c7g`, `c7gn`, …) pick the **Arm64 / Graviton** AMI.
2. **Install Docker** (same commands as §2).
3. **Clone and configure** (same as §3, copy `.env` from the old box):
   ```bash
   scp -i key.pem ubuntu@OLD_IP:~/Violin/.env .
   scp -i key.pem .env ubuntu@NEW_IP:~/Violin/
   ```
4. **Deploy** on the new box (`docker compose up -d --build`).
5. **Switch DNS** in Route 53: change the A record's value to the new IP. Lower the TTL to 60s a day ahead of time so propagation is near-instant.
6. **Decommission the old box** once you've watched the new one for a day or two: EBS snapshot first, then terminate.

If you don't have an Elastic IP and you're using a domain (Route 53 A record), DNS is the only thing that needs to move. No EIP required.

## Backups

The only state worth backing up is `jobs/stats.sqlite` (the on-disk uploads and outputs are auto-deleted after `api.job_ttl_hours`, default 24h, so they aren't worth backing up).

```bash
# Manual snapshot (run from local machine)
docker compose --context REMOTE cp violin:/app/jobs/stats.sqlite ./stats.sqlite.$(date +%F)
```

Or use **AWS EBS snapshots** (Lifecycle Manager → daily, retain 7) to back up the whole disk. Requires `iam:GetRole` / `iam:CreateRole` permission to create the default DLM role; otherwise create a snapshot manually from EC2 → Volumes → Actions → Create snapshot.

## Health check

```
GET /health   → {"ok": true}   (200, accepts HEAD too)
```

Use this with UptimeRobot, CloudWatch, etc. The root path `/` only supports GET, so HEAD-only monitors should target `/health`.
