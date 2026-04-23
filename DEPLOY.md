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
| `CORS_ORIGINS` | No | Comma-separated allowed origins (default: `*`) |
| `TOGETHER_TTS_BASE_URL` | No | Custom base URL for Together AI dedicated endpoints |

## Architecture

```
Internet → Caddy (:80/:443, auto-HTTPS) → Violin API (:8000) → Together AI
                                            ↕
                                       jobs/ volume (persistent)
```

- **Caddy** handles TLS termination and reverse proxying.
- **Violin** runs as a single uvicorn process with a thread pool for concurrent jobs.
- **jobs/** is a Docker named volume so job data persists across container rebuilds.
