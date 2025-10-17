# Deployment Guide — Micro_Credit_V_2.0

This document explains a reliable way to deploy the backend so it runs 24/7 on an Ubuntu VPS (DigitalOcean, AWS EC2, Linode, etc.) using Docker and systemd.

Summary of chosen approach
- Host: any Ubuntu 22.04 LTS (or similar) VPS (DigitalOcean/AWS Lightsail/Hetzner). Rationale: inexpensive, reliable, and gives full control.
- Containerization: Docker + docker compose. Rationale: isolates dependencies (TensorFlow + scikit-learn), portable, easy to run locally and on server.
- Process manager: systemd unit that launches docker compose. Ensures auto-start on boot and restarts on failure.

Files added
- `Dockerfile` — builds an image with dependencies and runs the app via gunicorn.
- `.dockerignore` — excludes local/large files from the image context.
- `docker-compose.yml` — helper for local and server runs.
- `.env.example` — example env vars; do NOT commit real secrets.
- `deploy/micro_credit.service` — systemd template to run docker-compose as a service.

Provisioning a new Ubuntu server (quick steps)
1. Create a VPS (Ubuntu 22.04 LTS). Ensure a user with sudo is available.
2. SSH into the server and update:

```bash
sudo apt update && sudo apt upgrade -y
```

3. Install Docker and docker-compose (official steps):

```bash
# Install prerequisites
sudo apt install -y ca-certificates curl gnupg lsb-release

# Add Docker GPG key and repo
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add your deploy user to the docker group (replace $USER)
sudo usermod -aG docker $USER
```

Deployment steps (recommended)
1. Copy repo to server (git clone or rsync). Example using git:

```bash
cd /opt
sudo git clone https://github.com/<your-repo>/Micro_Credit_V_2.0.git micro_credit
sudo chown -R $USER:$USER micro_credit
cd micro_credit
cp .env.example .env
# Edit .env and set SECRET_KEY and any other vars
```

2. (Optional) Place data files
- If you have `credit_risk_data.csv`, copy it into `/opt/micro_credit` so the app can auto-train.

3. Build and run with docker compose

```bash
docker compose build --pull
docker compose up -d
```

4. Create the systemd service (optional but recommended)

```bash
sudo cp deploy/micro_credit.service /etc/systemd/system/micro_credit.service
# Edit the service file paths if you deployed elsewhere
sudo systemctl daemon-reload
sudo systemctl enable micro_credit.service
sudo systemctl start micro_credit.service
sudo systemctl status micro_credit.service
```

Monitoring & logging
- The application logs are available from Docker: `docker compose logs -f` or `docker logs -f <container>`.
- Use external uptime monitoring (UptimeRobot / StatusCake) to poll `https://<your-domain-or-ip>/health` every 1-5 minutes.
- For more advanced monitoring: integrate with Prometheus + Grafana or cloud provider monitoring.

Healthcheck and automatic restart
- systemd `Restart=always` and Docker `restart: unless-stopped` together ensure the app restarts on crashes and reboots.
- The Dockerfile includes a /health endpoint healthcheck used by container managers.

Secrets and environment variables
- Keep secrets out of git. Use the VPS `.env` file (owner root or deploy user) and `EnvironmentFile` in systemd.
- For production, consider a secret manager (AWS Secrets Manager / HashiCorp Vault) and inject variables at runtime.

Verifying the deployment
1. Check container status:

```bash
docker compose ps
```

2. Check logs for errors:

```bash
docker compose logs -f
```

3. Health endpoint test:

```bash
curl -f http://localhost:5000/health
```

4. Run simple assess cycle (only after models trained):

```bash
curl -X POST http://localhost:5000/api/assess -H 'Content-Type: application/json' \
  -d '{"total_transactions":10000,"avg_transaction_amount":1500,"payment_consistency_score":90,"business_age_months":48,"digital_footprint_score":85}'
```

Backup & updates
- To update code: git pull, then `docker compose build && docker compose up -d`.
- Keep frequent backups of important data (csvs, saved models) if you persist them.

Further improvements (optional)
- Move model training to a separate worker and persist trained model artifacts (`joblib`/`h5`) in a mounted volume.
- Add TLS (Let's Encrypt) and nginx reverse proxy.
- Add proper structured logging and log rotation (filebeat/ELK or cloud logging).

Completion checklist
- [ ] Server provisioned
- [ ] Docker + compose installed
- [ ] Repo deployed to /opt/micro_credit
- [ ] `.env` created and secrets configured
- [ ] Docker image built and container running
- [ ] systemd service created and enabled
- [ ] Health endpoint monitored by external uptime service

If you want, I can perform the provisioning and deployment steps for you (I will need SSH access or a droplet created by you). Otherwise, follow the steps above and I can help debug any issue you encounter.
