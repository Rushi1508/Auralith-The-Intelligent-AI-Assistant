# Deploy on Oracle Cloud Always Free

One ARM VM runs the whole stack (API, UI, Postgres, Caddy for HTTPS) with Gemini as the generator. Cost: $0, as long as you stay inside the Always Free limits.

Always Free Ampere A1 (as of Nov 2026): about **2 OCPU and 12 GB RAM** in total (1,500 OCPU-hours and 9,000 GB-hours a month). Oracle halved it from 4/24 in 2026 and can disable instances that go over, so check the [Always Free resources page](https://docs.oracle.com/iaas/Content/FreeTier/resourceref.htm) before you create the VM.

## 1. Create the VM

1. Sign up at cloud.oracle.com/free. A card is asked for and not charged. Pick a home region close to your users; you can't change it later.
2. Compute → Instances → Create instance:
   - Image: **Canonical Ubuntu 24.04** (aarch64).
   - Shape: **VM.Standard.A1.Flex**, **2 OCPU, 12 GB**. If you see "Out of capacity", try another availability domain or try again later.
   - Networking: keep "Assign a public IPv4 address". Upload or download an SSH key.
   - Boot volume: the default ~47 GB is enough (the free total is 200 GB).
3. Open the web ports: Networking → Virtual cloud networks → your VCN → Security Lists → Default → **Add ingress rules**: source `0.0.0.0/0`, TCP, destination ports `80,443`.

## 2. Prepare the VM

```bash
ssh ubuntu@<public-ip>
# Oracle's Ubuntu image ships an iptables REJECT rule; let web traffic in and keep it across reboots
sudo iptables -I INPUT 6 -p tcp -m multiport --dports 80,443 -m state --state NEW -j ACCEPT
sudo netfilter-persistent save
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && exit   # log back in so the group applies
```

## 3. Configure

```bash
ssh ubuntu@<public-ip>
git clone https://github.com/<you>/<repo>.git auralith && cd auralith
cp .env.example .env
```

Edit `.env` and set:

| Variable | Value |
|---|---|
| `AURALITH_DOMAIN` | `<ip-with-dashes>.sslip.io`, e.g. `141-145-1-2.sslip.io` (free; no domain to buy), or your own domain with an A record for both `@` and `api` |
| `GEMINI_API_KEY` | free key from aistudio.google.com |
| `POSTGRES_PASSWORD` | `openssl rand -hex 24` |
| `AURALITH_API_KEYS` | your own admin key; see below |
| `AURALITH_ADMINS` | your user name, e.g. `alice` |

```bash
# builds the image (10-20 min the first time), then prints a raw key and its .env entry
docker compose build api
docker compose run --rm --no-deps api python -m auralith.registry create-key alice
```

Put the entries in `.env`, comma-separated: `AURALITH_API_KEYS=alice:<hash>,bob:<hash>`.

After startup, add everyone else at runtime (no restart). Each response shows the raw key once:

```bash
curl -s -X POST https://api.<AURALITH_DOMAIN>/admin/keys -H "X-API-Key: <alice key>" -H "Content-Type: application/json" -d '{"user":"carol"}'
curl -s https://api.<AURALITH_DOMAIN>/admin/keys -H "X-API-Key: <alice key>"                 # list
curl -s -X DELETE https://api.<AURALITH_DOMAIN>/admin/keys/<id> -H "X-API-Key: <alice key>"  # revoke
```

Give each person their raw key. They type it into the UI sidebar or send it as `X-API-Key`.

`POSTGRES_PASSWORD` is applied only when the database is first created. Changing it later needs `ALTER USER` inside Postgres, or a fresh `pgdata` volume.

## 4. Start

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api   # Ctrl+C when "Application startup complete"
```

Then:

- UI: `https://<AURALITH_DOMAIN>`
- API: `https://api.<AURALITH_DOMAIN>/health` should show `"status": "ok"` and `"auth": true`; interactive docs are at `/docs`.

Only Caddy (ports 80 and 443) is reachable from outside. Postgres, the API and the UI ports aren't published. Image mode works when the generator is Gemini; with Ollama it needs a vision model (`AURALITH_VISION_MODEL`, default `llava`) pulled on the server.

## Operate

```bash
alias dc='docker compose -f docker-compose.yml -f docker-compose.prod.yml'
dc ps                                   # health of every service
dc logs -f api                          # JSON access log: rid, path, status, ms, user
git pull && dc up -d --build            # update
dc up -d api                            # after editing .env; restart doesn't re-read it
```

Back up nightly (run `crontab -e` and add the line below). Keep a copy off the VM too, because losing the VM loses `pgdata`:

```
0 3 * * * cd ~/auralith && mkdir -p ~/backups && docker compose exec -T postgres pg_dump -U auralith auralith | gzip > ~/backups/auralith-$(date +\%F).sql.gz && find ~/backups -mtime +14 -delete
```

Restore: `gunzip -c backup.sql.gz | dc exec -T postgres psql -U auralith auralith`.
