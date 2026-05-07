FROM debian:bookworm-slim

# ── Dipendenze di sistema ──────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    default-jre \
    curl wget procps gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Installazione MongoDB 7 ────────────────────────────────────
RUN curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc \
    | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg && \
    echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] \
    https://repo.mongodb.org/apt/debian bookworm/mongodb-org/7.0 main" \
    > /etc/apt/sources.list.d/mongodb-org-7.0.list && \
    apt-get update && apt-get install -y mongodb-org && \
    rm -rf /var/lib/apt/lists/*

# ── Installazione nocache (compilato dai sorgenti GitHub) ──────
RUN apt-get update && apt-get install -y \
        git build-essential libattr1-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth=1 https://github.com/Feh/nocache.git /tmp/nocache \
    && cd /tmp/nocache && make && make install \
    && rm -rf /tmp/nocache

# ── Directory di lavoro ────────────────────────────────────────
WORKDIR /app

# ── Dipendenze Python ──────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# ── Copia progetto ─────────────────────────────────────────────
COPY . .

# ── Directory dati MongoDB e risultati ────────────────────────
RUN mkdir -p /data/db /app/results

CMD ["python3", "master_runner.py"]
