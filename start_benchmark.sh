#!/bin/bash

echo "===================================================="
echo "  INIZIALIZZAZIONE BENCHMARK MONGODB + YCSB  "
echo "===================================================="

# 1. Controllo permessi di root
if [ "$EUID" -ne 0 ]; then
  echo "Errore: Esegui questo script con sudo!"
  exit 1
fi

# ──────────────────────────────────────────────────────────────
# 2. RILEVAMENTO AMBIENTE
# ──────────────────────────────────────────────────────────────
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Ambiente rilevato: WSL"
    export BENCHMARK_ENV=wsl
else
    echo "Ambiente rilevato: Server Debian"
    export BENCHMARK_ENV=server
fi

# ──────────────────────────────────────────────────────────────
# 3. YCSB: Download automatico se non presente
# ──────────────────────────────────────────────────────────────
YCSB_VERSION="0.17.0"
YCSB_DIR="ycsb"
YCSB_TARBALL="ycsb-mongodb-binding-${YCSB_VERSION}.tar.gz"
YCSB_URL="https://github.com/brianfrankcooper/YCSB/releases/download/${YCSB_VERSION}/${YCSB_TARBALL}"

if [ ! -f "${YCSB_DIR}/bin/ycsb.sh" ]; then
    echo "YCSB non trovato. Download in corso (versione ${YCSB_VERSION})..."
    if command -v wget &> /dev/null; then
        wget -q "$YCSB_URL" -O "$YCSB_TARBALL"
    else
        curl -sL "$YCSB_URL" -o "$YCSB_TARBALL"
    fi
    echo "Estrazione archivio..."
    tar -xzf "$YCSB_TARBALL"
    mv "ycsb-mongodb-binding-${YCSB_VERSION}" "$YCSB_DIR"
    rm "$YCSB_TARBALL"
    echo "YCSB installato in ./${YCSB_DIR}/"
else
    echo "YCSB già presente."
fi

# ──────────────────────────────────────────────────────────────
# 4. BUILD IMMAGINE (rebuild automatica se i file sono cambiati)
# ──────────────────────────────────────────────────────────────
# Calcoliamo un hash di tutti i file Python + Dockerfile + requirements.
# Se l'hash è diverso dall'ultima build → rebuild automatica.
# Il file .docker_build_hash va aggiunto sia a .dockerignore che a .gitignore.
HASH_FILE=".docker_build_hash"
CURRENT_HASH=$(find . -maxdepth 1 \( -name "*.py" -o -name "Dockerfile" -o -name "requirements.txt" \) \
    | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)

LAST_HASH=""
if [ -f "$HASH_FILE" ]; then
    LAST_HASH=$(cat "$HASH_FILE")
fi

if ! docker image inspect benchmark-all-in-one:latest &> /dev/null; then
    echo "Immagine non trovata. Build in corso..."
    docker build -t benchmark-all-in-one:latest .
    echo "$CURRENT_HASH" > "$HASH_FILE"
    echo "Immagine pronta."
elif [ "$CURRENT_HASH" != "$LAST_HASH" ]; then
    echo "File modificati rilevati — rebuild automatica in corso..."
    docker rmi benchmark-all-in-one:latest 2>/dev/null
    docker build -t benchmark-all-in-one:latest .
    echo "$CURRENT_HASH" > "$HASH_FILE"
    echo "Immagine aggiornata."
else
    echo "Nessuna modifica rilevata — uso immagine esistente."
fi

# ──────────────────────────────────────────────────────────────
# 5. CARTELLA RISULTATI SULL'HOST
# ──────────────────────────────────────────────────────────────
RESULTS_DIR="$(pwd)/results"
mkdir -p "$RESULTS_DIR"
export RESULTS_DIR
echo "Risultati salvati in: $RESULTS_DIR"

# ──────────────────────────────────────────────────────────────
# 6. AVVIO CONTAINER UNICO
# ──────────────────────────────────────────────────────────────
echo "Avvio del Benchmark (ENV=$BENCHMARK_ENV)..."

docker run --rm \
    --name benchmark-all-in-one \
    --privileged \
    --memory "10g" \
    --memory-swap "10g" \
    -e BENCHMARK_ENV="$BENCHMARK_ENV" \
    -e PYTHONUNBUFFERED=1 \
    -e RESULTS_DIR="/app/results" \
    -v "$(pwd)/results":/app/results \
    -v "$(pwd)/ycsb":/app/ycsb \
    benchmark-all-in-one:latest

echo "Benchmark completato. Risultati in: $RESULTS_DIR"