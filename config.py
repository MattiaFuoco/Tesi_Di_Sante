# ==========================================
# LIBRERIE DI SISTEMA E CALCOLO (NATIVE PYTHON)
# ==========================================
import os
import time
import subprocess
import re
import signal

# ==========================================
# 1. SPAZIO DI RICERCA GLOBALE
# ==========================================
CACHE_SIZES = [0.25, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0]
JOURNAL_INTERVALS = [1, 100, 500]
COMPRESSORS = ["none", "snappy", "zstd"]
DISTRIBUTIONS = ["uniform", "zipfian", "latest"]

# ==========================================
# 2. BUDGET GLOBALE PER GLI ALGORITMI
# ==========================================
MAX_EVALUATIONS = 90
# ==========================================
# 3. PARAMETRI DEL BENCHMARK
# ==========================================
REPETITIONS = 5
FIXED_THREADS = 16
OPS_PER_THREAD = 60000
PRELOAD_DOCS = 220000

# ────────────────────────────── SWITCH AMBIENTE ──────────────────────────
IS_SERVER = os.environ.get("BENCHMARK_ENV", "wsl") == "server"
USE_DIRECT_IO = IS_SERVER

# Percorsi interni al container (unico container, tutto dentro)
DATA_BASE_PATH = "/data/db"       # /data/db/none, /data/db/snappy, /data/db/zstd
MONGO_LOG_PATH = "/var/log/mongodb"
MONGO_URI      = "mongodb://localhost:27017/"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YCSB_PATH   = os.path.join(CURRENT_DIR, "ycsb", "bin", "ycsb.sh")

# ==========================================
# 4. DIRECTORY DI LAVORO
# ==========================================
# IMPORTANTE: get_master_dir() viene usato dai file algoritmi per leggere 
# sempre il valore CORRENTE della variabile d'ambiente, non il valore al momento dell'import
def get_master_dir():
    """Restituisce SEMPRE il valore CORRENTE di BENCHMARK_MASTER_DIR."""
    return os.environ.get("BENCHMARK_MASTER_DIR", os.getcwd())

master_dir   = os.environ.get("BENCHMARK_MASTER_DIR", os.getcwd())
WORKLOAD_TYPE = os.environ.get("WORKLOAD_TYPE", "A")

# Processo mongod globale (uno alla volta)
_mongod_process = None

# ==========================================
# 🛠️ HELPER INTERNI
# ==========================================
def _data_dir(compressor):
    """Directory dati dedicata per ogni compressore, dentro il container."""
    return os.path.join(DATA_BASE_PATH, compressor)

def _ensure_dirs():
    """Crea le directory dati e log se non esistono."""
    os.makedirs(MONGO_LOG_PATH, exist_ok=True)
    for comp in COMPRESSORS:
        os.makedirs(_data_dir(comp), exist_ok=True)

# ==========================================
# 🛠️ CORE API 1: GESTIONE MONGOD INTERNO
# ==========================================
def _stop_mongod():
    """Ferma mongod se è in esecuzione."""
    global _mongod_process
    if _mongod_process is not None:
        try:
            _mongod_process.send_signal(signal.SIGTERM)
            _mongod_process.wait(timeout=30)
        except Exception:
            _mongod_process.kill()
        _mongod_process = None
    # Fallback: kill di eventuali mongod rimasti
    subprocess.run(["pkill", "-f", "mongod"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

def _start_mongod(compressor, cache_gb=1.0, journal_ms=100):
    """Avvia mongod con i parametri specificati sulla directory del compressore."""
    global _mongod_process
    _stop_mongod()

    data_dir = _data_dir(compressor)
    log_file = os.path.join(MONGO_LOG_PATH, f"mongod_{compressor}.log")

    cmd = [
        "nocache", "mongod",
        "--dbpath", data_dir,
        "--logpath", log_file,
        "--port", "27017",
        "--wiredTigerCacheSizeGB", str(cache_gb),
        "--wiredTigerCollectionBlockCompressor", compressor,
        "--setParameter", f"journalCommitInterval={journal_ms}",
    ]

    if USE_DIRECT_IO:
        cmd += ["--wiredTigerEngineConfigString", "direct_io=[data,log]"]
    else:
        cmd += ["--wiredTigerEngineConfigString", "direct_io=[data]"]

    _mongod_process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Attende che mongod sia pronto
    for _ in range(30):
        result = subprocess.run(
            ["mongosh", "--quiet", "--eval", "db.adminCommand('ping')"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise Exception(f"❌ mongod ({compressor}) non si è avviato.")

def stop_all_containers_keep_data():
    """Compatibilità con master_runner — ferma mongod senza cancellare dati."""
    _stop_mongod()

def full_teardown():
    """💥 Ferma mongod e cancella tutti i dati per ripartire pulito."""
    _stop_mongod()
    for comp in COMPRESSORS:
        subprocess.run(["rm", "-rf", _data_dir(comp)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.makedirs(_data_dir(comp), exist_ok=True)

def drop_os_cache():
    """Svuota la Linux Page Cache.
    Siamo dentro il container con --privileged e /proc host montato,
    quindi echo 3 colpisce direttamente il kernel host reale.
    """
    subprocess.run(
        ["sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

# ==========================================
# 🛠️ CORE API 2: SETUP YCSB E MEGA-LOAD
# ==========================================
def setup_and_load_data(compressor):
    """Inserisce i documenti nel database tramite YCSB."""
    load_workload_file = os.path.join(
        CURRENT_DIR, "ycsb", "workloads", f"workload{WORKLOAD_TYPE.lower()}"
    )
    cmd_load = [
        "sh", YCSB_PATH, "load", "mongodb", "-s",
        "-P", load_workload_file,
        "-p", f"mongodb.url={MONGO_URI}",
        "-p", f"recordcount={PRELOAD_DOCS}",
        "-p", "fieldlength=1000",
        "-threads", str(FIXED_THREADS)
    ]
    print(f"\n   [LOAD] Scrittura di {PRELOAD_DOCS} documenti "
          f"(Compressione: {compressor.upper()}) in corso...")
    subprocess.run(cmd_load, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def global_setup_database():
    """🔥 Mega-Load iniziale: crea i 3 dataset una sola volta."""
    print(f"\n{'='*70}")
    print(f"🚀 FASE 1: TRIPLO MEGA-LOAD (3 DB da {PRELOAD_DOCS} record)")
    print(f"{'='*70}")

    _ensure_dirs()
    full_teardown()

    for comp in COMPRESSORS:
        print(f"\n---> Preparazione: {comp.upper()} <---")
        _start_mongod(comp)
        setup_and_load_data(comp)

        # fsync per garantire flush su disco
        subprocess.run(
            ["mongosh", "--quiet", "--eval", "db.adminCommand({fsync: 1})"],
            check=True
        )
        print(f"💾 fsync completato.")
        time.sleep(15)
        print(f"✅ Mega-Load {comp.upper()} completato.")
        _stop_mongod()

    print("\n✅ Tutti e 3 i dataset sono pronti. Avvio benchmark!")

# ==========================================
# 🛠️ CORE API 3: ESECUZIONE BENCHMARK
# ==========================================
def restart_mongodb_for_run(cache_gb, compressor, journal_ms):
    """Riavvia mongod con i parametri di test sul dataset del compressore."""
    _start_mongod(compressor, cache_gb=cache_gb, journal_ms=journal_ms)

def run_benchmark(distribution):
    """Esegue la fase RUN di YCSB."""
    wl_file = os.path.join(
        CURRENT_DIR, "ycsb", "workloads", f"workload{WORKLOAD_TYPE.lower()}"
    )
    cmd_run = [
        "sh", YCSB_PATH, "run", "mongodb", "-s",
        "-P", wl_file,
        "-p", f"mongodb.url={MONGO_URI}",
        "-p", f"requestdistribution={distribution}",
        "-p", f"operationcount={OPS_PER_THREAD * FIXED_THREADS}",
        "-p", "fieldlength=1000",
        "-threads", str(FIXED_THREADS)
    ]
    process = subprocess.run(cmd_run, capture_output=True, text=True)

    if process.returncode != 0:
        print(f"\n❌ ERRORE YCSB: {process.stderr}")
        return 0.0

    match = re.search(
        r"\[OVERALL\], Throughput\(ops/sec\), ([0-9.]+)", process.stdout
    )
    return float(match.group(1)) if match else 0.0

# ==========================================
# 🚀 MASTER API
# ==========================================
def execute_full_test(cache_gb, journal_ms, compressor, dist):
    """Cold Cache → avvio mongod → benchmark → stop."""
    throughputs = []
    durations   = []

    for r in range(REPETITIONS):
        start_time = time.time()

        print(f"   [Rep {r+1}/{REPETITIONS}] [CLEANUP] Drop OS Page Cache...", flush=True)
        drop_os_cache()

        print(f"   [Rep {r+1}/{REPETITIONS}] [START] mongod "
              f"(Comp={compressor.upper()}, Cache={cache_gb}GB, Journal={journal_ms}ms)...",
              flush=True)
        restart_mongodb_for_run(cache_gb, compressor, journal_ms)

        print(f"   [Rep {r+1}/{REPETITIONS}] [RUN] Benchmark...", flush=True)
        thr = run_benchmark(dist)

        _stop_mongod()

        dur = time.time() - start_time
        throughputs.append(thr)
        durations.append(dur)

    return sum(throughputs)/len(throughputs), sum(durations)/len(durations)