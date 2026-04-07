import subprocess
import time
import csv
import os
import datetime
import random
import string
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymongo

# ==========================================
# CONFIGURAZIONI GLOBALI
# ==========================================
CACHE_SIZES = [0.25,0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
DISTRIBUTIONS = ["uniform", "zipfian", "sequential"] 
JOURNAL_INTERVALS = [1, 100, 500] 
REPETITIONS = 5 

FIXED_THREADS = 6 # <-- Fissiamo i thread a 6 per tutte le run
OPS_PER_THREAD = 50000 # 50k operazioni per thread (WORKLOAD A: 25k read + 25k write)
PRELOAD_DOCS = 480000 # 480k documenti per saturare la cache
DOC_SIZE = 4096 
CONTAINER_RAM = 4 # GB

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "testdb"
COLLECTION_NAME = "testcol"
BASE_DIR = os.path.join(os.getcwd(), "Mattia_Final_Research")

# ==========================================
# FUNZIONI DI SETUP E PULIZIA
# ==========================================
def generate_random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def setup_docker_mongo(cache_size):
    # Rimozione preventiva ignorando gli errori se non esistono
    subprocess.run(["docker", "rm", "-f", "mongo-test"], capture_output=True)
    subprocess.run(["docker", "volume", "rm", "mongo_data"], capture_output=True)
    
    # NOTA: Assicurati che CONTAINER_RAM in alto sia = 4 (numero intero)
    mongo_cmd = [
        "docker", "run", "-d", 
        "--name", "mongo-test", 
        "-p", "27017:27017", 
        "--memory", f"{CONTAINER_RAM}g", 
        "--memory-swap", f"{CONTAINER_RAM}g",          
        "-v", "mongo_data:/data/db",         
        "mongo:7", 
        "--wiredTigerCacheSizeGB", str(cache_size),
        "--wiredTigerEngineConfigString=direct_io=[data,log]" 
    ]
    
    # Esecuzione diretta senza passare per la shell
    subprocess.run(mongo_cmd, check=True, capture_output=True)
    
    for _ in range(30):
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1000)
            client.admin.command('ping')
            client.close()
            return 
        except Exception:
            time.sleep(1)
    raise Exception("MongoDB non si è avviato!")

def preload_data(target_journal_ms):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    
    # Journal rilassato per velocizzare il caricamento
    client.admin.command("setParameter", 1, journalCommitInterval=500)
    
    batch_size = 10000
    print(f"   -> Preload rapido {PRELOAD_DOCS} docs in corso (batch: {batch_size})...")
    
    dummy_payload = generate_random_string(DOC_SIZE)
    
    for i in range(0, PRELOAD_DOCS, batch_size):
        end = min(i + batch_size, PRELOAD_DOCS)
        batch = [{"_id": j, "payload": dummy_payload} for j in range(i, end)]
        
        # Gestione duplicati residui nel preload
        try:
            col.insert_many(batch, ordered=False)
        except pymongo.errors.BulkWriteError:
            pass
        
        if end % 50000 == 0 or end == PRELOAD_DOCS:
            print(f"      - Caricati: {end}/{PRELOAD_DOCS}")
        del batch 

    # RIPRISTINO DEL JOURNAL TARGET DEL TEST (es. 1ms o 100ms o 500ms)
    client.admin.command("setParameter", 1, journalCommitInterval=target_journal_ms)
    client.admin.command("fsync", lock=False) 
    time.sleep(3) 
    client.close()

# ==========================================
# WORKLOAD 
# ==========================================
def worker_task(thread_id, client, distribution, num_threads):
    col = client[DB_NAME][COLLECTION_NAME]
    for i in range(OPS_PER_THREAD):
        if random.random() < 0.5:
            if distribution == "uniform":
                read_id = random.randint(0, PRELOAD_DOCS - 1)
            elif distribution == "zipfian":
                read_id = (np.random.zipf(1.99) - 1) % PRELOAD_DOCS
            else: 
                read_id = i % PRELOAD_DOCS
            col.find_one({"_id": int(read_id)})
        else:
            new_id = PRELOAD_DOCS + (thread_id * OPS_PER_THREAD) + i
            try:
                col.insert_one({"_id": new_id, "payload": generate_random_string(DOC_SIZE)})
            except pymongo.errors.DuplicateKeyError:
                pass

def run_workload(distribution, num_threads):
    client = MongoClient(MONGO_URI, maxPoolSize=num_threads)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for t in range(num_threads):
            executor.submit(worker_task, t, client, distribution, num_threads)
    client.close()
    duration = time.time() - start_time
    throughput = (num_threads * OPS_PER_THREAD) / duration
    return duration, throughput

# ==========================================
# GENERAZIONE GRAFICI (Throughput + Duration)
# ==========================================
def plot_results(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    sns.set_theme(style="whitegrid")
    
    # Creiamo il nostro dizionario di colori personalizzato
    # Puoi usare i nomi in inglese ("red", "blue", "green") o codici HEX (es. "#FF0000")
    custom_palette = {
        1: "red",
        100: "blue",
        500: "green"
    }
    
    # ------------------------------------------
    # 1. GRAFICO DEL THROUGHPUT
    # ------------------------------------------
    g_thr = sns.relplot(
        data=df, 
        x="cache_GB", 
        y="throughput", 
        hue="journal_ms",    
        col="distribution",  
        kind="line", 
        marker="o", 
        linewidth=2.5,
        palette=custom_palette, # <--- Usiamo la nostra palette personalizzata!
        facet_kws={'sharey': False} 
    )
    
    g_thr.set_axis_labels("Cache Size (GB)", "Throughput (ops/sec)")
    g_thr.set_titles("Distribution: {col_name}")
    g_thr.fig.subplots_adjust(top=0.85)
    g_thr.fig.suptitle(f"Scalabilità MongoDB - THROUGHPUT\n(Container RAM: {CONTAINER_RAM}GB | Threads: {FIXED_THREADS})", fontsize=16)
    
    plt.savefig(f"{output_prefix}_throughput_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------
    # 2. GRAFICO DEL TEMPO DI ESECUZIONE
    # ------------------------------------------
    g_dur = sns.relplot(
        data=df, 
        x="cache_GB", 
        y="duration",   
        hue="journal_ms",    
        col="distribution",  
        kind="line", 
        marker="X",     
        linewidth=2.5,
        palette=custom_palette, # <--- Applichiamo la stessa palette anche qui!
        facet_kws={'sharey': False} 
    )
    
    g_dur.set_axis_labels("Cache Size (GB)", "Tempo di Esecuzione (Secondi)")
    g_dur.set_titles("Distribution: {col_name}")
    g_dur.fig.subplots_adjust(top=0.85)
    g_dur.fig.suptitle(f"Scalabilità MongoDB - TEMPO DI ESECUZIONE\n(Container RAM: {CONTAINER_RAM}GB | Threads: {FIXED_THREADS})", fontsize=16)
    
    plt.savefig(f"{output_prefix}_duration_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# MAIN LOOP 
# ==========================================
def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Usiamo un nome fisso per permettere l'aggiunta di logica auto-resume futura se serve
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"results_journal_{ts}.csv")
    
    total_runs = len(CACHE_SIZES) * len(DISTRIBUTIONS) * len(JOURNAL_INTERVALS) * REPETITIONS
    current_run = 0

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["cache_GB", "distribution", "journal_ms", "repetition", "duration", "throughput"])

    print(f"🚀 PARTENZA: {total_runs} run totali. Metodologia: COLD START puro.")

    for c_gb in CACHE_SIZES:
        for dist in DISTRIBUTIONS:
            for j_ms in JOURNAL_INTERVALS: # <-- Cicliamo sui Journal
                for r in range(REPETITIONS):
                    current_run += 1
                    print(f"\n[{current_run}/{total_runs}] Parametri: C={c_gb}GB | D={dist} | J={j_ms}ms | R={r+1}")
                    
                    setup_docker_mongo(c_gb)
                    preload_data(j_ms) # Passiamo il target_journal alla fase di preload
                    
                    print(f"   -> ⚙️ TEST IN CORSO (Journal: {j_ms}ms, Threads: {FIXED_THREADS})...")
                    dur, thr = run_workload(dist, FIXED_THREADS) # Passiamo i thread fissi
                    
                    with open(csv_filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([c_gb, dist, j_ms, r+1, dur, thr])
                    print(f"   -> Workload completato: {thr:.2f} ops/sec in {dur:.2f}s")
                    
                    # Distruzione pulita
                    subprocess.run(["docker", "rm", "-f", "mongo-test"], capture_output=True)
                    subprocess.run(["docker", "volume", "rm", "mongo_data"], capture_output=True)

    print(f"\n📊 Generazione grafici in corso...")
    plot_results(csv_filename, os.path.join(BASE_DIR, f"analisys_{ts}"))
    print(f"✅ Fatto! Trovi i risultati e il grafico in: {BASE_DIR}")

if __name__ == "__main__":
    main()