import subprocess
import time
import csv
import os
import datetime
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from adjustText import adjust_text
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
import pymongo

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURAZIONI GLOBALI
# ==========================================
CACHE_SIZES = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
JOURNAL_INTERVALS = [1, 100, 500]
DISTRIBUTIONS = ["uniform", "zipfian", "sequential"] 
REPETITIONS = 5 

FIXED_THREADS = 6 # <-- Fissiamo i thread a 6 per tutte le run
OPS_PER_THREAD = 50000 # 50k operazioni per thread (WORKLOAD A: 25k read + 25k write)
PRELOAD_DOCS = 480000 # # 480k documenti per saturare la cache
DOC_SIZE = 4096 
CONTAINER_RAM = 4 # GB

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "testdb"
COLLECTION_NAME = "testcol"

BASE_DIR = os.path.join(os.getcwd(), "Random Search Result")

# ==========================================
# PARAMETRI RANDOM SEARCH
# ==========================================
MAX_EVALUATIONS = 12  # Quanti punti esplorare a caso (Budget uguale al SA)

# ==========================================
# FUNZIONI DI SETUP E PULIZIA
# ==========================================
def generate_random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def setup_docker_mongo(cache_size):
    subprocess.run(["docker", "rm", "-f", "mongo-test"], capture_output=True)
    subprocess.run(["docker", "volume", "rm", "mongo_data"], capture_output=True)
    
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
    client.admin.command("setParameter", 1, journalCommitInterval=500)
    batch_size = 10000
    
    dummy_payload = generate_random_string(DOC_SIZE)
    for i in range(0, PRELOAD_DOCS, batch_size):
        end = min(i + batch_size, PRELOAD_DOCS)
        batch = [{"_id": j, "payload": dummy_payload} for j in range(i, end)]
        try:
            col.insert_many(batch, ordered=False)
        except pymongo.errors.BulkWriteError:
            pass
        del batch 

    client.admin.command("setParameter", 1, journalCommitInterval=target_journal_ms)
    client.admin.command("fsync", lock=False) 
    time.sleep(3) 
    client.close()

# ==========================================
# WORKLOAD E VALUTAZIONE
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

def evaluate_point(c_idx, j_idx, dist, csv_filename, step_num):
    c_gb = CACHE_SIZES[c_idx]
    j_ms = JOURNAL_INTERVALS[j_idx]
    
    print(f"\n   [Lancio Dado {step_num}/{MAX_EVALUATIONS}] Testo: Cache={c_gb}GB, Journal={j_ms}ms")
    
    throughputs, durations = [], []
    for r in range(REPETITIONS):
        print(f"      -> Ripetizione {r+1}/{REPETITIONS}...", end="", flush=True)
        setup_docker_mongo(c_gb)
        preload_data(j_ms)
        
        client = MongoClient(MONGO_URI, maxPoolSize=FIXED_THREADS)
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=FIXED_THREADS) as executor:
            for t in range(FIXED_THREADS):
                executor.submit(worker_task, t, client, dist, FIXED_THREADS)
        client.close()
        
        dur = time.time() - start_time
        thr = (FIXED_THREADS * OPS_PER_THREAD) / dur
        
        throughputs.append(thr)
        durations.append(dur)
        print(f" {thr:.2f} ops/sec")
        
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step_num, c_gb, dist, j_ms, r+1, dur, thr])
            
        subprocess.run(["docker", "rm", "-f", "mongo-test"], capture_output=True)
        subprocess.run(["docker", "volume", "rm", "mongo_data"], capture_output=True)
        
    avg_thr = np.mean(throughputs)
    print(f"   => Media Ottenuta: {avg_thr:.2f} ops/sec")
    return avg_thr

# ==========================================
# GENERAZIONE GRAFICI (RANDOM SEARCH)
# ==========================================
def plot_rs_master(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Random Search - Esplorazione Spaziale\n(Nessun percorso logico: i punti vengono testati casualmente)", fontsize=16, fontweight='bold')
    contour_plot = None 

    for i, dist in enumerate(DISTRIBUTIONS):
        ax = axes[i]
        df_dist = df[df['distribution'] == dist]
        if df_dist.empty: continue

        # Sfondo Topografico Interpolato
        df_grouped = df_dist.groupby(['cache_GB', 'journal_ms'])['throughput'].mean().reset_index()
        x_coords = df_grouped['cache_GB'].map(cache_map).values
        y_coords = df_grouped['journal_ms'].map(journal_map).values
        z_vals = df_grouped['throughput'].values

        grid_x, grid_y = np.mgrid[0:len(CACHE_SIZES)-1:100j, 0:len(JOURNAL_INTERVALS)-1:100j]
        grid_z = griddata((x_coords, y_coords), z_vals, (grid_x, grid_y), method='linear')
        
        contour_plot = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdYlGn', alpha=0.6)
        ax.contour(grid_x, grid_y, grid_z, levels=10, colors='black', linewidths=0.5, alpha=0.3)

        # Tracciamo i punti casuali (Senza frecce di collegamento!)
        df_steps = df_dist.groupby(['step', 'cache_GB', 'journal_ms'])['throughput'].mean().reset_index().sort_values('step')
        
        texts = []
        for _, row in df_steps.iterrows():
            px = cache_map[row['cache_GB']]
            py = journal_map[row['journal_ms']]
            step = int(row['step'])
            
            ax.scatter(px, py, color='white', s=150, zorder=8, edgecolors='black', linewidth=1.5)
            t = ax.text(px, py, f"T{step}", ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='black', zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1, alpha=0.9))
            texts.append(t)

        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.7))

        ax.set_xticks(range(len(CACHE_SIZES))); ax.set_xticklabels(CACHE_SIZES)
        ax.set_yticks(range(len(JOURNAL_INTERVALS))); ax.set_yticklabels(JOURNAL_INTERVALS)
        ax.set_xlabel("Cache Size (GB)"); ax.set_ylabel("Journal Interval (ms)")
        ax.set_title(f"Distribuzione: {dist}", fontsize=14)
        ax.set_xlim(-0.3, len(CACHE_SIZES)-0.7); ax.set_ylim(-0.3, len(JOURNAL_INTERVALS)-0.7)

    plt.tight_layout()
    fig.subplots_adjust(top=0.82, right=0.92) 
    if contour_plot:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7]) 
        fig.colorbar(contour_plot, cax=cbar_ax, orientation='vertical').set_label('Throughput Interpolato (ops/sec)')
    
    plt.savefig(f"{output_prefix}_RS_Map.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_rs_convergence(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Random Search - Convergence Plot\n(I salti sono casuali, la linea blu indica il miglior risultato trovato finora)", fontsize=16, fontweight='bold')

    for i, dist in enumerate(DISTRIBUTIONS):
        ax = axes[i]
        df_dist = df[df['distribution'] == dist]
        if df_dist.empty: continue
            
        df_grouped = df_dist.groupby('step')['throughput'].mean().reset_index().sort_values('step')
        steps = df_grouped['step'].tolist()
        throughputs = df_grouped['throughput'].tolist()
        
        # Calcoliamo il "Best so far" (il record attuale ad ogni lancio di dado)
        best_so_far = []
        current_best = 0
        for thr in throughputs:
            if thr > current_best:
                current_best = thr
            best_so_far.append(current_best)
        
        # Disegniamo i punti testati a caso
        ax.plot(steps, throughputs, color='gray', linestyle=':', linewidth=1.5, zorder=1, label="Test Casuali")
        ax.scatter(steps, throughputs, color='white', s=100, zorder=5, edgecolors='gray')
        
        # Disegniamo la linea del record assoluto
        ax.plot(steps, best_so_far, color='blue', linestyle='-', linewidth=2.5, marker='o', zorder=6, label="Record (Best so far)")
        
        ax.set_xticks(steps)
        ax.set_xlabel("Numero di Tentativi")
        ax.set_ylabel("Throughput (ops/sec)")
        ax.set_title(f"Distribuzione: {dist}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{output_prefix}_RS_Convergence.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN LOOP (RANDOM SEARCH)
# ==========================================
def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"results_RS_{ts}.csv")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "cache_GB", "distribution", "journal_ms", "repetition", "duration", "throughput"])

    print("🎲 PARTENZA ALGORITMO: RANDOM SEARCH (Baseline)")

    for dist in DISTRIBUTIONS:
        print(f"\n" + "="*50)
        print(f"🎯 RICERCA CASUALE PER DISTRIBUZIONE: {dist.upper()}")
        print("="*50)
        
        # Creiamo lo spazio delle coordinate possibili (per evitare di testare due volte lo stesso identico punto a caso)
        unvisited = [(c, j) for c in range(len(CACHE_SIZES)) for j in range(len(JOURNAL_INTERVALS))]
        
        best_thr = -1
        best_point = None

        for step in range(1, MAX_EVALUATIONS + 1):
            if not unvisited:
                print("Esaurita la griglia!")
                break
                
            # Selezione puramente casuale
            c_idx, j_idx = random.choice(unvisited)
            unvisited.remove((c_idx, j_idx))
            
            thr = evaluate_point(c_idx, j_idx, dist, csv_filename, step)
            
            if thr > best_thr:
                best_thr = thr
                best_point = (c_idx, j_idx)
                print(f"   🎉 NUOVO RECORD! ({best_thr:.2f} ops/s)")

        print(f"\n🏁 FINE RANDOM SEARCH PER {dist}.")
        print(f"🏆 IL MEGLIO TROVATO A CASO: Cache={CACHE_SIZES[best_point[0]]}GB, Journal={JOURNAL_INTERVALS[best_point[1]]}ms")

    print(f"\n📊 Generazione grafici del Random Search in corso...")
    plot_rs_master(csv_filename, os.path.join(BASE_DIR, f"Analisi_{ts}"))
    plot_rs_convergence(csv_filename, os.path.join(BASE_DIR, f"Analisi_{ts}"))
    
    print(f"✅ Fatto! Trovi i risultati e i grafici in: {BASE_DIR}")

if __name__ == "__main__":
    main()