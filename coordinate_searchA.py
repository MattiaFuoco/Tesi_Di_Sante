import subprocess
import time
import csv
import os
import datetime
import random
import string
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
CACHE_SIZES = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
JOURNAL_INTERVALS = [1, 100, 500]
DISTRIBUTIONS = ["uniform", "zipfian", "sequential"] 
REPETITIONS = 5 

FIXED_THREADS = 6 # <-- Fissiamo i thread a 6 per tutte le run
OPS_PER_THREAD = 50000 # 50 operazioni per thread (WORKLOAD A: 25k read + 25k write)
PRELOAD_DOCS = 480000 # 480k documenti per saturare la cache
DOC_SIZE = 4096 
CONTAINER_RAM = 4 # GB

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "testdb"
COLLECTION_NAME = "testcol"

# Salvataggio diretto nella cartella dello script con il nuovo nome
BASE_DIR = os.path.join(os.getcwd(), "Coordinate Search Resault")

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
# WORKLOAD E VALUTAZIONE PUNTO
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

def evaluate_point(c_idx, j_idx, dist, visited_points, csv_filename, step_num):
    # Se abbiamo già valutato questo punto, non lo rifacciamo (Ottimizzazione!)
    if (c_idx, j_idx) in visited_points:
        return visited_points[(c_idx, j_idx)]
    
    c_gb = CACHE_SIZES[c_idx]
    j_ms = JOURNAL_INTERVALS[j_idx]
    
    print(f"\n   [Step {step_num}] Valutazione Punto: Cache={c_gb}GB, Journal={j_ms}ms")
    
    throughputs = []
    durations = []
    
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
    visited_points[(c_idx, j_idx)] = avg_thr
    print(f"   => Media Punto: {avg_thr:.2f} ops/sec")
    return avg_thr

def get_cross_neighbors(c_idx, j_idx):
    # Genera i vicini a "croce" (Su, Giù, Destra, Sinistra) rispettando i limiti degli array
    neighbors = []
    if c_idx > 0: neighbors.append((c_idx - 1, j_idx)) # Sinistra
    if c_idx < len(CACHE_SIZES) - 1: neighbors.append((c_idx + 1, j_idx)) # Destra
    if j_idx > 0: neighbors.append((c_idx, j_idx - 1)) # Giù
    if j_idx < len(JOURNAL_INTERVALS) - 1: neighbors.append((c_idx, j_idx + 1)) # Su
    return neighbors

# ==========================================
# GENERAZIONE GRAFICI (Ottimizzazione con Path Gradiente Blu)
# ==========================================
import matplotlib.cm as cm
import numpy as np

def plot_optimization_path(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    df_grouped = df.groupby(['distribution', 'step', 'cache_GB', 'journal_ms'])[['throughput', 'duration']].mean().reset_index()

    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    # Funzione per ricostruire il VERO percorso dell'algoritmo
    def get_true_path(df, dist, df_dist_mean):
        path_x, path_y, path_steps = [], [], []
        df_raw_dist = df[df['distribution'] == dist]
        if df_raw_dist.empty: return [], [], []
        
        start_point = df_raw_dist[df_raw_dist['step'] == 1].iloc[0]
        curr_c, curr_j = start_point['cache_GB'], start_point['journal_ms']
        curr_thr = df_dist_mean[(df_dist_mean['cache_GB'] == curr_c) & (df_dist_mean['journal_ms'] == curr_j)]['throughput'].values[0]
        
        path_x.append(curr_c)
        path_y.append(curr_j)
        path_steps.append(1)
        
        for s in range(1, df_dist_mean['step'].max() + 1):
            df_s = df_dist_mean[df_dist_mean['step'] == s]
            if df_s.empty: continue
            best_s = df_s.loc[df_s['throughput'].idxmax()]
            if best_s['throughput'] > curr_thr:
                curr_c, curr_j, curr_thr = best_s['cache_GB'], best_s['journal_ms'], best_s['throughput']
                path_x.append(curr_c)
                path_y.append(curr_j)
                path_steps.append(s + 1)
                
        return path_x, path_y, path_steps

    # ---------------------------------------------------------
    # 1. GRAFICO DEL THROUGHPUT
    # ---------------------------------------------------------
    fig_thr, axes_thr = plt.subplots(1, 3, figsize=(18, 5))
    fig_thr.suptitle("Coordinate Search - Ottimizzazione Throughput\n(Frecce: Progressione Blu | Pallini: Performance)", fontsize=16, fontweight='bold')

    for i, dist in enumerate(DISTRIBUTIONS):
        ax = axes_thr[i]
        df_dist_mean = df_grouped[df_grouped['distribution'] == dist].copy()
        if df_dist_mean.empty: continue

        x_coords = df_dist_mean['cache_GB'].map(cache_map).tolist()
        y_coords = df_dist_mean['journal_ms'].map(journal_map).tolist()
        throughputs = df_dist_mean['throughput'].tolist()

        # Pallini: Performance Reale
        sc_thr = ax.scatter(x_coords, y_coords, c=throughputs, cmap='RdYlGn', s=250, zorder=5, edgecolors='black', linewidth=1.5)

        path_c, path_j, path_steps = get_true_path(df, dist, df_dist_mean)
        path_x = [cache_map[c] for c in path_c]
        path_y = [journal_map[j] for j in path_j]

        # Generiamo le tonalità di BLU in base al numero di step
        num_arrows = len(path_x) - 1
        blue_shades = cm.Blues(np.linspace(0.4, 1.0, max(2, num_arrows + 1)))

        # Disegniamo le frecce con il colore blu incrementale
        for j in range(num_arrows):
            arrow_color = blue_shades[j + 1] # Diventa sempre più scuro
            ax.annotate("", xy=(path_x[j+1], path_y[j+1]), xytext=(path_x[j], path_y[j]), 
                        arrowprops=dict(arrowstyle="-|>,head_length=1.0,head_width=0.5", color=arrow_color, lw=3.5, shrinkA=12, shrinkB=12), zorder=6)
            
        # Etichette mantenute e ben leggibili
        for j in range(len(path_x)):
            ax.annotate(f"Step {path_steps[j]}", xy=(path_x[j], path_y[j]), xytext=(0, 15), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', zorder=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

        ax.set_xticks(range(len(CACHE_SIZES))); ax.set_xticklabels(CACHE_SIZES)
        ax.set_yticks(range(len(JOURNAL_INTERVALS))); ax.set_yticklabels(JOURNAL_INTERVALS)
        ax.set_xlabel("Cache Size (GB)"); ax.set_ylabel("Journal Interval (ms)")
        ax.set_title(f"Distribuzione: {dist}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(-0.5, len(CACHE_SIZES) - 0.5); ax.set_ylim(-0.5, len(JOURNAL_INTERVALS) - 0.3)

    plt.tight_layout()
    fig_thr.subplots_adjust(top=0.82, right=0.92)
    cbar_ax = fig_thr.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar_thr = fig_thr.colorbar(sc_thr, cax=cbar_ax, orientation='vertical')
    cbar_thr.set_label('Throughput (ops/sec)', fontsize=12)
    
    plt.savefig(f"{output_prefix}_optimization_THROUGHPUT.png", dpi=300, bbox_inches='tight')
    plt.close(fig_thr)

    # ---------------------------------------------------------
    # 2. GRAFICO DEL TEMPO DI ESECUZIONE (DURATION)
    # ---------------------------------------------------------
    fig_dur, axes_dur = plt.subplots(1, 3, figsize=(18, 5))
    fig_dur.suptitle("Coordinate Search - Ottimizzazione Tempo\n(Frecce: Progressione Blu | Pallini: Performance)", fontsize=16, fontweight='bold')

    for i, dist in enumerate(DISTRIBUTIONS):
        ax = axes_dur[i]
        df_dist_mean = df_grouped[df_grouped['distribution'] == dist].copy()
        if df_dist_mean.empty: continue

        x_coords = df_dist_mean['cache_GB'].map(cache_map).tolist()
        y_coords = df_dist_mean['journal_ms'].map(journal_map).tolist()
        durations = df_dist_mean['duration'].tolist()

        # Colori INVERTITI per il tempo (Verde = Veloce)
        sc_dur = ax.scatter(x_coords, y_coords, c=durations, cmap='RdYlGn_r', s=250, zorder=5, edgecolors='black', linewidth=1.5)

        path_c, path_j, path_steps = get_true_path(df, dist, df_dist_mean)
        path_x = [cache_map[c] for c in path_c]
        path_y = [journal_map[j] for j in path_j]

        num_arrows = len(path_x) - 1
        blue_shades = cm.Blues(np.linspace(0.4, 1.0, max(2, num_arrows + 1)))

        for j in range(num_arrows):
            arrow_color = blue_shades[j + 1]
            ax.annotate("", xy=(path_x[j+1], path_y[j+1]), xytext=(path_x[j], path_y[j]), 
                        arrowprops=dict(arrowstyle="-|>,head_length=1.0,head_width=0.5", color=arrow_color, lw=3.5, shrinkA=12, shrinkB=12), zorder=6)
            
        for j in range(len(path_x)):
            ax.annotate(f"Step {path_steps[j]}", xy=(path_x[j], path_y[j]), xytext=(0, 15), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', zorder=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

        ax.set_xticks(range(len(CACHE_SIZES))); ax.set_xticklabels(CACHE_SIZES)
        ax.set_yticks(range(len(JOURNAL_INTERVALS))); ax.set_yticklabels(JOURNAL_INTERVALS)
        ax.set_xlabel("Cache Size (GB)"); ax.set_ylabel("Journal Interval (ms)")
        ax.set_title(f"Distribuzione: {dist}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(-0.5, len(CACHE_SIZES) - 0.5); ax.set_ylim(-0.5, len(JOURNAL_INTERVALS) - 0.3)

    plt.tight_layout()
    fig_dur.subplots_adjust(top=0.82, right=0.92)
    cbar_ax_dur = fig_dur.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar_dur = fig_dur.colorbar(sc_dur, cax=cbar_ax_dur, orientation='vertical')
    cbar_dur.set_label('Tempo (Secondi)', fontsize=12)
    
    plt.savefig(f"{output_prefix}_optimization_DURATION.png", dpi=300, bbox_inches='tight')
    plt.close(fig_dur)

# ==========================================
# MAIN LOOP (COORDINATE SEARCH)
# ==========================================
def main():
    # La cartella verrà creata automaticamente se non esiste
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"results_cross_search_{ts}.csv")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "cache_GB", "distribution", "journal_ms", "repetition", "duration", "throughput"])

    print("🚀 PARTENZA ALGORITMO: COORDINATE SEARCH (Ricerca a Croce)")

    for dist in DISTRIBUTIONS:
        print(f"\n" + "="*50)
        print(f"🎯 OTTIMIZZAZIONE PER DISTRIBUZIONE: {dist.upper()}")
        print("="*50)
        
        visited_points = {}
        
        # 1. Punto di partenza (Partiamo dal centro della griglia)
        current_c_idx = len(CACHE_SIZES) // 2
        current_j_idx = len(JOURNAL_INTERVALS) // 2
        step = 1
        
        while True:
            print(f"\n--- INIZIO STEP {step} ---")
            # Valuta il centro della croce
            best_thr = evaluate_point(current_c_idx, current_j_idx, dist, visited_points, csv_filename, step)
            best_c_idx = current_c_idx
            best_j_idx = current_j_idx
            
            # 2. Genera e valuta i vicini (le "braccia" della croce)
            neighbors = get_cross_neighbors(current_c_idx, current_j_idx)
            found_better = False
            
            for n_c, n_j in neighbors:
                thr = evaluate_point(n_c, n_j, dist, visited_points, csv_filename, step)
                if thr > best_thr:
                    best_thr = thr
                    best_c_idx = n_c
                    best_j_idx = n_j
                    found_better = True
            
            # 3. Decisione: Ci spostiamo o abbiamo finito?
            if found_better:
                print(f"\n✅ Miglioramento trovato! Mi sposto verso: Cache={CACHE_SIZES[best_c_idx]}GB, Journal={JOURNAL_INTERVALS[best_j_idx]}ms")
                current_c_idx = best_c_idx
                current_j_idx = best_j_idx
                step += 1
            else:
                print(f"\n🛑 NESSUN VICINO È MIGLIORE. Ottimo Locale raggiunto per {dist}!")
                print(f"🏆 CONFIGURAZIONE OTTIMA: Cache={CACHE_SIZES[current_c_idx]}GB, Journal={JOURNAL_INTERVALS[current_j_idx]}ms")
                break # Esce dal while, passa alla prossima distribuzione

    print(f"\n📊 Generazione grafici del percorso di ottimizzazione in corso...")
    plot_optimization_path(csv_filename, os.path.join(BASE_DIR, f"cross_search_{ts}"))
    print(f"✅ Fatto! Trovi i risultati e il grafico in: {BASE_DIR}")

if __name__ == "__main__":
    main()