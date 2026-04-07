import subprocess
import time
import csv
import os
import datetime
import random
import string
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata
from adjustText import adjust_text
import pymongo

# ==========================================
# CONFIGURAZIONI GLOBALI
# ==========================================
CACHE_SIZES = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
JOURNAL_INTERVALS = [1, 100, 500]
DISTRIBUTIONS = ["uniform", "zipfian", "sequential"] 
REPETITIONS = 5 

FIXED_THREADS = 6 # <-- Fissiamo i thread a 6 per tutte le run
OPS_PER_THREAD = 50000 # 50k operazioni per thread (WORKLOAD A: 25k read + 25k write) 
PRELOAD_DOCS = 480000 # 480k documenti per saturare la cache (considerando overhead e spazio occupato dai journal)  
DOC_SIZE = 4096 
CONTAINER_RAM = 4 # GB

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "testdb"
COLLECTION_NAME = "testcol"

# Salvataggio diretto nella cartella dello script con il nuovo nome
BASE_DIR = os.path.join(os.getcwd(), "Simulated Annealing Result")

# ==========================================
# PARAMETRI SIMULATED ANNEALING
# ==========================================
INITIAL_TEMPERATURE = 100.0  # Temperatura di partenza
COOLING_RATE = 0.8           # Fattore di raffreddamento (80% della temp precedente ad ogni step)
MAX_STEPS = 12               # Numero massimo di iterazioni per distribuzione

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

def evaluate_point(c_idx, j_idx, dist, visited_points):
    # MEMOIZATION: Risparmiamo tempo se il punto è già stato testato!
    if (c_idx, j_idx) in visited_points:
        return visited_points[(c_idx, j_idx)]
    
    c_gb = CACHE_SIZES[c_idx]
    j_ms = JOURNAL_INTERVALS[j_idx]
    
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
            
        subprocess.run(["docker", "rm", "-f", "mongo-test"], capture_output=True)
        subprocess.run(["docker", "volume", "rm", "mongo_data"], capture_output=True)
        
    avg_thr = np.mean(throughputs)
    avg_dur = np.mean(durations)
    visited_points[(c_idx, j_idx)] = (avg_thr, avg_dur)
    
    return avg_thr, avg_dur

def get_random_neighbor(c_idx, j_idx):
    # Genera i vicini validi e ne sceglie UNO a caso (esplorazione tipica del SA)
    neighbors = []
    if c_idx > 0: neighbors.append((c_idx - 1, j_idx))
    if c_idx < len(CACHE_SIZES) - 1: neighbors.append((c_idx + 1, j_idx))
    if j_idx > 0: neighbors.append((c_idx, j_idx - 1))
    if j_idx < len(JOURNAL_INTERVALS) - 1: neighbors.append((c_idx, j_idx + 1))
    return random.choice(neighbors)

# ==========================================
# GENERAZIONE GRAFICI 
# ==========================================
def plot_sa_master(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Simulated Annealing - Mappa Topografica dell'Esplorazione\n(Sfondo: Interpolazione | Tx in box = Accettati | Tx piccoli = Rifiutati)", fontsize=16, fontweight='bold')

    contour_plot = None 

    for i, dist in enumerate(DISTRIBUTIONS):
        ax = axes[i]
        df_dist = df[df['distribution'] == dist]
        if df_dist.empty: continue

        # 1. SFONDO TOPOGRAFICO
        df_grouped = df_dist.groupby(['cache_GB', 'journal_ms'])['throughput'].mean().reset_index()
        x_coords = df_grouped['cache_GB'].map(cache_map).values
        y_coords = df_grouped['journal_ms'].map(journal_map).values
        z_vals = df_grouped['throughput'].values

        grid_x, grid_y = np.mgrid[0:len(CACHE_SIZES)-1:100j, 0:len(JOURNAL_INTERVALS)-1:100j]
        grid_z = griddata((x_coords, y_coords), z_vals, (grid_x, grid_y), method='linear')
        
        contour_plot = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdYlGn', alpha=0.6)
        ax.contour(grid_x, grid_y, grid_z, levels=10, colors='black', linewidths=0.5, alpha=0.3)

        texts = [] 

        # 2. PUNTI RIFIUTATI (I tentativi falliti RAGGRUPPATI E ANTI-SOVRAPPOSIZIONE)
        df_rejected = df_dist[df_dist['accepted'] == False]
        df_accepted = df_dist[df_dist['accepted'] == True]
        
        accepted_coords = set(zip(df_accepted['cache_GB'], df_accepted['journal_ms']))
        grouped_rejected = df_rejected.groupby(['cache_GB', 'journal_ms'])['step'].apply(list).reset_index()
        
        for _, row in grouped_rejected.iterrows():
            rx = cache_map[row['cache_GB']]
            ry = journal_map[row['journal_ms']]
            
            steps_str = ", ".join([f"T{int(s)}" for s in row['step']])
            
            if (row['cache_GB'], row['journal_ms']) in accepted_coords:
                sx, sy = rx + 0.15, ry - 0.15
                ax.plot([rx, sx], [ry, sy], color='gray', linestyle=':', lw=1.5, zorder=2)
            else:
                sx, sy = rx, ry
                
            ax.scatter(sx, sy, color='gray', s=40, alpha=0.7, zorder=9)
            
            t = ax.text(sx + 0.08, sy - 0.08, steps_str, ha='left', va='top', 
                        fontsize=8, color='dimgray', fontstyle='italic', fontweight='bold', zorder=11,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
            texts.append(t)

        # 3. PUNTI ACCETTATI (Il percorso ufficiale)
        df_accepted = df_accepted.sort_values('step')
        path_x = df_accepted['cache_GB'].map(cache_map).tolist()
        path_y = df_accepted['journal_ms'].map(journal_map).tolist()
        path_steps = df_accepted['step'].tolist()
        path_thr = df_accepted['throughput'].tolist()

        for j in range(len(path_x) - 1):
            if path_x[j] != path_x[j+1] or path_y[j] != path_y[j+1]:
                is_improvement = path_thr[j+1] > path_thr[j]
                arrow_color = "darkblue" if is_improvement else "darkorange"
                arrow_style = "-" if is_improvement else "dashed"
                
                ax.annotate("", xy=(path_x[j+1], path_y[j+1]), xytext=(path_x[j], path_y[j]), 
                            arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.4", 
                                            color=arrow_color, ls=arrow_style, lw=2.5, 
                                            shrinkA=12, shrinkB=12, connectionstyle="arc3,rad=0.2"), zorder=6)
            
        for j in range(len(path_x)):
            ax.scatter(path_x[j], path_y[j], color='white', s=150, zorder=8, edgecolors='black', linewidth=1.5)
            t = ax.text(path_x[j], path_y[j], f"T{path_steps[j]}", ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='black', zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1, alpha=0.9))
            texts.append(t)

        # 4. MAGIA ANTI-SOVRAPPOSIZIONE
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.7))

        # 5. FORMATTAZIONE
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
    
    plt.savefig(f"{output_prefix}_Topographic_Map.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_sa_convergence(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    df_accepted = df[df['accepted'] == True].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Simulated Annealing - Andamento delle Prestazioni (Convergence Plot)\n(Le discese indicano l'esplorazione di zone temporaneamente peggiori)", fontsize=16, fontweight='bold')

    for i, dist in enumerate(DISTRIBUTIONS):
        ax = axes[i]
        df_dist = df_accepted[df_accepted['distribution'] == dist].sort_values('step')
        if df_dist.empty: continue
            
        steps = df_dist['step'].tolist()
        throughputs = df_dist['throughput'].tolist()
        
        ax.plot(steps, throughputs, color='gray', linestyle='-', linewidth=2, zorder=1)
        
        colors = []
        for j in range(len(throughputs)):
            if j == 0:
                colors.append('blue') 
            elif throughputs[j] > throughputs[j-1]:
                colors.append('green') 
            else:
                colors.append('orange') 
                
        ax.scatter(steps, throughputs, c=colors, s=150, zorder=5, edgecolors='black', linewidth=1.5)
        
        ax.set_xticks(steps)
        ax.set_xlabel("Step dell'Algoritmo")
        ax.set_ylabel("Throughput (ops/sec)")
        ax.set_title(f"Distribuzione: {dist}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, markeredgecolor='black')
        ]
        ax.legend(custom_lines, ['Partenza', 'Miglioramento', 'Esplorazione (Peggio)'], loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{output_prefix}_Convergence_Plot.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN LOOP (SIMULATED ANNEALING)
# ==========================================
def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"results_SA_{ts}.csv")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "cache_GB", "distribution", "journal_ms", "duration", "throughput", "temperature", "accepted"])

    print("🔥 PARTENZA ALGORITMO: SIMULATED ANNEALING")

    for dist in DISTRIBUTIONS:
        print(f"\n" + "="*50)
        print(f"🎯 OTTIMIZZAZIONE PER DISTRIBUZIONE: {dist.upper()}")
        print("="*50)
        
        visited_points = {}
        
        # 1. Punto di partenza random 
        current_c_idx = random.randint(0, len(CACHE_SIZES)-1)
        current_j_idx = random.randint(0, len(JOURNAL_INTERVALS)-1)
        
        current_temp = INITIAL_TEMPERATURE
        
        print(f"\n--- STEP 1 (Partenza) ---")
        curr_thr, curr_dur = evaluate_point(current_c_idx, current_j_idx, dist, visited_points)
        
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([1, CACHE_SIZES[current_c_idx], dist, JOURNAL_INTERVALS[current_j_idx], curr_dur, curr_thr, current_temp, True])
        
        best_c_idx, best_j_idx, best_thr = current_c_idx, current_j_idx, curr_thr
        
        for step in range(2, MAX_STEPS + 1):
            print(f"\n--- STEP {step} (Temperatura: {current_temp:.1f}) ---")
            
            n_c, n_j = get_random_neighbor(current_c_idx, current_j_idx)
            print(f"   Vado a curiosare in: Cache={CACHE_SIZES[n_c]}GB, Journal={JOURNAL_INTERVALS[n_j]}ms")
            
            n_thr, n_dur = evaluate_point(n_c, n_j, dist, visited_points)
            
            accepted = False
            if n_thr > curr_thr:
                print(f"   ✅ MIGLIORAMENTO: {n_thr:.2f} > {curr_thr:.2f}. Accettato!")
                accepted = True
            else:
                delta = n_thr - curr_thr 
                probability = math.exp(delta / current_temp)
                rand_val = random.random()
                
                if rand_val < probability:
                    print(f"   ⚠️ PEGGIORAMENTO ACCETTATO per esplorare: Prob={probability:.2f}, Rand={rand_val:.2f}")
                    accepted = True
                else:
                    print(f"   ❌ RIFIUTATO: Prob={probability:.2f}, Rand={rand_val:.2f}. Resto dove sono.")

            if accepted:
                current_c_idx, current_j_idx = n_c, n_j
                curr_thr = n_thr
                if curr_thr > best_thr:
                    best_c_idx, best_j_idx, best_thr = current_c_idx, current_j_idx, curr_thr

            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, CACHE_SIZES[n_c], dist, JOURNAL_INTERVALS[n_j], n_dur, n_thr, current_temp, accepted])
            
            current_temp *= COOLING_RATE
            
        print(f"\n🏁 FINE SA PER {dist}. Miglior configurazione trovata: Cache={CACHE_SIZES[best_c_idx]}GB, Journal={JOURNAL_INTERVALS[best_j_idx]}ms")

    print(f"\n📊 Generazione grafici in corso...")
    
    # Richiamo le funzioni dei grafici aggiornate
    plot_sa_master(csv_filename, os.path.join(BASE_DIR, f"Analisi_{ts}"))
    plot_sa_convergence(csv_filename, os.path.join(BASE_DIR, f"Analisi_{ts}"))
    
    print(f"✅ Fatto! Trovi i risultati e i grafici in: {BASE_DIR}")

if __name__ == "__main__":
    main()