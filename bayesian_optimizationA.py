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
import matplotlib.cm as cm
from scipy.interpolate import griddata
from adjustText import adjust_text
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
import pymongo

# Import per l'Ottimizzazione Bayesiana
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import warnings
warnings.filterwarnings("ignore") # Nasconde i warning matematici del GP

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

# Cartella dedicata per la Bayesian Optimization
BASE_DIR = os.path.join(os.getcwd(), "Bayesian Optimization Result")

# ==========================================
# PARAMETRI BAYESIAN OPTIMIZATION
# ==========================================
INIT_POINTS = 3       # Quanti punti random testare all'inizio per "istruire" il modello
OPT_STEPS = 9         # Quanti passi intelligenti fare guidati dalla matematica
TOTAL_STEPS = INIT_POINTS + OPT_STEPS

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

def evaluate_point(c_idx, j_idx, dist, visited_points, csv_filename, step_num, phase):
    if (c_idx, j_idx) in visited_points:
        return visited_points[(c_idx, j_idx)]
    
    c_gb = CACHE_SIZES[c_idx]
    j_ms = JOURNAL_INTERVALS[j_idx]
    
    print(f"\n   [Step {step_num} | {phase}] Valutazione: Cache={c_gb}GB, Journal={j_ms}ms")
    
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
            writer.writerow([step_num, phase, c_gb, dist, j_ms, r+1, dur, thr])
            
        subprocess.run(["docker", "rm", "-f", "mongo-test"], capture_output=True)
        subprocess.run(["docker", "volume", "rm", "mongo_data"], capture_output=True)
        
    avg_thr = np.mean(throughputs)
    visited_points[(c_idx, j_idx)] = avg_thr
    print(f"   => Media Punto: {avg_thr:.2f} ops/sec")
    return avg_thr

# ==========================================
# GENERAZIONE GRAFICI BAYESIANI
# ==========================================
def plot_bo_master(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Bayesian Optimization - Esplorazione Spaziale\n(Frecce: Ordine Cronologico | Sfondo: Mappa Termica Prestazioni)", fontsize=16, fontweight='bold')
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

        # Percorso Temporale BO
        df_path = df_grouped.merge(df_dist[['cache_GB', 'journal_ms', 'step', 'phase']].drop_duplicates(), on=['cache_GB', 'journal_ms']).sort_values('step')
        path_x = df_path['cache_GB'].map(cache_map).tolist()
        path_y = df_path['journal_ms'].map(journal_map).tolist()
        path_steps = df_path['step'].tolist()
        path_phases = df_path['phase'].tolist()

        texts = []
        for j in range(len(path_x) - 1):
            ax.annotate("", xy=(path_x[j+1], path_y[j+1]), xytext=(path_x[j], path_y[j]), 
                        arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.4", 
                                        color="blue", lw=2.0, alpha=0.7,
                                        shrinkA=12, shrinkB=12, connectionstyle="arc3,rad=0.2"), zorder=6)
            
        for j in range(len(path_x)):
            # Distinguiamo i punti di Inizializzazione (Grigi) da quelli Bayesiani (Bianchi)
            pt_color = 'lightgray' if path_phases[j] == 'Init' else 'white'
            ax.scatter(path_x[j], path_y[j], color=pt_color, s=150, zorder=8, edgecolors='black', linewidth=1.5)
            
            t = ax.text(path_x[j], path_y[j], f"T{path_steps[j]}", ha='center', va='center', 
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
    
    plt.savefig(f"{output_prefix}_BO_Map.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_bo_convergence(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Bayesian Optimization - Convergence Plot\n(Progressione Temporale del Throughput)", fontsize=16, fontweight='bold')

    for i, dist in enumerate(DISTRIBUTIONS):
        ax = axes[i]
        df_dist = df[df['distribution'] == dist]
        if df_dist.empty: continue
            
        df_grouped = df_dist.groupby('step').first().reset_index()
        steps = df_grouped['step'].tolist()
        throughputs = df_grouped['throughput'].tolist()
        phases = df_grouped['phase'].tolist()
        
        ax.plot(steps, throughputs, color='gray', linestyle='-', linewidth=2, zorder=1)
        
        colors = ['gray' if p == 'Init' else 'blue' for p in phases]
        ax.scatter(steps, throughputs, c=colors, s=150, zorder=5, edgecolors='black', linewidth=1.5)
        
        ax.set_xticks(steps)
        ax.set_xlabel("Step dell'Algoritmo")
        ax.set_ylabel("Throughput (ops/sec)")
        ax.set_title(f"Distribuzione: {dist}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{output_prefix}_BO_Convergence.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN LOOP (BAYESIAN OPTIMIZATION)
# ==========================================
def get_normalized_coords(c_idx, j_idx):
    # Normalizza le coordinate tra 0 e 1 per far lavorare bene la matematica del Processo Gaussiano
    return [c_idx / (len(CACHE_SIZES) - 1), j_idx / (len(JOURNAL_INTERVALS) - 1)]

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"results_BO_{ts}.csv")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "phase", "cache_GB", "distribution", "journal_ms", "repetition", "duration", "throughput"])

    print("🧠 PARTENZA ALGORITMO: BAYESIAN OPTIMIZATION")

    # Modello Matematico (Gaussian Process)
    # Usiamo un kernel Matern, perfetto per superfici di ottimizzazione lisce ma con picchi
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
    
    for dist in DISTRIBUTIONS:
        print(f"\n" + "="*50)
        print(f"🎯 OTTIMIZZAZIONE PER DISTRIBUZIONE: {dist.upper()}")
        print("="*50)
        
        visited_points = {}
        X_train = [] # Conterrà le coordinate testate
        y_train = [] # Conterrà le prestazioni ottenute
        
        # Generiamo la lista di tutti i punti possibili della griglia (Spazio di ricerca)
        unvisited = [(c, j) for c in range(len(CACHE_SIZES)) for j in range(len(JOURNAL_INTERVALS))]
        
        # 1. FASE DI INIZIALIZZAZIONE (Campionamento Random)
        print(f"\n--- FASE 1: INIZIALIZZAZIONE ({INIT_POINTS} PUNTI) ---")
        init_points = random.sample(unvisited, INIT_POINTS)
        
        step = 1
        for c_idx, j_idx in init_points:
            thr = evaluate_point(c_idx, j_idx, dist, visited_points, csv_filename, step, "Init")
            X_train.append(get_normalized_coords(c_idx, j_idx))
            y_train.append(thr)
            unvisited.remove((c_idx, j_idx))
            step += 1
            
        # 2. FASE BAYESIANA (Il modello sceglie dove andare)
        print(f"\n--- FASE 2: OTTIMIZZAZIONE BAYESIANA (SURROGATE MODEL) ---")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        
        for opt_step in range(OPT_STEPS):
            print(f"\n👉 Calcolo Processo Gaussiano - Step {step}...")
            # Addestriamo il modello con i dati raccolti finora
            gp.fit(X_train, y_train)
            
            best_ucb = -float('inf')
            best_point = None
            
            # Acquisition Function: Upper Confidence Bound (UCB)
            # Calcoliamo l'UCB per tutti i punti NON ancora visitati
            for c_idx, j_idx in unvisited:
                x_test = np.array([get_normalized_coords(c_idx, j_idx)])
                mu, sigma = gp.predict(x_test, return_std=True)
                
                # UCB = Media Prevista + (Kappa * Incertezza). Kappa=1.96 significa 95% di confidenza.
                kappa = 1.96
                ucb = mu[0] + kappa * sigma[0]
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_point = (c_idx, j_idx)
            
            # Il modello ha deciso qual è il punto più promettente (esplorazione vs sfruttamento)
            c_idx, j_idx = best_point
            print(f"   🤖 Il Modello Bayesiano ha scelto: Cache={CACHE_SIZES[c_idx]}GB, Journal={JOURNAL_INTERVALS[j_idx]}ms")
            
            thr = evaluate_point(c_idx, j_idx, dist, visited_points, csv_filename, step, "BO")
            
            X_train.append(get_normalized_coords(c_idx, j_idx))
            y_train.append(thr)
            unvisited.remove((c_idx, j_idx))
            step += 1
            
        # Trova il vincitore assoluto
        best_idx = np.argmax(y_train)
        best_c_idx = int(round(X_train[best_idx][0] * (len(CACHE_SIZES) - 1)))
        best_j_idx = int(round(X_train[best_idx][1] * (len(JOURNAL_INTERVALS) - 1)))
        print(f"\n🏁 FINE BO PER {dist}. Miglior configurazione trovata: Cache={CACHE_SIZES[best_c_idx]}GB, Journal={JOURNAL_INTERVALS[best_j_idx]}ms con {y_train[best_idx]:.2f} ops/s")

    print(f"\n📊 Generazione grafici Bayesiani in corso...")
    plot_bo_master(csv_filename, os.path.join(BASE_DIR, f"Analisi_{ts}"))
    plot_bo_convergence(csv_filename, os.path.join(BASE_DIR, f"Analisi_{ts}"))
    
    print(f"✅ Fatto! Trovi i risultati e i grafici in: {BASE_DIR}")

if __name__ == "__main__":
    main()