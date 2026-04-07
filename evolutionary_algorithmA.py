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

BASE_DIR = os.path.join(os.getcwd(), "Evolutionary Algorithm Result")

# ==========================================
# PARAMETRI ALGORITMO GENETICO (DARWIN)
# ==========================================
POPULATION_SIZE = 4   # Numero di individui (configurazioni) per generazione
GENERATIONS = 5       # Numero di generazioni (cicli di evoluzione)
MUTATION_RATE = 0.3   # 30% di probabilità che un gene muti

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

def evaluate_point(c_idx, j_idx, dist, visited_points, csv_filename, gen_num, ind_id):
    if (c_idx, j_idx) in visited_points:
        return visited_points[(c_idx, j_idx)]
    
    c_gb = CACHE_SIZES[c_idx]
    j_ms = JOURNAL_INTERVALS[j_idx]
    
    print(f"\n   [Gen {gen_num} | Ind {ind_id}] Valutazione: Cache={c_gb}GB, Journal={j_ms}ms")
    
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
        
        # is_best_of_gen sarà calcolato e aggiornato nel dataframe alla fine, 
        # intanto salviamo i dati grezzi
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([gen_num, ind_id, c_gb, dist, j_ms, r+1, dur, thr, False])
            
        subprocess.run(["docker", "rm", "-f", "mongo-test"], capture_output=True)
        subprocess.run(["docker", "volume", "rm", "mongo_data"], capture_output=True)
        
    avg_thr = np.mean(throughputs)
    visited_points[(c_idx, j_idx)] = avg_thr
    print(f"   => Fitness (Throughput): {avg_thr:.2f} ops/sec")
    return avg_thr

# ==========================================
# LOGICA GENETICA DARWINIANA
# ==========================================
def get_random_individual():
    return (random.randint(0, len(CACHE_SIZES)-1), random.randint(0, len(JOURNAL_INTERVALS)-1))

def crossover(parent1, parent2):
    # Miscela genetica: scambia le caratteristiche di Cache e Journal
    if random.random() < 0.5:
        return (parent1[0], parent2[1]), (parent2[0], parent1[1])
    else:
        return parent1, parent2 # Nessun incrocio

def mutate(individual):
    c_idx, j_idx = individual
    if random.random() < MUTATION_RATE:
        c_idx = random.randint(0, len(CACHE_SIZES)-1)
    if random.random() < MUTATION_RATE:
        j_idx = random.randint(0, len(JOURNAL_INTERVALS)-1)
    return (c_idx, j_idx)

# ==========================================
# GENERAZIONE GRAFICI (EVOLUTIONARY)
# ==========================================
def plot_ea_master(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Evolutionary Algorithm - Mappa dell'Evoluzione\n(Sfondo: Interpolazione | Frecce: Evoluzione dell'Alfa | Gx piccoli: Estinti)", fontsize=16, fontweight='bold')
    contour_plot = None 

    for i, dist in enumerate(DISTRIBUTIONS):
        ax = axes[i]
        df_dist = df[df['distribution'] == dist]
        if df_dist.empty: continue

        # Sfondo Topografico Interpolato
        df_grouped_all = df_dist.groupby(['cache_GB', 'journal_ms'])['throughput'].mean().reset_index()
        x_coords = df_grouped_all['cache_GB'].map(cache_map).values
        y_coords = df_grouped_all['journal_ms'].map(journal_map).values
        z_vals = df_grouped_all['throughput'].values

        grid_x, grid_y = np.mgrid[0:len(CACHE_SIZES)-1:100j, 0:len(JOURNAL_INTERVALS)-1:100j]
        grid_z = griddata((x_coords, y_coords), z_vals, (grid_x, grid_y), method='linear')
        
        contour_plot = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdYlGn', alpha=0.6)
        ax.contour(grid_x, grid_y, grid_z, levels=10, colors='black', linewidths=0.5, alpha=0.3)

        texts = []
        
        # Definiamo chi è il "Best" per ogni generazione agganciando l'is_best_of_gen
        df_gen = df_dist.groupby(['generation', 'ind_id', 'cache_GB', 'journal_ms'])['throughput'].mean().reset_index()
        
        # Calcoliamo i best
        best_per_gen = df_gen.loc[df_gen.groupby('generation')['throughput'].idxmax()]
        best_coords = set(zip(best_per_gen['cache_GB'], best_per_gen['journal_ms']))

        # --- 1. INDIVIDUI ESTINTI (Il resto della popolazione) ---
        others = df_gen[~df_gen.index.isin(best_per_gen.index)]
        grouped_others = others.groupby(['cache_GB', 'journal_ms'])['generation'].apply(lambda x: list(set(x))).reset_index()

        for _, row in grouped_others.iterrows():
            rx = cache_map[row['cache_GB']]
            ry = journal_map[row['journal_ms']]
            gen_str = ", ".join([f"G{int(g)}" for g in sorted(row['generation'])])
            
            # Effetto Satellite
            if (row['cache_GB'], row['journal_ms']) in best_coords:
                sx, sy = rx + 0.15, ry - 0.15
                ax.plot([rx, sx], [ry, sy], color='gray', linestyle=':', lw=1.5, zorder=2)
            else:
                sx, sy = rx, ry
                
            ax.scatter(sx, sy, color='gray', s=40, alpha=0.7, zorder=9)
            t = ax.text(sx + 0.08, sy - 0.08, gen_str, ha='left', va='top', 
                        fontsize=8, color='dimgray', fontstyle='italic', fontweight='bold', zorder=11,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
            texts.append(t)

        # --- 2. INDIVIDUI ALFA (L'evoluzione del migliore) ---
        path_x = best_per_gen['cache_GB'].map(cache_map).tolist()
        path_y = best_per_gen['journal_ms'].map(journal_map).tolist()
        path_gens = best_per_gen['generation'].tolist()
        path_thr = best_per_gen['throughput'].tolist()

        for j in range(len(path_x) - 1):
            if path_x[j] != path_x[j+1] or path_y[j] != path_y[j+1]:
                is_improvement = path_thr[j+1] >= path_thr[j]
                arrow_color = "darkblue" if is_improvement else "darkorange"
                arrow_style = "-" if is_improvement else "dashed"
                ax.annotate("", xy=(path_x[j+1], path_y[j+1]), xytext=(path_x[j], path_y[j]), 
                            arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.4", 
                                            color=arrow_color, ls=arrow_style, lw=2.5, 
                                            shrinkA=12, shrinkB=12, connectionstyle="arc3,rad=0.2"), zorder=6)
            
        for j in range(len(path_x)):
            ax.scatter(path_x[j], path_y[j], color='white', s=150, zorder=8, edgecolors='black', linewidth=1.5)
            # Raggruppiamo i Best se rimangono gli stessi per più generazioni
            t = ax.text(path_x[j], path_y[j], f"Best G{path_gens[j]}", ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='black', zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1, alpha=0.9))
            texts.append(t)

        if texts:
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
    
    plt.savefig(f"{output_prefix}_EA_Map.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_ea_convergence(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Evolutionary Algorithm - Convergence Plot\n(Progressione Temporale del Throughput della Specie)", fontsize=16, fontweight='bold')

    for i, dist in enumerate(DISTRIBUTIONS):
        ax = axes[i]
        df_dist = df[df['distribution'] == dist]
        if df_dist.empty: continue
            
        df_gen = df_dist.groupby(['generation', 'ind_id'])['throughput'].mean().reset_index()
        best_thr = df_gen.groupby('generation')['throughput'].max().tolist()
        avg_thr = df_gen.groupby('generation')['throughput'].mean().tolist()
        gens = sorted(df_gen['generation'].unique())
        
        # Linea del Migliore (Individuo Alfa)
        ax.plot(gens, best_thr, color='blue', linestyle='-', linewidth=2.5, marker='o', markersize=8, label="Miglior Individuo (Alfa)")
        # Linea della Media (L'intera Popolazione)
        ax.plot(gens, avg_thr, color='orange', linestyle='--', linewidth=2, marker='X', markersize=8, label="Media Popolazione")
        
        ax.set_xticks(gens)
        ax.set_xlabel("Generazione")
        ax.set_ylabel("Throughput (ops/sec)")
        ax.set_title(f"Distribuzione: {dist}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{output_prefix}_EA_Convergence.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN LOOP (EVOLUTIONARY ALGORITHM)
# ==========================================
def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"results_EA_{ts}.csv")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "ind_id", "cache_GB", "distribution", "journal_ms", "repetition", "duration", "throughput", "is_best_of_gen"])

    print("🧬 PARTENZA ALGORITMO: EVOLUTIONARY ALGORITHM (Darwin)")

    for dist in DISTRIBUTIONS:
        print(f"\n" + "="*50)
        print(f"🎯 EVOLUZIONE PER DISTRIBUZIONE: {dist.upper()}")
        print("="*50)
        
        visited_points = {}
        population = [get_random_individual() for _ in range(POPULATION_SIZE)]
        best_overall = None
        best_thr_overall = -1

        for gen in range(1, GENERATIONS + 1):
            print(f"\n🌍 --- GENERAZIONE {gen} ---")
            fitness_scores = []
            
            # Valutazione della Popolazione (Sopravvivenza)
            for i, ind in enumerate(population):
                c_idx, j_idx = ind
                thr = evaluate_point(c_idx, j_idx, dist, visited_points, csv_filename, gen, i+1)
                fitness_scores.append((thr, ind))
                
            # Ordiniamo in base al fitness (dal migliore al peggiore)
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            best_gen_thr, best_gen_ind = fitness_scores[0]
            print(f"   🏆 L'Alfa della Gen {gen} è Cache={CACHE_SIZES[best_gen_ind[0]]}GB, Journal={JOURNAL_INTERVALS[best_gen_ind[1]]}ms ({best_gen_thr:.2f} ops/s)")
            
            if best_gen_thr > best_thr_overall:
                best_thr_overall = best_gen_thr
                best_overall = best_gen_ind

            # Se siamo all'ultima generazione, non ci serve accoppiarli
            if gen == GENERATIONS: break
                
            # Selezione e Riproduzione
            new_population = []
            
            # Elitismo: L'Alfa passa automaticamente alla generazione successiva
            new_population.append(best_gen_ind)
            
            # Torneo: Selezioniamo i genitori e creiamo i figli
            while len(new_population) < POPULATION_SIZE:
                # Tournament Selection
                parent1 = random.choice(fitness_scores[:3])[1] # Sceglie tra i 3 migliori
                parent2 = random.choice(fitness_scores[:3])[1]
                
                # Crossover
                child1, child2 = crossover(parent1, parent2)
                
                # Mutazione
                child1 = mutate(child1)
                new_population.append(child1)
                
                if len(new_population) < POPULATION_SIZE:
                    child2 = mutate(child2)
                    new_population.append(child2)
                    
            population = new_population

        print(f"\n🏁 FINE EVOLUZIONE PER {dist}.")
        print(f"🏆 CONFIGURAZIONE ASSOLUTA: Cache={CACHE_SIZES[best_overall[0]]}GB, Journal={JOURNAL_INTERVALS[best_overall[1]]}ms")

    # Modifichiamo il CSV in blocco per aggiornare il flag "is_best_of_gen" (utile per i grafici)
    df = pd.read_csv(csv_filename)
    df_mean = df.groupby(['distribution', 'generation', 'ind_id'])['throughput'].mean().reset_index()
    
    for dist in DISTRIBUTIONS:
        df_dist = df_mean[df_mean['distribution'] == dist]
        for gen in df_dist['generation'].unique():
            df_gen = df_dist[df_dist['generation'] == gen]
            best_ind = df_gen.loc[df_gen['throughput'].idxmax(), 'ind_id']
            # Aggiorna il DataFrame originale
            mask = (df['distribution'] == dist) & (df['generation'] == gen) & (df['ind_id'] == best_ind)
            df.loc[mask, 'is_best_of_gen'] = True

    df.to_csv(csv_filename, index=False)

    print(f"\n📊 Generazione grafici dell'Evoluzione in corso...")
    plot_ea_master(csv_filename, os.path.join(BASE_DIR, f"Analisi_{ts}"))
    plot_ea_convergence(csv_filename, os.path.join(BASE_DIR, f"Analisi_{ts}"))
    
    print(f"✅ Fatto! Trovi i risultati e i grafici in: {BASE_DIR}")

if __name__ == "__main__":
    main()