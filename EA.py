# ==========================================
# IMPORTAZIONI DI BASE E SISTEMA
# ==========================================
import os
import time
import datetime
import csv
import random
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# ==========================================
# LIBRERIE PER GRAFICI 3x3 (HEATMAP)
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
try:
    from adjustText import adjust_text
except ImportError:
    pass

# ==========================================
# IMPORTAZIONE DAL "CERVELLO CENTRALE"
# ==========================================
from config import *

# ==========================================
# VARIABILI GLOBALI E COSTANTI DARWINIANE
# ==========================================
BASE_DIR = os.path.join(get_master_dir(), "Evolutionary Algorithm")

# PARAMETRI DELL'ALGORITMO GENETICO (PURI)
# Dimensione della popolazione per generazione
POPULATION_SIZE = max(4, MAX_EVALUATIONS // 8)    # (se dispari-->arrotondameto per difetto)
# Partirà una nuova generazione ogni volta che avremo completato la valutazione di POPULATION_SIZE individui, fino a raggiungere il budget totale di MAX_EVALUATIONS.
# Quindi se MAX_EVALUATIONS è dispari, l'ultima generazione potrebbe essere parziale, ma non supererà mai il budget totale.
# In questo modo garantiamo un numero di generazioni dinamico e adattivo in base al budget definito in config.py. 

# Calcoliamo dinamicamente le generazioni in base al budget di config.py
# Usiamo // per la divisione intera (es. 40 // 5 = 8 generazioni)
GENERATIONS = MAX_EVALUATIONS // POPULATION_SIZE     

MUTATION_RATE = 0.40     # Tasso di mutazione FISSO, come in natura

# ==========================================
# FUNZIONI GENETICHE E VALUTAZIONE
# ==========================================
# create_random_individual() simula la nascita casuale di un individuo con un DNA unico.
def create_random_individual():
    """Genesi: Crea un individuo con DNA casuale."""

    # Ogni individuo è una tupla di 4 geni (cache_gb, journal_ms, compressor, distribution),
    # scelti casualmente dallo spazio di ricerca globale definito in config.py.
    return (
        random.choice(CACHE_SIZES),
        random.choice(JOURNAL_INTERVALS),
        random.choice(COMPRESSORS),
        random.choice(DISTRIBUTIONS)
    )

# evaluate_individual() simula la valutazione della forza di un individuo nel suo ambiente (YCSB).
def evaluate_individual(individual, visited_points, csv_filename, gen_num, eval_id, start_time_global):
    """Fitness: Valuta la forza dell'individuo nel suo ambiente (YCSB)."""
    c_gb, j_ms, comp, dist = individual
    
    # --- MEMOIZATION: Memoria Genetica ---
    # Se questo identico DNA è già nato in passato, conosciamo già la sua forza.
    if individual in visited_points:
        avg_thr, avg_dur = visited_points[individual]
        print(f"      -> ⏭️ DNA già noto! Recupero dalla memoria genetica: {avg_thr:.2f} ops/sec")
        elapsed_minutes = (time.time() - start_time_global) / 60.0
        
        # Salviamo comunque il risultato nel CSV, indicando che è un punto già visitato (is_best_of_gen=False).
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([gen_num, eval_id, c_gb, j_ms, comp, dist, "mean", avg_dur, avg_thr, False, round(elapsed_minutes, 2)])
        return avg_thr
    
    print(f"\n   [Gen {gen_num} | Ind {eval_id}] Valuto: C={c_gb}GB, J={j_ms}ms, Comp={comp}, Dist={dist.upper()} ...", end="", flush=True)
    
    # Esecuzione nativa tramite la MASTER API in config.py
    avg_thr, avg_dur = execute_full_test(c_gb, j_ms, comp, dist)
    
    print(f"   🚀 THROUGHPUT FINALE MEDIO: {avg_thr:.2f} ops/sec\n")
    
    elapsed_minutes = (time.time() - start_time_global) / 60.0
    
    # Salviamo il risultato nel CSV, indicando che è un punto nuovo (is_best_of_gen=False).
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([gen_num, eval_id, c_gb, j_ms, comp, dist, "mean", avg_dur, avg_thr, False, round(elapsed_minutes, 2)])
        
    # Aggiorniamo la memoria genetica con questo nuovo punto visitato, così da evitare di ripetere test già fatti in futuro.
    visited_points[individual] = (avg_thr, avg_dur)
    
    return avg_thr

# ==========================================
# FUNZIONI DI SELEZIONE, CROSSOVER E MUTAZIONE
# ==========================================
# crossover() simula la riproduzione sessuata, mescolando i tratti di due genitori per creare due figli unici.
def crossover(parent1, parent2):
    """Riproduzione: Uniform Crossover classico. I figli mescolano i tratti dei genitori."""

    # Ogni gene ha il 50% di probabilità di essere ereditato da uno dei due genitori, creando così una combinazione unica.
    child1, child2 = list(parent1), list(parent2)

    # Per ogni gene (parametro), decidiamo casualmente se scambiarlo tra i due figli o lasciarlo invariato, mantenendo così la diversità genetica.
    for i in range(4): # Per ogni cromosoma (parametro)
        if random.random() > 0.5:
            # Scambiamo il gene i tra i due figli, creando così nuove combinazioni di tratti.
            child1[i], child2[i] = child2[i], child1[i]
    # Restituiamo i due figli come tuple, pronti per essere valutati nella prossima generazione.
    return tuple(child1), tuple(child2)

# mutate() simula la mutazione genetica, introducendo variazioni casuali nei tratti di un individuo con una certa probabilità.
def mutate(individual):
    """Mutazione: pura probabilità darwiniana applicata a OGNI SINGOLO gene in modo indipendente."""

    # Ogni gene ha una probabilità fissa (MUTATION_RATE) di essere mutato, cioè sostituito da un nuovo valore casuale preso dallo spazio di ricerca.
    ind = list(individual)
    if random.random() < MUTATION_RATE: ind[0] = random.choice(CACHE_SIZES)
    if random.random() < MUTATION_RATE: ind[1] = random.choice(JOURNAL_INTERVALS)
    if random.random() < MUTATION_RATE: ind[2] = random.choice(COMPRESSORS)
    if random.random() < MUTATION_RATE: ind[3] = random.choice(DISTRIBUTIONS)

    # Restituiamo l'individuo mutato come tupla, pronto per essere valutato nella prossima generazione.
    return tuple(ind)

# ==========================================
# GENERAZIONE GRAFICI MATRICE (3x3)
# ==========================================
def plot_ea_master(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(len(COMPRESSORS), len(DISTRIBUTIONS), figsize=(20, 15))
    fig.suptitle("Evolutionary Algorithm - Mappa Topografica Globale\n(Grigio: Specie Estinte | Bianco: Individuo Alfa | Frecce: Salti Generazionali)", fontsize=20, fontweight='bold')

    vmin = df['throughput'].min()
    vmax = df['throughput'].max()
    contour_plot = None

    if not df.empty:
        max_row = df.loc[df['throughput'].idxmax()]
        absolute_best_eval_id = max_row['evaluation_id']
    else:
        absolute_best_eval_id = -1

    for r, comp in enumerate(COMPRESSORS):
        for c, dist in enumerate(DISTRIBUTIONS):
            ax = axes[r, c]
            df_dist = df[df['distribution'] == dist]
            df_plane = df_dist[df_dist['compressor'] == comp]
            
            if not df_plane.empty and len(df_plane.groupby(['cache_GB', 'journal_ms'])) >= 2:
                df_grouped = df_plane.groupby(['cache_GB', 'journal_ms'])['throughput'].mean(numeric_only=True).reset_index()
                x_coords = df_grouped['cache_GB'].map(cache_map).values
                y_coords = df_grouped['journal_ms'].map(journal_map).values
                z_vals = df_grouped['throughput'].values

                grid_x, grid_y = np.mgrid[0:len(CACHE_SIZES)-1:100j, 0:len(JOURNAL_INTERVALS)-1:100j]
                try:
                    grid_z = griddata((x_coords, y_coords), z_vals, (grid_x, grid_y), method='linear')
                    grid_z_nearest = griddata((x_coords, y_coords), z_vals, (grid_x, grid_y), method='nearest')
                    grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)
                    contour_plot = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdYlGn', alpha=0.5, vmin=vmin, vmax=vmax)
                except Exception:
                    pass

            texts = []
            
            # --- SPECIE ESTINTE (Pallini Grigi) ---
            df_regular = df_plane[df_plane['is_best_of_gen'] == False]
            if not df_regular.empty:
                df_reg_grouped = df_regular.groupby(['cache_GB', 'journal_ms'])['generation'].apply(
                    lambda x: ', '.join([f"G{int(gen)}" for gen in sorted(x.unique())])
                ).reset_index()
                
                for _, row in df_reg_grouped.iterrows():
                    ex = cache_map[row['cache_GB']]
                    ey = journal_map[row['journal_ms']]
                    eval_labels = row['generation']
                    ax.scatter(ex, ey, color='gray', s=50, alpha=0.6, zorder=4)
                    t = ax.text(ex, ey - 0.12, eval_labels, ha='center', va='top', fontsize=9, color='dimgray', zorder=5)
                    texts.append(t)

            # --- INDIVIDUI ALFA (Percorso Globale ed Evolutivo) ---
            df_best_global = df[df['is_best_of_gen'] == True].groupby(['generation', 'evaluation_id', 'cache_GB', 'journal_ms', 'compressor', 'distribution']).mean(numeric_only=True).reset_index().sort_values('generation')
            path_gens = df_best_global['generation'].tolist()
            path_c = df_best_global['cache_GB'].tolist()
            path_j = df_best_global['journal_ms'].tolist()
            path_comp = df_best_global['compressor'].tolist()
            path_dist = df_best_global['distribution'].tolist()
            path_evals = df_best_global['evaluation_id'].tolist()
            
            for i in range(len(path_gens)):
                if path_comp[i] == comp and path_dist[i] == dist: 
                    px = cache_map[path_c[i]]
                    py = journal_map[path_j[i]]
                    
                    if path_evals[i] == absolute_best_eval_id:
                        # Stella dorata — annotate ancorato al punto, NON in texts
                        # così adjust_text non lo sposta mai fuori dalla stella.
                        ax.scatter(px, py, color='gold', marker='*', s=1200, zorder=12,
                                   edgecolors='black', linewidth=1.5)
                        ax.annotate(f"G{int(path_gens[i])}", xy=(px, py),
                                    ha='center', va='center',
                                    fontsize=7, fontweight='bold', color='black',
                                    xycoords='data', zorder=14,
                                    annotation_clip=False)
                        # NON aggiungiamo a texts → adjust_text non lo tocca
                    else:
                        # Cerchio bianco per gli Alfa di generazione
                        ax.scatter(px, py, color='white', s=150, zorder=8, edgecolors='black', linewidth=1.5)
                        t = ax.text(px, py, f"G{int(path_gens[i])}", ha='center', va='center', fontsize=10, fontweight='bold', color='black', zorder=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1, alpha=0.9))
                        texts.append(t)
                    
                    # Frecce dell'evoluzione (da una generazione all'altra se nello stesso riquadro)
                    if i < len(path_gens) - 1 and path_comp[i+1] == comp and path_dist[i+1] == dist:
                        nx = cache_map[path_c[i+1]]
                        ny = journal_map[path_j[i+1]]
                        if px != nx or py != ny:
                            ax.annotate("", xy=(nx, ny), xytext=(px, py), arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.4", color="darkblue", lw=2.5, shrinkA=12, shrinkB=12, connectionstyle="arc3,rad=0.2"), zorder=6)

            if texts:
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.7))

            ax.set_xticks(range(len(CACHE_SIZES))); ax.set_xticklabels(CACHE_SIZES)
            ax.set_yticks(range(len(JOURNAL_INTERVALS))); ax.set_yticklabels(JOURNAL_INTERVALS)
            if r == len(COMPRESSORS) - 1: ax.set_xlabel("Cache Size (GB)", fontsize=12)
            if c == 0: ax.set_ylabel(f"Compressor: {comp.upper()}\nJournal Interval (ms)", fontsize=12, fontweight='bold')
            if r == 0: ax.set_title(f"Distribuzione: {dist.upper()}", fontsize=14, fontweight='bold')
            ax.set_xlim(-0.3, len(CACHE_SIZES)-0.7); ax.set_ylim(-0.3, len(JOURNAL_INTERVALS)-0.7)
            ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, right=0.92, hspace=0.3) 
    if contour_plot:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7]) 
        fig.colorbar(contour_plot, cax=cbar_ax, orientation='vertical').set_label('Throughput (ops/sec)', fontsize=14)
    
    plt.savefig(f"{output_prefix}_Griglia_EA.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_EA.pdf",           bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_EA.svg",           bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN LOOP (ALGORITMO EVOLUTIVO)
# ==========================================
def main():
    print(f"🧬 PARTENZA ALGORITMO: ALGORITMO EVOLUTIVO (Workload {WORKLOAD_TYPE})")
    
    # ⚠️ DEBUG: Verifica che le variabili d'ambiente siano impostate correttamente
    master_dir = get_master_dir()
    env_var = os.environ.get("BENCHMARK_MASTER_DIR")
    print(f"   [DEBUG] BENCHMARK_MASTER_DIR (env var): {env_var}")
    print(f"   [DEBUG] get_master_dir() ritorna: {master_dir}")
    print(f"   [DEBUG] BASE_DIR sarà: {BASE_DIR}")
    
    # Creazione della cartella per i risultati, con timestamp per evitare sovrascritture e mantenere ordine cronologico.
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"RESULTS_EA_WL_{WORKLOAD_TYPE}_{ts}.csv")
    
    # Creazione del file CSV con intestazione, pronto per accogliere i risultati di tutte le valutazioni degli individui nel corso delle generazioni.
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "evaluation_id", "cache_GB", "journal_ms", "compressor", "distribution", "repetition", "duration", "throughput", "is_best_of_gen", "elapsed_minutes"])

    start_time_global = time.time()
    
    # Dizionario per la memoization (Salva i genotipi già valutati)
    visited_points = {}
    eval_id = 1
    
    print("\n--- FASE 1: GENESI DELLA POPOLAZIONE INIZIALE ---")

    # Creazione della popolazione iniziale con individui generati casualmente,
    # ognuno con un DNA unico che rappresenta una combinazione di parametri da testare.
    population = [create_random_individual() for _ in range(POPULATION_SIZE)]
    
    # Loop delle Generazioni
    for gen in range(1, GENERATIONS + 1):
        print(f"\n🌱 GENERAZIONE {gen}/{GENERATIONS}")
        
        # Lista per memorizzare i risultati di fitness (throughput) di ogni individuo in questa generazione,
        # insieme al suo DNA e ID di valutazione.
        fitness_scores = []
        best_gen_thr = -1
        best_gen_ind = None
        best_gen_eval_id = -1
        
        # 1. Valutazione Fitness (throughput) per tutta la popolazione
        for ind in population:

            # Hard stop di sicurezza nel caso in cui il budget non sia un multiplo perfetto
            if eval_id > MAX_EVALUATIONS:
                print(f"\n🛑 Raggiunto il limite globale ({MAX_EVALUATIONS}). Fine evoluzione anticipata.")
                break

            thr = evaluate_individual(ind, visited_points, csv_filename, gen, eval_id, start_time_global)
            fitness_scores.append((thr, ind, eval_id))
            
            # Troviamo l'individuo più forte della generazione corrente
            if thr > best_gen_thr:
                best_gen_thr = thr
                best_gen_ind = ind
                best_gen_eval_id = eval_id
            
            eval_id += 1
            
        print(f"   🏆 L'ALFA della Gen {gen} è Ind{best_gen_eval_id} con {best_gen_thr:.2f} ops/sec")
        
        # Aggiorniamo il CSV indicando chi è l'Alfa
        df = pd.read_csv(csv_filename)
        df.loc[df['evaluation_id'] == best_gen_eval_id, 'is_best_of_gen'] = True
        df.to_csv(csv_filename, index=False)
        
        # Se è l'ultima generazione, si ferma l'evoluzione
        if gen == GENERATIONS: break
            
        print(f"   💞 Torneo, Accoppiamento e Mutazione (Tasso Fisso: {MUTATION_RATE})...")
        
        # 2. Selezione ed Elitarismo
        # Il migliore di questa generazione (Alfa) viene automaticamente promosso alla prossima generazione,
        # garantendo così che i tratti più forti non vadano persi.
        new_population = [best_gen_ind] 
        
        # 3. Riproduzione per generare la nuova stirpe, fino a raggiungere la dimensione della popolazione desiderata
        while len(new_population) < POPULATION_SIZE:
            # Selezione tramite Torneo Naturale (Prendi 3 a caso, vincono i 2 migliori)
            tournament = random.sample(fitness_scores, min(3, len(fitness_scores)))

            # Ordiniamo i partecipanti al torneo in base alla loro forza (throughput)
            # e prendiamo i 2 migliori come genitori per il crossover.
            tournament.sort(key=lambda x: x[0], reverse=True) 

            # I genitori sono i DNA dei 2 migliori individui del torneo, pronti per mescolare i loro tratti e creare nuova vita evolutiva.
            parent1, parent2 = tournament[0][1], tournament[1][1]
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutazione
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            # Aggiungiamo i figli alla nuova popolazione, assicurandoci di non superare la dimensione massima.
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)
                
        # Sostituzione generazionale
        population = new_population

    # Calcoliamo il tempo totale impiegato per completare l'algoritmo evolutivo, per avere un'idea del tempo necessario per questo tipo di esplorazione darwiniana.
    total_minutes = (time.time() - start_time_global) / 60.0
    
    print(f"\n🏁 Algoritmo Evolutivo Completato in {total_minutes:.1f} minuti.")
    
    print(f"\n📊 Generazione Grafico Mappa Topografica in corso...")
    try:
        plot_ea_master(csv_filename, os.path.join(BASE_DIR, f"HEATMAP_WL_{WORKLOAD_TYPE}_{ts}"))
        print(f"✅ Fatto! Trovi i risultati e la mappa in: {BASE_DIR}")
    except Exception as e:
        print(f"⚠️ Impossibile generare la heatmap: {e}")

if __name__ == "__main__":
    main()