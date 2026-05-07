# ==========================================
# IMPORTAZIONI DI BASE E SISTEMA
# ==========================================
import os
import time
import datetime
import csv
import random
import warnings
import math
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
# VARIABILI GLOBALI E COSTANTI SA
# ==========================================
BASE_DIR = os.path.join(master_dir, "Simulated Annealing")

T_INIT = 1.0       # (Rappresenta una tolleranza iniziale alta per cali di performance)
T_MIN = 0.001      # (Fine del raffreddamento)

# 2. Definiamo quanti tentativi fare a ogni step di temperatura
# Vogliamo un raffreddamento continuo, quindi la temperatura scende a ogni singolo test
ITER_PER_TEMP = 1

# 3. IL CALCOLO MAGICO DELL'ALPHA
# Calcoliamo l'alpha in modo che l'algoritmo tocchi T_MIN esattamente al test MAX_EVALUATIONS
ALPHA = math.pow((T_MIN / T_INIT), (1.0 / (MAX_EVALUATIONS - 1)))

print(f"🌡️ Termodinamica Calibrata: Budget={MAX_EVALUATIONS}, Alpha calcolato={ALPHA:.4f}")

# ==========================================
# FUNZIONI CORE DELL'ALGORITMO (PURISTA)
# ==========================================
def initialize_sa():
    """Inizializza lo stato iniziale e la temperatura."""
    # Lo stato iniziale viene generato casualmente, pescando un valore per ogni parametro dal rispettivo spazio di ricerca globale.
    initial_state = (
        random.choice(CACHE_SIZES),
        random.choice(JOURNAL_INTERVALS),
        random.choice(COMPRESSORS),
        random.choice(DISTRIBUTIONS)
    )
    return initial_state, T_INIT

# ==========================================
# FUNZIONE DI VALUTAZIONE FITNESS (PURO)
# ==========================================
# Questa funzione valuta la configurazione attuale, gestisce la memoization per evitare valutazioni ridondanti,
def evaluate_configuration_sa(state, visited_points, csv_filename, temp, iter_num, eval_id, is_accepted, start_time_global):
    """Fitness: Valuta la configurazione nativamente e gestisce la memoization."""
    c_gb, j_ms, comp, dist = state
    
    # --- MEMOIZATION: Memoria Antica ---
    # Prima di eseguire un test completo, controlliamo se questa configurazione è già stata valutata in passato.
    if state in visited_points:
        avg_thr, avg_dur = visited_points[state]
        print(f"      -> ⏭️ Stato già noto! Recupero dalla memoria: {avg_thr:.2f} ops/sec")
        elapsed_minutes = (time.time() - start_time_global) / 60.0
        
        # Anche se è un punto già visitato, vogliamo comunque registrare questa "valutazione" nel CSV,
        # indicando che è stata accettata o meno in questa iterazione, e quanto tempo è passato dall'inizio dell'algoritmo.
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([eval_id, temp, iter_num, c_gb, j_ms, comp, dist, "mean", avg_dur, avg_thr, is_accepted, round(elapsed_minutes, 2)])
        return avg_thr

    print(f"\n   [Test SA {eval_id} | T={temp:.1f}] Valuto: C={c_gb}GB, J={j_ms}ms, Comp={comp}, Dist={dist.upper()} ...", end="", flush=True)
    
    # Esecuzione Nativa Reale da config.py
    avg_thr, avg_dur = execute_full_test(c_gb, j_ms, comp, dist)
    
    print(f"   🚀 THROUGHPUT FINALE MEDIO: {avg_thr:.2f} ops/sec\n")
    
    elapsed_minutes = (time.time() - start_time_global) / 60.0
    
    # Registriamo il risultato di questa valutazione nel CSV, insieme a tutti i dettagli e al tempo trascorso dall'inizio dell'algoritmo.
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([eval_id, temp, iter_num, c_gb, j_ms, comp, dist, "mean", avg_dur, avg_thr, is_accepted, round(elapsed_minutes, 2)])
        
    # Aggiorniamo la memoria con il risultato di questa nuova configurazione,
    # in modo che se la incontreremo di nuovo, potremo recuperare i risultati senza dover eseguire nuovamente il test completo.
    visited_points[state] = (avg_thr, avg_dur)
    
    return avg_thr

# get_neighbor(state) è la funzione che definisce la "neighborhood" per il Simulated Annealing,
# generando una configurazione vicina modificando un solo gene (parametro) alla volta.
def get_neighbor(state):
    """Neighborhood: Genera una configurazione vicina garantendo che il gene mutato sia DIVERSO da quello attuale."""
    # Prendiamo lo stato attuale e scomponiamolo nei suoi parametri.
    c_gb, j_ms, comp, dist = list(state)

    # Scegliamo casualmente quale parametro mutare, ma assicuriamoci di scegliere un nuovo valore diverso da quello attuale per quel parametro.
    param_to_mutate = random.randint(0, 3)
    
    # Per ogni parametro, creiamo una lista di opzioni valide che esclude il valore attuale, e poi scegliamo casualmente da quella lista.
    if param_to_mutate == 0:
        # Scegliamo un valore casuale tra tutti quelli possibili TRANNE quello attuale
        opzioni_valide = [x for x in CACHE_SIZES if x != c_gb]
        if opzioni_valide: c_gb = random.choice(opzioni_valide)
        
    elif param_to_mutate == 1:
        opzioni_valide = [x for x in JOURNAL_INTERVALS if x != j_ms]
        if opzioni_valide: j_ms = random.choice(opzioni_valide)
        
    elif param_to_mutate == 2:
        opzioni_valide = [x for x in COMPRESSORS if x != comp]
        if opzioni_valide: comp = random.choice(opzioni_valide)
        
    elif param_to_mutate == 3:
        opzioni_valide = [x for x in DISTRIBUTIONS if x != dist]
        if opzioni_valide: dist = random.choice(opzioni_valide)
        
    # Restituiamo il nuovo stato vicino, che differisce dal precedente per un solo parametro, pronto per essere valutato.
    return (c_gb, j_ms, comp, dist)

# ==========================================
# GENERAZIONE GRAFICI MATRICE (3x3)
# ==========================================
def plot_sa_master(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(len(COMPRESSORS), len(DISTRIBUTIONS), figsize=(25, 18))
    fig.suptitle("Simulated Annealing - Mappa Topografica Globale\n(Grigio: Scartati | Bianco: Stati Accettati | Freccia Blu: Miglioramento | Freccia Arancione: Peggioramento Accettato)", fontsize=20, fontweight='bold')

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
            
            # --- PUNTI RIFIUTATI (Pallini Grigi) ---
            df_rejected = df_plane[df_plane['accepted'] == False]
            if not df_rejected.empty:
                df_rej_grouped = df_rejected.groupby(['cache_GB', 'journal_ms'])['evaluation_id'].apply(
                    lambda x: ', '.join([f"T{int(eval)}" for eval in sorted(x.unique())])
                ).reset_index()
                
                for _, row in df_rej_grouped.iterrows():
                    ex = cache_map[row['cache_GB']]
                    ey = journal_map[row['journal_ms']]
                    eval_labels = row['evaluation_id']
                    
                    ax.scatter(ex, ey, color='gray', s=50, alpha=0.6, zorder=4)
                    t = ax.text(ex, ey - 0.12, eval_labels, ha='center', va='top', fontsize=9, color='dimgray', zorder=5)
                    texts.append(t)

            # --- PERCORSO DEGLI STATI ACCETTATI (Bianchi, Stella e Frecce Bicolore) ---
            df_accepted_global = df[df['accepted'] == True].groupby(['evaluation_id', 'cache_GB', 'journal_ms', 'compressor', 'distribution', 'throughput']).mean(numeric_only=True).reset_index().sort_values('evaluation_id')
            path_evals = df_accepted_global['evaluation_id'].tolist()
            path_c = df_accepted_global['cache_GB'].tolist()
            path_j = df_accepted_global['journal_ms'].tolist()
            path_comp = df_accepted_global['compressor'].tolist()
            path_dist = df_accepted_global['distribution'].tolist()
            path_thr = df_accepted_global['throughput'].tolist()
            
            for i in range(len(path_evals)):
                if path_comp[i] == comp and path_dist[i] == dist: 
                    px = cache_map[path_c[i]]
                    py = journal_map[path_j[i]]
                    
                    if path_evals[i] == absolute_best_eval_id:
                        # Stella dorata — annotate ancorato al punto, NON in texts
                        # così adjust_text non lo sposta mai fuori dalla stella.
                        ax.scatter(px, py, color='gold', marker='*', s=1200, zorder=12,
                                   edgecolors='black', linewidth=1.5)
                        ax.annotate(f"T{int(path_evals[i])}", xy=(px, py),
                                    ha='center', va='center',
                                    fontsize=7, fontweight='bold', color='black',
                                    xycoords='data', zorder=14,
                                    annotation_clip=False)
                        # NON aggiungiamo a texts → adjust_text non lo tocca
                    else:
                        # PALLINO BIANCO (Stato Accettato)
                        ax.scatter(px, py, color='white', s=150, zorder=8, edgecolors='black', linewidth=1.5)
                        t = ax.text(px, py, f"T{int(path_evals[i])}", ha='center', va='center', fontsize=10, fontweight='bold', color='black', zorder=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1, alpha=0.9))
                        texts.append(t)
                    
                    # Frecce: Blu per Miglioramento, Arancione per Peggioramento accettato
                    if i < len(path_evals) - 1 and path_comp[i+1] == comp and path_dist[i+1] == dist:
                        nx = cache_map[path_c[i+1]]
                        ny = journal_map[path_j[i+1]]
                        if px != nx or py != ny:
                            is_improvement = path_thr[i+1] > path_thr[i]
                            arrow_color = "darkblue" if is_improvement else "darkorange"
                            ax.annotate("", xy=(nx, ny), xytext=(px, py), arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.4", color=arrow_color, lw=2.5, shrinkA=12, shrinkB=12, connectionstyle="arc3,rad=0.2"), zorder=6)

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
    
    plt.savefig(f"{output_prefix}_Griglia_SA.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_SA.pdf",           bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_SA.svg",           bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN LOOP (SIMULATED ANNEALING PURO)
# ==========================================
def main():
    print(f"🌡️ PARTENZA ALGORITMO: SIMULATED ANNEALING (Workload {WORKLOAD_TYPE})")
    
    # Creiamo la directory dei risultati se non esiste, e prepariamo il file CSV per registrare tutte le valutazioni, accettazioni e tempi.
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"RESULTS_SA_WL_{WORKLOAD_TYPE}_{ts}.csv")
    
    # Scriviamo l'intestazione del CSV, che conterrà tutte le informazioni rilevanti per ogni valutazione, inclusi i parametri testati,
    # la throughput ottenuta, se è stata accettata o meno, e quanto tempo è passato dall'inizio dell'algoritmo.
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation_id", "temperature", "iteration", "cache_GB", "journal_ms", "compressor", "distribution", "repetition", "duration", "throughput", "accepted", "elapsed_minutes"])

    start_time_global = time.time()
    visited_points = {}
    eval_id = 1
    best_thr_global = -1
    
    # 1. Genesi: Stato Iniziale e Temperatura Iniziale
    print("\n--- FASE 1: GENESI DELLO STATO INIZIALE A CALDO ---")
    current_state, current_temp = initialize_sa()
    # Il primo stato è accettato per forza
    current_thr = evaluate_configuration_sa(current_state, visited_points, csv_filename, current_temp, 0, eval_id, True, start_time_global)
    
    # Impostiamo la best_thr_global al primo stato, che è il nostro punto di partenza e il primo miglioramento assoluto trovato.
    best_thr_global = current_thr
    eval_id += 1
    
    print(f"\n--- FASE 2: CICLO DI RAFFREDDAMENTO GEOMETRICO (ALPHA={ALPHA}) ---")
    
    # Loop Principale — usiamo un CONTATORE esplicito invece di "current_temp > T_MIN"
    # perché il confronto floating point è instabile: ALPHA^(MAX_EVALUATIONS-1) non
    # raggiunge T_MIN esattamente e il while potrebbe girare una volta in più o in meno.
    # Con il contatore il numero di valutazioni è garantito = MAX_EVALUATIONS (1 iniziale
    # + MAX_EVALUATIONS-1 nel loop).
    cooling_steps = MAX_EVALUATIONS - 1   # iterazioni rimanenti dopo la valutazione iniziale
    step_count = 0

    while step_count < cooling_steps:

        # 5. Raffreddamento Geometrico → all'inizio così la temperatura usata per
        # il criterio di Metropolis decresce correttamente ad ogni passo.
        current_temp *= ALPHA
        step_count += 1

        print(f"\n❄️ Temperatura attuale: {current_temp:.4f}")
        
        # Loop interno: Tentativi a temperatura costante
        for i in range(1, ITER_PER_TEMP + 1):
            
            # 2. Neighborhood: Generazione del Vicino
            neighbor_state = get_neighbor(current_state)
            
            # Valutiamo prima come "False" (non ancora accettato)
            neighbor_thr = evaluate_configuration_sa(neighbor_state, visited_points, csv_filename, current_temp, i, eval_id, False, start_time_global)
            
            # 3. Calcoliamo la variazione PERCENTUALE (es. 0.05 significa che il vicino è peggiore del 5%)
            if current_thr > 0:
                delta_throughput = (current_thr - neighbor_thr) / current_thr
            else:
                delta_throughput = 0.0
            
            # 4. Criterio di Accettazione di Metropolis (PURO)
            accepted = False
            if delta_throughput < 0:
                # Caso A: Il vicino è MIGLIORE. Lo accettiamo sempre.
                print(f"      -> ✅ Vicino MIGLIORE accettato (Thr: {neighbor_thr:.2f})")
                accepted = True
                
                # Se questo vicino migliore è anche il nuovo record globale, aggiorniamo la best_thr_global e stampiamo un messaggio di celebrazione.
                if neighbor_thr > best_thr_global:
                    best_thr_global = neighbor_thr
                    print(f"         🏆 NUOVO RECORD GLOBALE TROVATO: {best_thr_global:.2f} ops/sec!")
            else:
                # Caso B: Il vicino è PEGGIORE. Applichiamo la probabilità di Metropolis.
                # La probabilità di accettare un vicino peggiore diminuisce all'aumentare della differenza di throughput (delta_throughput) 
                # e al diminuire della temperatura (current_temp).
                acceptance_probability = math.exp(-delta_throughput / current_temp)
                
                # Generiamo un numero casuale tra 0 e 1, e se è inferiore alla probabilità di accettazione,
                # accettiamo comunque questo vicino peggiore, permettendo così all'algoritmo di esplorare soluzioni che altrimenti verrebbero scartate,
                # e potenzialmente superare i massimi locali.
                if random.random() < acceptance_probability:
                    print(f"      -> 🎲 Vicino PEGGIORE accettato per Metropolis (Prob: {acceptance_probability * 100:.2f}%)")
                    accepted = True
                # Se non accettiamo il vicino peggiore, stampiamo un messaggio che indica che è stato rifiutato,
                # insieme alla probabilità di accettazione che è stata calcolata.
                else:
                    print(f"      -> ❌ Vicino peggiore rifiutato (Prob: {acceptance_probability * 100:.2f}%)")

            # Se lo abbiamo accettato (sia per miglioramento che per Metropolis), aggiorniamo il CSV e il nostro stato
            if accepted:
                current_state = neighbor_state
                current_thr = neighbor_thr
                
                # Aggiorniamo il CSV per questo vicino accettato, modificando la riga corrispondente a questa valutazione (eval_id) per indicare che è stata accettata.
                df = pd.read_csv(csv_filename)
                df.loc[df['evaluation_id'] == eval_id, 'accepted'] = True
                df.to_csv(csv_filename, index=False)

            eval_id += 1

    # Calcoliamo il tempo totale impiegato per completare l'algoritmo di Simulated Annealing, per avere un'idea del tempo necessario per questo tipo di esplorazione guidata.
    total_minutes = (time.time() - start_time_global) / 60.0

    print(f"\n🏁 Simulated Annealing Completato in {total_minutes:.1f} minuti.")
    print(f"🏆 OTTIMO ASSOLUTO TROVATO: Throughput: {best_thr_global:.2f} ops/sec")
    
    print(f"\n📊 Generazione Grafico Mappa Topografica in corso...")
    try:
        plot_sa_master(csv_filename, os.path.join(BASE_DIR, f"HEATMAP_WL_{WORKLOAD_TYPE}_{ts}"))
        print(f"✅ Fatto! Trovi i risultati e la mappa in: {BASE_DIR}")
    except Exception as e:
        print(f"⚠️ Impossibile generare la heatmap: {e}")

if __name__ == "__main__":
    main()