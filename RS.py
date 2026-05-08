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
# VARIABILI GLOBALI E COSTANTI
# ==========================================

# ==========================================
# GENERAZIONE GRAFICI MATRICE (3x3) CON STELLINA
# ==========================================
def plot_rs_master(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(len(COMPRESSORS), len(DISTRIBUTIONS), figsize=(20, 15))
    fig.suptitle("Random Search - Mappa Topografica Globale\n(Grigio: Tentativi | Bianco: Record intermedi | Stella Dorata: Record Assoluto)", fontsize=20, fontweight='bold')

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
            
            # --- PALLINI GRIGI (Tentativi Falliti) ---
            df_regular = df_plane[df_plane['is_new_best'] == False]
            if not df_regular.empty:
                df_reg_grouped = df_regular.groupby(['cache_GB', 'journal_ms'])['evaluation_id'].apply(
                    lambda x: ', '.join([f"T{int(eval)}" for eval in sorted(x.unique())])
                ).reset_index()
                
                for _, row in df_reg_grouped.iterrows():
                    ex = cache_map[row['cache_GB']]
                    ey = journal_map[row['journal_ms']]
                    eval_labels = row['evaluation_id']
                    
                    ax.scatter(ex, ey, color='gray', s=50, alpha=0.6, zorder=4)
                    t = ax.text(ex, ey - 0.12, eval_labels, ha='center', va='top', 
                                fontsize=9, color='dimgray', zorder=5)
                    texts.append(t)

            # --- RECORD GLOBALI (Bianchi e Stella) ---
            df_best_global = df[df['is_new_best'] == True].groupby(['evaluation_id', 'cache_GB', 'journal_ms', 'compressor', 'distribution']).mean(numeric_only=True).reset_index().sort_values('evaluation_id')
            path_evals = df_best_global['evaluation_id'].tolist()
            path_c = df_best_global['cache_GB'].tolist()
            path_j = df_best_global['journal_ms'].tolist()
            path_comp = df_best_global['compressor'].tolist()
            path_dist = df_best_global['distribution'].tolist()
            
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
                        # PALLINO BIANCO (Record intermedio)
                        ax.scatter(px, py, color='white', s=150, zorder=8, edgecolors='black', linewidth=1.5)
                        t = ax.text(px, py, f"T{int(path_evals[i])}", ha='center', va='center', 
                                    fontsize=10, fontweight='bold', color='black', zorder=10,
                                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1, alpha=0.9))
                        texts.append(t)
                    
                    # (NOTA: Frecce blu rimosse da qui come richiesto)

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
    
    plt.savefig(f"{output_prefix}_Griglia_RS.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_RS.pdf",           bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_RS.svg",           bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN LOOP (RANDOM SEARCH)
# ==========================================
def main():
    print(f"🎲 PARTENZA ALGORITMO: RANDOM SEARCH (Workload {WORKLOAD_TYPE})")
    
    # ⚠️ DEBUG: Verifica che le variabili d'ambiente siano impostate correttamente
    master_dir = get_master_dir()
    env_var = os.environ.get("BENCHMARK_MASTER_DIR")
    print(f"   [DEBUG] BENCHMARK_MASTER_DIR (env var): {env_var}")
    print(f"   [DEBUG] get_master_dir() ritorna: {master_dir}")
    
    # Calcola BASE_DIR DENTRO main() per usare il valore CORRETTO di BENCHMARK_MASTER_DIR
    BASE_DIR = os.path.join(get_master_dir(), "Random Search")
    print(f"   [DEBUG] BASE_DIR sarà: {BASE_DIR}")
    
    # Creiamo la cartella dei risultati se non esiste e prepariamo il file CSV per salvare i risultati in modo pulito e strutturato.
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"RESULTS_RS_WL_{WORKLOAD_TYPE}_{ts}.csv")
    
    # Scriviamo l'intestazione del CSV, definendo chiaramente le colonne per una successiva analisi e visualizzazione.
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation_id", "cache_GB", "journal_ms", "compressor", "distribution", "repetition", "duration", "throughput", "is_new_best", "elapsed_minutes"])
    
    # Iniziamo il processo di esplorazione casuale, tenendo traccia del tempo totale e del miglior risultato globale.
    start_time_global = time.time()
    best_thr_global = -1
    
    # Creiamo una lista di tutte le combinazioni possibili, che ci servirà per estrarre casualmente le configurazioni da testare, garantendo un'esplorazione ampia e non guidata da alcun bias.
    unvisited = [(c, j, comp, d) for c in range(len(CACHE_SIZES)) 
                                 for j in range(len(JOURNAL_INTERVALS)) 
                                 for comp in range(len(COMPRESSORS)) 
                                 for d in range(len(DISTRIBUTIONS))]
    
    # Loop fino a raggiungere il numero massimo di valutazioni o fino a esaurire tutte le combinazioni possibili.
    for eval_id in range(1, MAX_EVALUATIONS + 1):
        if not unvisited: 
            print("\n🏁 Tutte le combinazioni possibili sono state esplorate!")
            break
            
        # Scegliamo casualmente una combinazione non ancora esplorata, per garantire un'esplorazione ampia e non guidata da alcun bias.
        c_idx, j_idx, comp_idx, d_idx = random.choice(unvisited)
        unvisited.remove((c_idx, j_idx, comp_idx, d_idx))
        
        c_gb = CACHE_SIZES[c_idx]
        j_ms = JOURNAL_INTERVALS[j_idx]
        comp = COMPRESSORS[comp_idx]
        dist = DISTRIBUTIONS[d_idx]
        
        print(f"\n[Test RS {eval_id}/{MAX_EVALUATIONS}] Valuto: C={c_gb}GB, J={j_ms}ms, Comp={comp}, Dist={dist.upper()} ...", end="", flush=True)
        
        # ====================================================================
        # LA MAGIA DELL'API: Deleghiamo tutto il test a config.py
        # ====================================================================
        avg_thr, avg_dur = execute_full_test(c_gb, j_ms, comp, dist)
        
        print(f"   🚀 THROUGHPUT FINALE MEDIO: {avg_thr:.2f} ops/sec\n")
        
        # Verifichiamo se questo nuovo risultato è un nuovo record globale, aggiornando la variabile e segnalandolo nel CSV.
        is_new_best = False
        if avg_thr > best_thr_global:
            print(f"   🏆 NUOVO RECORD GLOBALE CASUALE: {avg_thr:.2f} ops/sec!")
            best_thr_global = avg_thr
            is_new_best = True
            
        elapsed_minutes = (time.time() - start_time_global) / 60.0
        
        # Salviamo ogni risultato in modo strutturato nel file CSV, con tutte le informazioni necessarie per analisi e visualizzazioni future.
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([eval_id, c_gb, j_ms, comp, dist, "mean", avg_dur, avg_thr, is_new_best, round(elapsed_minutes, 2)])

    # Calcoliamo il tempo totale impiegato per completare la Random Search, per avere un'idea del tempo necessario per questo tipo di esplorazione casuale.
    total_minutes = (time.time() - start_time_global) / 60.0

    print(f"\n✅ Random Search Completa in {total_minutes:.1f} minuti.")
    
    print(f"\n📊 Generazione Grafico Mappa Topografica in corso...")
    try:
        plot_rs_master(csv_filename, os.path.join(BASE_DIR, f"HEATMAP_WL_{WORKLOAD_TYPE}_{ts}"))
        print(f"✅ Fatto! Trovi i risultati e la mappa in: {BASE_DIR}")
    except Exception as e:
        print(f"⚠️ Impossibile generare la heatmap: {e}")

if __name__ == "__main__":
    main()