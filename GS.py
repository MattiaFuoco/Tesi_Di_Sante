# ==========================================
# IMPORTAZIONI DI BASE E SISTEMA
# ==========================================
import os
import time
import datetime
import csv
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# ==========================================
# LIBRERIE PER GRAFICI 3x3 (HEATMAP)
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, Rbf

# ==========================================
# IMPORTAZIONE DAL "CERVELLO CENTRALE"
# ==========================================
from config import *

# ==========================================
# VARIABILI GLOBALI E COSTANTI
# ==========================================

# ==========================================
# GENERAZIONE GRAFICI MATRICE (3x3)
# (Questa funzione rimane INVARIATA come richiesto)
# ==========================================
def plot_gs_master(csv_file, output_prefix, global_vmin=None, global_vmax=None):
    df = pd.read_csv(csv_file)
    if 'throughput' in df.columns and 'throughput_avg' not in df.columns:
        df = df.rename(columns={'throughput': 'throughput_avg'})
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(len(COMPRESSORS), len(DISTRIBUTIONS), figsize=(20, 15))

    # Aggreghiamo i dati per trovare la singola configurazione migliore in assoluto
    df_agg = df.groupby(['cache_GB', 'journal_ms', 'compressor', 'distribution'])['throughput_avg'].mean(numeric_only=True).reset_index()
    vmin = global_vmin if global_vmin is not None else df_agg['throughput_avg'].min()
    vmax = global_vmax if global_vmax is not None else df_agg['throughput_avg'].max()
    
    best_row = df_agg.loc[df_agg['throughput_avg'].idxmax()]
    best_c = best_row['cache_GB']
    best_j = best_row['journal_ms']
    best_comp = best_row['compressor']
    best_dist = best_row['distribution']
    best_thr = best_row['throughput_avg']

    contour_plot = None

    for r, comp in enumerate(COMPRESSORS):
        for c, dist in enumerate(DISTRIBUTIONS):
            ax = axes[r, c]
            df_dist = df_agg[df_agg['distribution'] == dist]
            df_plane = df_dist[df_dist['compressor'] == comp]
            
            if not df_plane.empty and len(df_plane) >= 2:
                x_coords = df_plane['cache_GB'].map(cache_map).values
                y_coords = df_plane['journal_ms'].map(journal_map).values
                z_vals = df_plane['throughput_avg'].values

                grid_x, grid_y = np.mgrid[-0.3:len(CACHE_SIZES)-0.7:100j, -0.3:len(JOURNAL_INTERVALS)-0.7:100j]
                try:
                    import scipy.ndimage
                    # Interpolazione "topografica" liscia su TUTTO il piano:
                    # RBF multiquadric estende organicamente fuori dall'inviluppo
                    # convesso dei punti, producendo contorni ondeggianti invece
                    # dei bordi rettangolari di griddata+nearest. Fallback su
                    # griddata se RBF fallisce (es. punti collineari).
                    try:
                        # smooth=0 â†’ RBF passa ESATTAMENTE per i punti misurati.
                        # Niente smoothing soppresso: le creste/valli reali emergono
                        # invece di essere "lisciate via" in grandi blob uniformi.
                        rbf = Rbf(x_coords, y_coords, z_vals, function='multiquadric', smooth=0)
                        grid_z = rbf(grid_x, grid_y)
                    except Exception:
                        try:
                            grid_z = griddata((x_coords, y_coords), z_vals, (grid_x, grid_y), method='cubic')
                        except Exception:
                            grid_z = griddata((x_coords, y_coords), z_vals, (grid_x, grid_y), method='linear')
                        grid_z_nearest = griddata((x_coords, y_coords), z_vals, (grid_x, grid_y), method='nearest')
                        grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)

                    # Smoothing minimo: arrotonda i contorni senza cancellare i dettagli
                    grid_z = scipy.ndimage.gaussian_filter(grid_z, sigma=0.6)

                    # Clipping al range globale + extend='both' su contourf:
                    # le oscillazioni RBF restano dentro [vmin, vmax] e ogni
                    # pixel viene colorato (niente "bolle" bianche).
                    grid_z = np.clip(grid_z, vmin, vmax)

                    contour_plot = ax.contourf(grid_x, grid_y, grid_z, levels=np.linspace(vmin, vmax, 30), cmap='RdYlGn', alpha=0.7, vmin=vmin, vmax=vmax, extend='both')
                except Exception:
                    pass
                
                # Disegniamo i micro-pallini neri per mostrare i punti testati
                ax.scatter(x_coords, y_coords, color='black', s=10, alpha=0.3, zorder=5)
                
                # Se l'OTTIMO ASSOLUTO si trova in QUESTO quadrante, disegniamo la stella
                if comp == best_comp and dist == best_dist:
                    ex = cache_map[best_c]
                    ey = journal_map[best_j]
                    ax.scatter(ex, ey, color='gold', s=600, marker='*', zorder=10, edgecolors='black', linewidth=1.5)
                    ax.text(ex, ey + 0.15, f"OTTIMO GLOBALE\n{best_thr:.1f} ops/s", ha='center', va='bottom', 
                            fontsize=11, fontweight='bold', color='black', zorder=11,
                            bbox=dict(boxstyle="round,pad=0.2", fc="gold", ec="black", lw=1, alpha=0.9))

            ax.set_xticks(range(len(CACHE_SIZES))); ax.set_xticklabels(CACHE_SIZES)
            ax.set_yticks(range(len(JOURNAL_INTERVALS))); ax.set_yticklabels(JOURNAL_INTERVALS)
            if r == len(COMPRESSORS) - 1: ax.set_xlabel("Cache Size (GB)", fontsize=12)
            if c == 0:
                ax.set_ylabel("Journal Interval (ms)", fontsize=12, fontweight='bold')
            if c == len(DISTRIBUTIONS) - 1:
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('left')
                ax.set_ylabel(f"Compressor: {comp.upper()}", fontsize=12, fontweight='bold', rotation=-90, labelpad=15)
            if r == 0: ax.set_title(f"Distribuzione: {dist.upper()}", fontsize=14, fontweight='bold')
            ax.set_xlim(-0.3, len(CACHE_SIZES)-0.7); ax.set_ylim(-0.3, len(JOURNAL_INTERVALS)-0.7)
            ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(top=0.97, right=0.92, hspace=0.3) 
    if contour_plot:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7]) 
        fig.colorbar(contour_plot, cax=cbar_ax, orientation='vertical').set_label('Throughput (ops/sec)', fontsize=14)
    
    plt.savefig(f"{output_prefix}_Griglia_GS.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_GS.pdf",           bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_GS.svg",           bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN LOOP (GRID SEARCH)
# ==========================================
def main():
    print(f"ðŸ—„ï¸ PARTENZA ALGORITMO: GRID SEARCH (Workload {WORKLOAD_TYPE})")
    
    # âš ï¸ DEBUG: Verifica che le variabili d'ambiente siano impostate correttamente
    master_dir = get_master_dir()
    env_var = os.environ.get("BENCHMARK_MASTER_DIR")
    print(f"   [DEBUG] BENCHMARK_MASTER_DIR (env var): {env_var}")
    print(f"   [DEBUG] get_master_dir() ritorna: {master_dir}")
    
    # Calcola BASE_DIR DENTRO main() per usare il valore CORRETTO di BENCHMARK_MASTER_DIR
    BASE_DIR = os.path.join(get_master_dir(), "Grid Search")
    print(f"   [DEBUG] BASE_DIR sarÃ : {BASE_DIR}")
    
    # Creiamo la cartella dei risultati se non esiste e prepariamo il file CSV per salvare i risultati in modo pulito e strutturato.
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"RESULTS_GS_WL_{WORKLOAD_TYPE}_{ts}.csv")

    # Scriviamo l'intestazione del CSV, definendo chiaramente le colonne per una successiva analisi e visualizzazione.
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation_id", "cache_GB", "journal_ms", "compressor", "distribution", "repetition",
                "duration_avg", "duration_min", "duration_max", "duration_std",
                "throughput_avg", "throughput_min", "throughput_max", "throughput_std",
                "elapsed_minutes"])

    TOTAL_EVALUATIONS = len(CACHE_SIZES) * len(JOURNAL_INTERVALS) * len(COMPRESSORS) * len(DISTRIBUTIONS)
    print(f"âš ï¸ ATTENZIONE: Mappatura totale dello spazio in corso. Configurazioni totali: {TOTAL_EVALUATIONS}")

    start_time_global = time.time()
    eval_id = 1
    
    # Loop nidificati: Esploriamo rigorosamente ogni singola combinazione
    for dist in DISTRIBUTIONS:
        for comp in COMPRESSORS:
            for j_ms in JOURNAL_INTERVALS:
                for c_gb in CACHE_SIZES:
                    
                    print(f"\n[Test GS {eval_id}/{TOTAL_EVALUATIONS}] Valuto: C={c_gb}GB, J={j_ms}ms, Comp={comp}, Dist={dist.upper()}")
                    
                    # ====================================================================
                    # ðŸš€ LA MAGIA DELL'API: Tutto il lavoro sporco Ã¨ delegato a config.py!
                    # ====================================================================
                    avg_thr, min_thr, max_thr, std_thr, avg_dur, min_dur, max_dur, std_dur = execute_full_test(c_gb, j_ms, comp, dist)
                    
                    print(f"   ðŸš€ THROUGHPUT FINALE MEDIO: {avg_thr:.2f} ops/sec\n")
                    
                    # Calcoliamo i minuti trascorsi dall'inizio del processo, per avere un'idea del tempo totale necessario per completare la mappatura.
                    elapsed_minutes = (time.time() - start_time_global) / 60.0

                    # Salviamo ogni risultato in modo strutturato nel file CSV, con tutte le informazioni necessarie per analisi e visualizzazioni future.
                    with open(csv_filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([eval_id, c_gb, j_ms, comp, dist, "mean",
                                       avg_dur, round(min_dur, 3), round(max_dur, 3), round(std_dur, 3),
                                       avg_thr, round(min_thr, 2), round(max_thr, 2), round(std_thr, 2),
                                       round(elapsed_minutes, 2)])
                        
                    eval_id += 1
                    
    # Calcoliamo il tempo totale impiegato per completare la Grid Search, per avere un'idea del tempo necessario per questo tipo di esplorazione esaustiva.
    total_minutes = (time.time() - start_time_global) / 60.0

    print(f"\nâœ… Grid Search Completa in {total_minutes:.1f} minuti.")
    
    print(f"\nðŸ“Š Generazione Grafico Mappa Topografica Esatta in corso...")
    
    try:
        # (Generazione Heatmap spostata al termine del workload da master_plotter)

        # plot_gs_master(csv_filename, os.path.join(BASE_DIR, f"HEATMAP_WL_{WORKLOAD_TYPE}_{ts}"))
        print(f"âœ… Fatto! Trovi i risultati e la mappa in: {BASE_DIR}")
    except Exception as e:
        print(f"âš ï¸ Impossibile generare la heatmap: {e}")

if __name__ == "__main__":
    main()
