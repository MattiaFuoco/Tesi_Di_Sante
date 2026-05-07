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
# LIBRERIE PER GRAFICI E MATEMATICA
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
BASE_DIR = os.path.join(master_dir, "Hill Climbing")

# ==========================================
# VALUTAZIONE CON MEMORIA E CLEAN CSV
# ==========================================
# Questa funzione è il cuore dell'algoritmo: 
# valuta un punto specifico (configurazione) e utilizza la memoria per evitare valutazioni ridondanti.
def evaluate_point(c_idx, j_idx, comp_idx, d_idx, visited_points, csv_filename, step_num, start_time_global, is_center=False):
    global evaluations_done
    
    c_gb = CACHE_SIZES[c_idx]
    j_ms = JOURNAL_INTERVALS[j_idx]
    comp = COMPRESSORS[comp_idx]
    dist = DISTRIBUTIONS[d_idx]
    
    # --- MEMORIA (Memoization) ---
    if (c_idx, j_idx, comp_idx, d_idx) in visited_points:
        avg_thr, avg_dur = visited_points[(c_idx, j_idx, comp_idx, d_idx)]
        print(f"      -> ⏭️ Già esplorato in passato! Recupero dalla memoria: {avg_thr:.2f} ops/sec")
        elapsed_minutes = (time.time() - start_time_global) / 60.0
        
        # Aggiorniamo il CSV anche per questo punto già visitato, in modo da avere un record completo di tutte le valutazioni, anche quelle memorizzate.
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([evaluations_done + 1, step_num, c_gb, j_ms, comp, dist, "mean", avg_dur, avg_thr, False, round(elapsed_minutes, 2)])
        evaluations_done += 1
        return avg_thr
    
    evaluations_done += 1
    
    tipo_punto = "POSIZIONE ATTUALE" if is_center else "VALUTAZIONE VICINO"
    print(f"\n   [{tipo_punto} - Val. {evaluations_done}/{MAX_EVALUATIONS}] Testo: C={c_gb}GB, J={j_ms}ms, Comp={comp}, Dist={dist.upper()} ...", end="", flush=True)
    
    # ====================================================================
    # 🚀 LA MAGIA DELL'API: ESECUZIONE NATIVA REALE DA config.py
    # ====================================================================
    avg_thr, avg_dur = execute_full_test(c_gb, j_ms, comp, dist)
    
    print(f"   🚀 THROUGHPUT FINALE MEDIO: {avg_thr:.2f} ops/sec\n")
    
    elapsed_minutes = (time.time() - start_time_global) / 60.0
    
    # Aggiorniamo il CSV con i risultati di questa valutazione, indicando se è un punto del percorso (is_path) o meno, e quanto tempo è passato dall'inizio dell'algoritmo.
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([evaluations_done, step_num, c_gb, j_ms, comp, dist, "mean", avg_dur, avg_thr, False, round(elapsed_minutes, 2)])

    visited_points[(c_idx, j_idx, comp_idx, d_idx)] = (avg_thr, avg_dur)
    
    return avg_thr

# ==========================================
# GENERAZIONE DEI VICINI (ORTOGONALI)
# ==========================================
def get_all_orthogonal_neighbors(c_idx, j_idx, comp_idx, d_idx):
    """
    Hill Climbing genera TUTTI i vicini ortogonali per valutarli prima di decidere.
    """
    neighbors = []
    if c_idx > 0: neighbors.append((c_idx - 1, j_idx, comp_idx, d_idx))
    if c_idx < len(CACHE_SIZES) - 1: neighbors.append((c_idx + 1, j_idx, comp_idx, d_idx))
    if j_idx > 0: neighbors.append((c_idx, j_idx - 1, comp_idx, d_idx))
    if j_idx < len(JOURNAL_INTERVALS) - 1: neighbors.append((c_idx, j_idx + 1, comp_idx, d_idx))
    if comp_idx > 0: neighbors.append((c_idx, j_idx, comp_idx - 1, d_idx))
    if comp_idx < len(COMPRESSORS) - 1: neighbors.append((c_idx, j_idx, comp_idx + 1, d_idx))
    if d_idx > 0: neighbors.append((c_idx, j_idx, comp_idx, d_idx - 1))
    if d_idx < len(DISTRIBUTIONS) - 1: neighbors.append((c_idx, j_idx, comp_idx, d_idx + 1))
    return neighbors

# ==========================================
# GENERAZIONE GRAFICI MATRICE (3x3)
# ==========================================
def plot_steepest_hc_master(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(len(COMPRESSORS), len(DISTRIBUTIONS), figsize=(20, 15))
    fig.suptitle("Hill Climbing - Mappa Topografica\n(Punti grigi = Vicini valutati | Punti Bianchi e Frecce = Vetta scalata)", fontsize=20, fontweight='bold')

    vmin = df['throughput'].min()
    vmax = df['throughput'].max()
    contour_plot = None

    # Identifica lo step del percorso con throughput assoluto massimo (per la stellina)
    df_path_all = df[df['is_path'] == True]
    if not df_path_all.empty:
        best_path_row = df_path_all.loc[df_path_all['throughput'].idxmax()]
        absolute_best_step = int(best_path_row['step'])
    else:
        absolute_best_step = -1

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
            
            # Pallini Grigi (Vicini valutati e scartati)
            df_discarded = df_plane[df_plane['is_path'] == False]
            if not df_discarded.empty:
                df_discarded_grouped = df_discarded.groupby(['cache_GB', 'journal_ms']).mean(numeric_only=True).reset_index()
                for _, row in df_discarded_grouped.iterrows():
                    ex = cache_map[row['cache_GB']]
                    ey = journal_map[row['journal_ms']]
                    ax.scatter(ex, ey, color='gray', s=50, alpha=0.6, zorder=4)

            # Pallini Bianchi e Frecce (Il percorso effettivo dell'alpinista)
            df_path = df[df['is_path'] == True].sort_values('step')
            path_points = df_path.groupby('step').first().reset_index()
            
            for i in range(len(path_points)):
                row = path_points.iloc[i]
                if row['compressor'] == comp and row['distribution'] == dist:
                    px = cache_map[row['cache_GB']]
                    py = journal_map[row['journal_ms']]
                    step_num = int(row['step'])
                    
                    is_best = (step_num == absolute_best_step)

                    if is_best:
                        # Stella dorata — annotate ancorato al punto, NON in texts
                        # così adjust_text non lo sposta mai fuori dalla stella.
                        ax.scatter(px, py, color='gold', marker='*', s=1200, zorder=12, edgecolors='black', linewidth=1.5)
                        ax.annotate(f"S{step_num}", xy=(px, py),
                                    ha='center', va='center',
                                    fontsize=7, fontweight='bold', color='black',
                                    xycoords='data', zorder=14,
                                    annotation_clip=False)
                        # NON aggiungiamo a texts → adjust_text non lo tocca
                    else:
                        ax.scatter(px, py, color='white', s=150, zorder=8, edgecolors='black', linewidth=1.5)
                        t = ax.text(px, py, f"S{step_num}", ha='center', va='center',
                                    fontsize=10, fontweight='bold', color='black', zorder=10,
                                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1, alpha=0.9))
                        texts.append(t)
                    
                    # Disegna la freccia solo se il passo successivo è nello stesso riquadro visivo
                    if i < len(path_points) - 1:
                        next_row = path_points.iloc[i+1]
                        if next_row['compressor'] == comp and next_row['distribution'] == dist:
                            nx = cache_map[next_row['cache_GB']]
                            ny = journal_map[next_row['journal_ms']]
                            if px != nx or py != ny:
                                ax.annotate("", xy=(nx, ny), xytext=(px, py), 
                                            arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.4", 
                                                            color="darkblue", lw=2.5, shrinkA=12, shrinkB=12), zorder=6)

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
    
    plt.savefig(f"{output_prefix}_Griglia_HC.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_HC.pdf",           bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_HC.svg",           bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN LOOP (HILL CLIMBING)
# ==========================================
evaluations_done = 0 

def main():
    global evaluations_done
    evaluations_done = 0
    
    # Creazione del CSV con intestazione
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"results_HC_WL_{WORKLOAD_TYPE}_{ts}.csv")

    # Scriviamo l'intestazione del CSV prima di iniziare le valutazioni,
    # in modo da avere un file pronto per essere popolato con i risultati di ogni punto valutato, incluso quelli recuperati dalla memoria.
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation_id", "step", "cache_GB", "journal_ms", "compressor", "distribution", "repetition", "duration", "throughput", "is_path", "elapsed_minutes"])

    print(f"🚀 PARTENZA ALGORITMO: HILL CLIMBING (Workload {WORKLOAD_TYPE})")

    start_time_global = time.time()

    # Per tenere traccia dei punti già visitati e dei loro risultati, utilizziamo un dizionario "visited_points"
    # che mappa ogni configurazione (c_idx, j_idx, comp_idx, d_idx) alla sua throughput media e durata media.
    visited_points = {}
    final_path = []
    
    # Partiamo dal punto centrale della matrice, che è una scelta ragionevole per iniziare la scalata,
    # in quanto ci permette di esplorare in tutte le direzioni.
    current_c_idx = len(CACHE_SIZES) // 2       
    current_j_idx = len(JOURNAL_INTERVALS) // 2 
    current_comp_idx = len(COMPRESSORS) // 2    
    current_d_idx = len(DISTRIBUTIONS) // 2     
    
    step = 1
    
    # Il ciclo principale continua finché non raggiungiamo il limite massimo di valutazioni (budget)
    # o finché non troviamo un ottimo locale.
    while evaluations_done < MAX_EVALUATIONS:
        print(f"\n" + "="*50)
        print(f"🎯 INIZIO STEP {step} (Valutazioni Totali: {evaluations_done}/{MAX_EVALUATIONS})")
        print("="*50)
        
        # 1. Valutiamo il punto in cui ci troviamo
        best_thr = evaluate_point(current_c_idx, current_j_idx, current_comp_idx, current_d_idx, visited_points, csv_filename, step, start_time_global, is_center=True)
        # Inizialmente, il miglior vicino in assoluto è il punto centrale stesso, che è il nostro punto di partenza.
        best_c_idx, best_j_idx, best_comp_idx, best_d_idx = current_c_idx, current_j_idx, current_comp_idx, current_d_idx
        
        # Registriamo che questo punto fa parte del percorso definitivo
        final_path.append((step, CACHE_SIZES[current_c_idx], JOURNAL_INTERVALS[current_j_idx], COMPRESSORS[current_comp_idx], DISTRIBUTIONS[current_d_idx]))
        
        # 2. Generiamo e TESTIAMO TUTTI i vicini ortogonali
        neighbors = get_all_orthogonal_neighbors(current_c_idx, current_j_idx, current_comp_idx, current_d_idx)
        found_better = False
        
        # Valutiamo tutti i vicini per trovare il miglioramento ASSOLUTO maggiore, e solo alla fine decidiamo se fare il passo verso quel vicino migliore.
        for n_c, n_j, n_comp, n_d in neighbors:

            # Prima di valutare questo vicino, controlliamo se abbiamo già raggiunto il limite massimo di valutazioni (budget).
            if evaluations_done >= MAX_EVALUATIONS:
                print("\n⚠️ Raggiunto il limite massimo di valutazioni (Budget = 40). Interruzione forzata.")
                break
                
            # Valutiamo questo vicino, che restituirà la sua throughput media. La funzione "evaluate_point" si occuperà di gestire la memoria e il CSV.
            thr = evaluate_point(n_c, n_j, n_comp, n_d, visited_points, csv_filename, step, start_time_global, is_center=False)

            # Se questo vicino ha una throughput migliore del miglior vicino trovato finora, aggiorniamo il miglior vicino in assoluto.
            if thr > best_thr:
                best_thr = thr
                best_c_idx, best_j_idx, best_comp_idx, best_d_idx = n_c, n_j, n_comp, n_d
                found_better = True
        
        # 3. Decisione: Facciamo il passo solo DOPO aver guardato tutti i vicini
        if found_better:
            print(f"\n✅ Il vicino migliore in assoluto è: Cache={CACHE_SIZES[best_c_idx]}GB | Journal={JOURNAL_INTERVALS[best_j_idx]}ms | Comp={COMPRESSORS[best_comp_idx]} | Dist={DISTRIBUTIONS[best_d_idx].upper()}")
            print(f"   Mi sposto qui e procedo allo step successivo!")
            current_c_idx, current_j_idx, current_comp_idx, current_d_idx = best_c_idx, best_j_idx, best_comp_idx, best_d_idx
            step += 1
        else:
            if evaluations_done < MAX_EVALUATIONS:
                print(f"\n🛑 NESSUN VICINO È MIGLIORE. L'algoritmo ha scoperto un Ottimo Locale (Cima della collina). Terminazione raggiunta.")
            break 

    # --- AGGIORNAMENTO DEL PERCORSO SUL CSV PER IL GRAFICO ---
    df = pd.read_csv(csv_filename)
    for path_step, p_c, p_j, p_comp, p_dist in final_path:
        mask = (df['cache_GB'] == p_c) & (df['journal_ms'] == p_j) & (df['compressor'] == p_comp) & (df['distribution'] == p_dist)
        df.loc[mask, 'step'] = path_step 
        df.loc[mask, 'is_path'] = True
    df.to_csv(csv_filename, index=False)

    # Calcoliamo il tempo totale impiegato per completare l'algoritmo di Hill Climbing, per avere un'idea del tempo necessario per questo tipo di esplorazione locale.
    total_minutes = (time.time() - start_time_global) / 60.0

    print(f"\n✅ Hill Climbing 4D completato in {total_minutes:.1f} minuti.")
    
    print(f"\n📊 Generazione Grafico Matrice 3x3 in corso...")
    plot_steepest_hc_master(csv_filename, os.path.join(BASE_DIR, f"Analisi_{ts}"))
    print(f"✅ Fatto! Trovi i risultati in: {BASE_DIR}")

if __name__ == "__main__":
    main()