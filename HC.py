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
from scipy.interpolate import griddata, Rbf
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
# VALUTAZIONE CON MEMORIA E CLEAN CSV
# ==========================================
# Questa funzione e il cuore dell'algoritmo:
# valuta un punto specifico (configurazione) e utilizza la memoria per evitare valutazioni ridondanti.
def evaluate_point(c_idx, j_idx, comp_idx, d_idx, visited_points, visited_eval_ids, csv_filename, step_num, start_time_global, is_center=False):
    global evaluations_done
    
    c_gb = CACHE_SIZES[c_idx]
    j_ms = JOURNAL_INTERVALS[j_idx]
    comp = COMPRESSORS[comp_idx]
    dist = DISTRIBUTIONS[d_idx]
    key  = (c_idx, j_idx, comp_idx, d_idx)
    
    # --- MEMORIA (Memoization) ---
    if key in visited_points:
        stats = visited_points[key]
        avg_thr, min_thr, max_thr, std_thr, avg_dur, min_dur, max_dur, std_dur = stats
        print(f"      -> Gia esplorato in passato! Recupero dalla memoria: {avg_thr:.2f} ops/sec")
        elapsed_minutes = (time.time() - start_time_global) / 60.0
        
        # Aggiorniamo il CSV anche per questo punto gia visitato, in modo da avere un record completo di tutte le valutazioni, anche quelle memorizzate.
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([evaluations_done + 1, step_num, c_gb, j_ms, comp, dist, "mean", 
                           avg_dur, round(min_dur, 3), round(max_dur, 3), round(std_dur, 3),
                           avg_thr, round(min_thr, 2), round(max_thr, 2), round(std_thr, 2), 
                           False, round(elapsed_minutes, 2)])
        evaluations_done += 1
        # Restituiamo anche l'eval_id ORIGINALE del punto (non quello da memoization)
        return avg_thr, visited_eval_ids[key]
    
    evaluations_done += 1
    current_eval_id = evaluations_done
    visited_eval_ids[key] = current_eval_id   # registra l'eval_id originale
    
    tipo_punto = "POSIZIONE ATTUALE" if is_center else "VALUTAZIONE VICINO"
    print(f"\n   [{tipo_punto} - Val. {current_eval_id}/{EVALUATIONS}] Testo: C={c_gb}GB, J={j_ms}ms, Comp={comp}, Dist={dist.upper()} ...", end="", flush=True)
    
    # ====================================================================
    # LA MAGIA DELL'API: ESECUZIONE NATIVA REALE DA config.py
    # ====================================================================
    avg_thr, min_thr, max_thr, std_thr, avg_dur, min_dur, max_dur, std_dur = execute_full_test(c_gb, j_ms, comp, dist)
    
    print(f"   THROUGHPUT FINALE MEDIO: {avg_thr:.2f} ops/sec\n")
    
    elapsed_minutes = (time.time() - start_time_global) / 60.0
    
    # Aggiorniamo il CSV con i risultati di questa valutazione, indicando se e un punto del percorso (is_path) o meno, e quanto tempo e passato dall'inizio dell'algoritmo.
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([current_eval_id, step_num, c_gb, j_ms, comp, dist, "mean",
                       avg_dur, round(min_dur, 3), round(max_dur, 3), round(std_dur, 3),
                       avg_thr, round(min_thr, 2), round(max_thr, 2), round(std_thr, 2),
                       False, round(elapsed_minutes, 2)])

    visited_points[key] = (avg_thr, min_thr, max_thr, std_thr, avg_dur, min_dur, max_dur, std_dur)
    
    return avg_thr, current_eval_id

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
def plot_steepest_hc_master(csv_file, output_prefix, global_vmin=None, global_vmax=None):
    df = pd.read_csv(csv_file)
    if 'throughput' in df.columns and 'throughput_avg' not in df.columns:
        df = df.rename(columns={'throughput': 'throughput_avg'})
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(len(COMPRESSORS), len(DISTRIBUTIONS), figsize=(20, 15))

    vmin = global_vmin if global_vmin is not None else df['throughput_avg'].min()
    vmax = global_vmax if global_vmax is not None else df['throughput_avg'].max()
    contour_plot = None

    # Identifica lo step del percorso con throughput assoluto massimo (per la stellina)
    df_path_all = df[df['is_path'] == True]
    if not df_path_all.empty:
        best_path_row = df_path_all.loc[df_path_all['throughput_avg'].idxmax()]
        absolute_best_step = int(best_path_row['step'])
    else:
        absolute_best_step = -1

    for r, comp in enumerate(COMPRESSORS):
        for c, dist in enumerate(DISTRIBUTIONS):
            ax = axes[r, c]
            df_dist = df[df['distribution'] == dist]
            df_plane = df_dist[df_dist['compressor'] == comp]
            
            if not df_plane.empty and len(df_plane.groupby(['cache_GB', 'journal_ms'])) >= 2:
                df_grouped = df_plane.groupby(['cache_GB', 'journal_ms'])['throughput_avg'].mean(numeric_only=True).reset_index()
                x_coords = df_grouped['cache_GB'].map(cache_map).values
                y_coords = df_grouped['journal_ms'].map(journal_map).values
                z_vals = df_grouped['throughput_avg'].values

                grid_x, grid_y = np.mgrid[-0.3:len(CACHE_SIZES)-0.7:100j, -0.3:len(JOURNAL_INTERVALS)-0.7:100j]
                try:
                    import scipy.ndimage
                    # Interpolazione "topografica" liscia su TUTTO il piano:
                    # RBF multiquadric estende organicamente fuori dall'inviluppo
                    # convesso dei punti, producendo contorni ondeggianti invece
                    # dei bordi rettangolari di griddata+nearest. Fallback su
                    # griddata se RBF fallisce (es. punti collineari).
                    try:
                        # smooth=0 -> RBF passa ESATTAMENTE per i punti misurati.
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

                    contour_plot = ax.contourf(grid_x, grid_y, grid_z, levels=np.linspace(vmin, vmax, 30), cmap='RdYlGn', alpha=0.5, vmin=vmin, vmax=vmax, extend='both')
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

            # Filtra solo i punti di questo pannello
            panel_pts = path_points[
                (path_points['compressor'] == comp) & (path_points['distribution'] == dist)
            ].copy()

            # Raggruppa per posizione (px, py) per gestire sovrapposizioni
            from collections import defaultdict
            pos_to_rows = defaultdict(list)
            for _, row in panel_pts.iterrows():
                px = cache_map[row['cache_GB']]
                py = journal_map[row['journal_ms']]
                pos_to_rows[(px, py)].append(row)

            # Disegna nodi (con offset orizzontale se sovrapposti)
            for (px, py), rows_at_pos in pos_to_rows.items():
                n = len(rows_at_pos)
                for k, row in enumerate(rows_at_pos):
                    step_num = int(row['step'])
                    is_best  = (step_num == absolute_best_step)
                    lx = px + (k - (n - 1) / 2) * 0.22 if n > 1 else px
                    ly = py

                    if is_best:
                        is_best_restart = bool(row.get('is_restart', False)) if 'is_restart' in df.columns else False
                        star_edge = 'magenta' if is_best_restart else 'black'
                        ax.scatter(lx, ly, color='gold', marker='*', s=1200, zorder=12,
                                    edgecolors=star_edge, linewidth=1.5)
                        ax.annotate(f"S{step_num}", xy=(lx, ly),
                                    ha='center', va='center',
                                    fontsize=7, fontweight='bold', color='black',
                                    xycoords='data', zorder=14, annotation_clip=False)
                    else:
                        is_this_restart = bool(row.get('is_restart', False)) if 'is_restart' in df.columns else False
                        node_color = 'magenta' if is_this_restart else 'white'
                        ax.scatter(lx, ly, color=node_color, s=150, zorder=8,
                                   edgecolors='black', linewidth=1.5)
                        t = ax.text(lx, ly, f"S{step_num}", ha='center', va='center',
                                    fontsize=10, fontweight='bold', color='black', zorder=10,
                                    bbox=dict(boxstyle="round,pad=0.2", fc=node_color, ec="black", lw=1, alpha=0.9))
                        texts.append(t)

            # Frecce tra step consecutivi (usa posizioni reali, non offset)
            panel_steps = sorted(panel_pts['step'].tolist())
            for idx in range(len(panel_steps) - 1):
                s_cur  = panel_steps[idx]
                s_next = panel_steps[idx + 1]
                r_cur  = panel_pts[panel_pts['step'] == s_cur].iloc[0]
                r_next = panel_pts[panel_pts['step'] == s_next].iloc[0]
                px, py = cache_map[r_cur['cache_GB']],  journal_map[r_cur['journal_ms']]
                nx, ny = cache_map[r_next['cache_GB']], journal_map[r_next['journal_ms']]
                is_restart_jump = bool(r_next.get('is_restart', False)) if 'is_restart' in df.columns else False
                if (px != nx or py != ny) and not is_restart_jump:
                    ax.annotate("", xy=(nx, ny), xytext=(px, py),
                                arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.4",
                                                color="darkblue", lw=2.5, shrinkA=12, shrinkB=12), zorder=6)

            if texts:
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.7))

            ax.set_xticks(range(len(CACHE_SIZES))); ax.set_xticklabels(CACHE_SIZES, fontsize=18)
            ax.set_yticks(range(len(JOURNAL_INTERVALS))); ax.set_yticklabels(JOURNAL_INTERVALS, fontsize=18)
            if r == len(COMPRESSORS) - 1: ax.set_xlabel("Cache Size (GB)", fontsize=19, fontweight='bold')
            if c == 0:
                ax.set_ylabel("Journal Interval (ms)", fontsize=19, fontweight='bold')
            if c == len(DISTRIBUTIONS) - 1:
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('left')
                ax.set_ylabel(f"Compressione: {comp.upper()}", fontsize=19, fontweight='bold', rotation=-90, labelpad=22)
            if r == 0: ax.set_title(f"Distribuzione: {dist.upper()}", fontsize=19, fontweight='bold')
            ax.set_xlim(-0.3, len(CACHE_SIZES)-0.7); ax.set_ylim(-0.3, len(JOURNAL_INTERVALS)-0.7)
            ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(top=0.97, right=0.92, hspace=0.3) 
    if contour_plot:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7]) 
        cbar = fig.colorbar(contour_plot, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Throughput (ops/sec)', fontsize=22)
        cbar.ax.tick_params(labelsize=17)
    
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
    
    # DEBUG: Verifica che le variabili d'ambiente siano impostate correttamente
    master_dir = get_master_dir()
    env_var = os.environ.get("BENCHMARK_MASTER_DIR")
    print(f"[DEBUG] BENCHMARK_MASTER_DIR (env var): {env_var}")
    print(f"[DEBUG] get_master_dir() ritorna: {master_dir}")
    
    # Calcola BASE_DIR DENTRO main() per usare il valore CORRETTO di BENCHMARK_MASTER_DIR
    BASE_DIR = os.path.join(get_master_dir(), "Hill Climbing")
    print(f"[DEBUG] BASE_DIR sara: {BASE_DIR}")
    
    # Creazione del CSV con intestazione
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"results_HC_WL_{WORKLOAD_TYPE}_{ts}.csv")

    # Scriviamo l'intestazione del CSV prima di iniziare le valutazioni,
    # in modo da avere un file pronto per essere popolato con i risultati di ogni punto valutato, incluso quelli recuperati dalla memoria.
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation_id", "step", "cache_GB", "journal_ms", "compressor", "distribution", "repetition",
                        "duration_avg", "duration_min", "duration_max", "duration_std",
                        "throughput_avg", "throughput_min", "throughput_max", "throughput_std",
                        "is_path", "elapsed_minutes"])

    print(f"PARTENZA ALGORITMO: HILL CLIMBING (Workload {WORKLOAD_TYPE})")

    start_time_global = time.time()

    # Per tenere traccia dei punti gia visitati e dei loro risultati, utilizziamo un dizionario "visited_points"
    # che mappa ogni configurazione (c_idx, j_idx, comp_idx, d_idx) alla sua throughput media e durata media.
    visited_points = {}
    visited_eval_ids = {}   # (c_idx,j_idx,comp_idx,d_idx) -> evaluation_id originale
    final_path = []
    
    # Partiamo dal punto centrale della matrice, che e una scelta ragionevole per iniziare la scalata,
    # in quanto ci permette di esplorare in tutte le direzioni.
    current_c_idx = len(CACHE_SIZES) // 2       
    current_j_idx = len(JOURNAL_INTERVALS) // 2 
    current_comp_idx = len(COMPRESSORS) // 2    
    current_d_idx = len(DISTRIBUTIONS) // 2     
    
    step = 1
    is_restart_flag = False
    
    # Valuto il punto centrale e lo registro come primo punto del percorso
    print(f"\nVALUTAZIONE PUNTO DI PARTENZA")
    best_thr, s1_eval_id = evaluate_point(current_c_idx, current_j_idx, current_comp_idx, current_d_idx, visited_points, visited_eval_ids, csv_filename, step, start_time_global, is_center=True)
    final_path.append((step, CACHE_SIZES[current_c_idx], JOURNAL_INTERVALS[current_j_idx], COMPRESSORS[current_comp_idx], DISTRIBUTIONS[current_d_idx], is_restart_flag, s1_eval_id))
    
    # Il ciclo principale continua finche non raggiungiamo il limite massimo di valutazioni (budget).
    while evaluations_done < EVALUATIONS:
        print(f"\n" + "="*50)
        print(f"INIZIO STEP {step} (Valutazioni Totali: {evaluations_done}/{EVALUATIONS})")
        print("="*50)
        
        # Inizialmente, il miglior vicino in assoluto e il punto centrale stesso, che e il nostro punto di partenza.
        best_c_idx, best_j_idx, best_comp_idx, best_d_idx = current_c_idx, current_j_idx, current_comp_idx, current_d_idx
        
        # Reset del flag restart dopo averlo registrato
        is_restart_flag = False
        
        # 2. Generiamo e TESTIAMO TUTTI i vicini ortogonali
        neighbors = get_all_orthogonal_neighbors(current_c_idx, current_j_idx, current_comp_idx, current_d_idx)
        best_next_c_idx, best_next_j_idx, best_next_comp_idx, best_next_d_idx = current_c_idx, current_j_idx, current_comp_idx, current_d_idx
        
        # Valutiamo tutti i vicini e scegliamo sempre il migliore tra quelli esaminati.
        for n_c, n_j, n_comp, n_d in neighbors:

            # Prima di valutare questo vicino, controlliamo se abbiamo gia raggiunto il limite massimo di valutazioni (budget).
            if evaluations_done >= EVALUATIONS:
                print(f"\nATTENZIONE: Raggiunto il limite massimo di valutazioni (Budget = {EVALUATIONS}). Interruzione forzata.")
                break
                
            # Valutiamo questo vicino, che restituira la sua throughput media. La funzione "evaluate_point" si occupera di gestire la memoria e il CSV.
            thr, _ = evaluate_point(n_c, n_j, n_comp, n_d, visited_points, visited_eval_ids, csv_filename, step, start_time_global, is_center=False)

            # Se questo vicino ha una throughput migliore del miglior vicino trovato finora, aggiorniamo il candidato successivo.
            if thr > best_thr:
                best_thr = thr
                best_next_c_idx, best_next_j_idx, best_next_comp_idx, best_next_d_idx = n_c, n_j, n_comp, n_d
        
        # 3. Decisione: facciamo sempre il passo sul migliore tra i vicini valutati.
        if evaluations_done < EVALUATIONS:
            if best_next_c_idx == current_c_idx and best_next_j_idx == current_j_idx and best_next_comp_idx == current_comp_idx and best_next_d_idx == current_d_idx:
                print("\n Nessun miglioramento. Ottimo locale raggiunto!")
                print(f" RANDOM RESTART! Sfrutto il budget rimanente ({EVALUATIONS - evaluations_done} valutazioni)...")
                current_c_idx = random.randint(0, len(CACHE_SIZES) - 1)
                current_j_idx = random.randint(0, len(JOURNAL_INTERVALS) - 1)
                current_comp_idx = random.randint(0, len(COMPRESSORS) - 1)
                current_d_idx = random.randint(0, len(DISTRIBUTIONS) - 1)
                
                # Cerca una configurazione INEDITA per non sprecare il budget
                tentativi = 0
                while (current_c_idx, current_j_idx, current_comp_idx, current_d_idx) in visited_points and tentativi < 1000:
                    current_c_idx = random.randint(0, len(CACHE_SIZES) - 1)
                    current_j_idx = random.randint(0, len(JOURNAL_INTERVALS) - 1)
                    current_comp_idx = random.randint(0, len(COMPRESSORS) - 1)
                    current_d_idx = random.randint(0, len(DISTRIBUTIONS) - 1)
                    tentativi += 1
                step += 1
                is_restart_flag = True
                
                # Valutiamo e salviamo il nuovo punto di restart
                best_thr, restart_eval_id = evaluate_point(current_c_idx, current_j_idx, current_comp_idx, current_d_idx, visited_points, visited_eval_ids, csv_filename, step, start_time_global, is_center=True)
                final_path.append((step, CACHE_SIZES[current_c_idx], JOURNAL_INTERVALS[current_j_idx], COMPRESSORS[current_comp_idx], DISTRIBUTIONS[current_d_idx], is_restart_flag, restart_eval_id))
                is_restart_flag = False
            else:
                print(f"\n Il vicino migliore in assoluto è¨: Cache={CACHE_SIZES[best_next_c_idx]}GB | Journal={JOURNAL_INTERVALS[best_next_j_idx]}ms | Comp={COMPRESSORS[best_next_comp_idx]} | Dist={DISTRIBUTIONS[best_next_d_idx].upper()}")
                print(f"   Mi sposto qui e procedo allo step successivo!")
                current_c_idx, current_j_idx, current_comp_idx, current_d_idx = best_next_c_idx, best_next_j_idx, best_next_comp_idx, best_next_d_idx
                step += 1
                move_eval_id = visited_eval_ids[(current_c_idx, current_j_idx, current_comp_idx, current_d_idx)]
                # Salviamo il punto scelto nel path
                final_path.append((step, CACHE_SIZES[current_c_idx], JOURNAL_INTERVALS[current_j_idx], COMPRESSORS[current_comp_idx], DISTRIBUTIONS[current_d_idx], is_restart_flag, move_eval_id))

    # --- AGGIORNAMENTO DEL PERCORSO SUL CSV PER IL GRAFICO ---
    df = pd.read_csv(csv_filename)
    if 'is_restart' not in df.columns:
        df['is_restart'] = False
        
    for path_step, p_c, p_j, p_comp, p_dist, is_restart, p_eval_id in final_path:
        mask = df['evaluation_id'] == p_eval_id      # chiave univoca
        df.loc[mask, 'step']    = path_step
        df.loc[mask, 'is_path'] = True
        if is_restart:
            df.loc[mask, 'is_restart'] = True
            
    df.to_csv(csv_filename, index=False)

    # Calcoliamo il tempo totale impiegato per completare l'algoritmo di Hill Climbing, per avere un'idea del tempo necessario per questo tipo di esplorazione locale.
    total_minutes = (time.time() - start_time_global) / 60.0

    print(f"\nHill Climbing 4D completato in {total_minutes:.1f} minuti.")
    
    print(f"\nGenerazione Grafico Matrice 3x3 in corso...")
    # (Generazione Heatmap spostata al termine del workload da master_plotter)

    # plot_steepest_hc_master(csv_filename, os.path.join(BASE_DIR, f"Analisi_{ts}"))
    print(f"Fatto! Trovi i risultati in: {BASE_DIR}")

if __name__ == "__main__":
    main()
