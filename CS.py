# ==========================================
# IMPORTAZIONI DI BASE E SISTEMA
# ==========================================
import os
import time
import datetime
import csv
import random
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# LIBRERIE PER GRAFICI E MATEMATICA
# ==========================================
import numpy as np
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
# CONTATORE DI VALUTAZIONI PER LA STOP CONDITION
# ==========================================
evaluations_done = 0 

# ==========================================
# VALUTAZIONE CON MEMORIA E DELEGAZIONE ALLA MASTER API
# ==========================================
# Questa funzione Ã¨ il cuore dell'esplorazione: valuta un punto specifico, ma prima controlla se Ã¨ giÃ  stato valutato (memoization).
# Se Ã¨ giÃ  stato valutato, recupera il risultato dalla memoria e lo registra nel CSV senza dover rieseguire il test.
# Altrimenti, esegue il test completo delegando al configuratore centrale, registra il risultato e lo memorizza.
def evaluate_point(c_idx, j_idx, comp_idx, d_idx, visited_points, visited_eval_ids, csv_filename, step_num, start_time_global, is_center=False):
    global evaluations_done
    
    c_gb = CACHE_SIZES[c_idx]
    j_ms = JOURNAL_INTERVALS[j_idx]
    comp = COMPRESSORS[comp_idx]
    dist = DISTRIBUTIONS[d_idx]
    key  = (c_idx, j_idx, comp_idx, d_idx)
    
    # --- MEMOIZATION ---
    if key in visited_points:
        stats = visited_points[key]
        avg_thr, min_thr, max_thr, std_thr, avg_dur, min_dur, max_dur, std_dur = stats
        print(f"      -> â­ï¸ GiÃ  esplorato! Recupero memoria: {avg_thr:.2f} ops/sec")
        elapsed_minutes = (time.time() - start_time_global) / 60.0
        
        evaluations_done += 1
        
        # Registrazione nel CSV anche per i punti di memoria, con is_path=False
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([evaluations_done, step_num, c_gb, j_ms, comp, dist, "mean",
                           avg_dur, round(min_dur, 3), round(max_dur, 3), round(std_dur, 3),
                           avg_thr, round(min_thr, 2), round(max_thr, 2), round(std_thr, 2),
                           False, round(elapsed_minutes, 2)])
        # Restituiamo anche l'eval_id ORIGINALE del punto (non quello da memoization)
        return avg_thr, visited_eval_ids[key]
    
    evaluations_done += 1
    current_eval_id = evaluations_done
    visited_eval_ids[key] = current_eval_id   # â† registra l'eval_id originale
    
    # Indico se Ã¨ il punto di partenza (PUNTO BASE) o un punto di esplorazione lungo un asse (ESPLORAZIONE ASSE).
    tipo_punto = "PUNTO BASE" if is_center else "ESPLORAZIONE ASSE"
    print(f"\n   [{tipo_punto} - Val. {current_eval_id}/{EVALUATIONS}] Testo: C={c_gb}GB, J={j_ms}ms, Comp={comp}, Dist={dist.upper()} ...", end="", flush=True)
    
    # Il configuratore si occuperÃ  di eseguire i vari run, calcolare la media e restituire i risultati.
    avg_thr, min_thr, max_thr, std_thr, avg_dur, min_dur, max_dur, std_dur = execute_full_test(c_gb, j_ms, comp, dist)
    
    print(f"   ðŸš€ THROUGHPUT FINALE MEDIO: {avg_thr:.2f} ops/sec\n")
    elapsed_minutes = (time.time() - start_time_global) / 60.0
    
    # Registrazione nel CSV
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([current_eval_id, step_num, c_gb, j_ms, comp, dist, "mean",
                       avg_dur, round(min_dur, 3), round(max_dur, 3), round(std_dur, 3),
                       avg_thr, round(min_thr, 2), round(max_thr, 2), round(std_thr, 2),
                       False, round(elapsed_minutes, 2)])

    # Memorizzo il risultato per evitare future valutazioni dello stesso punto (memoization)
    visited_points[key] = (avg_thr, min_thr, max_thr, std_thr, avg_dur, min_dur, max_dur, std_dur)
    return avg_thr, current_eval_id

# ==========================================
# LOGICA DI SPOSTAMENTO (UN SINGOLO ASSE ALLA VOLTA)
# ==========================================
# Questa funzione restituisce solo i vicini lungo l'asse specificato, tenendo fissi gli altri 3.
# In questo modo, quando esploro un asse, mi concentro solo su quel cambiamento specifico.
def get_axis_neighbors(c_idx, j_idx, comp_idx, d_idx, axis):
    """Restituisce TUTTI i punti lungo l'asse specificato (escluso il punto corrente),
    tenendo fissi gli altri 3 parametri. Questo Ã¨ il comportamento corretto della
    Coordinate Search: esplora l'intero asse prima di decidere dove spostarsi."""
    
    neighbors = []
    if axis == 0:   # Asse Cache
        for i in range(len(CACHE_SIZES)):
            if i != c_idx:
                neighbors.append((i, j_idx, comp_idx, d_idx))
    elif axis == 1: # Asse Journal
        for i in range(len(JOURNAL_INTERVALS)):
            if i != j_idx:
                neighbors.append((c_idx, i, comp_idx, d_idx))
    elif axis == 2: # Asse Compressore
        for i in range(len(COMPRESSORS)):
            if i != comp_idx:
                neighbors.append((c_idx, j_idx, i, d_idx))
    elif axis == 3: # Asse Distribuzione
        for i in range(len(DISTRIBUTIONS)):
            if i != d_idx:
                neighbors.append((c_idx, j_idx, comp_idx, i))
    return neighbors

# ==========================================
# GENERAZIONE GRAFICI MATRICE (3x3)
# ==========================================
def plot_cs_master(csv_file, output_prefix, global_vmin=None, global_vmax=None):
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

                    contour_plot = ax.contourf(grid_x, grid_y, grid_z, levels=np.linspace(vmin, vmax, 30), cmap='RdYlGn', alpha=0.5, vmin=vmin, vmax=vmax, extend='both')
                except Exception:
                    pass

            texts = []
            
            df_discarded = df_plane[df_plane['is_path'] == False]
            if not df_discarded.empty:
                df_discarded_grouped = df_discarded.groupby(['cache_GB', 'journal_ms']).mean(numeric_only=True).reset_index()
                for _, row in df_discarded_grouped.iterrows():
                    ex = cache_map[row['cache_GB']]
                    ey = journal_map[row['journal_ms']]
                    ax.scatter(ex, ey, color='gray', s=50, alpha=0.6, zorder=4)

            df_path = df[df['is_path'] == True].sort_values('step')
            path_points = df_path.groupby('step').first().reset_index()
            
            # Filtra solo i punti di questo pannello
            panel_pts = path_points[
                (path_points['compressor'] == comp) & (path_points['distribution'] == dist)
            ].copy()

            # â”€â”€ Raggruppa per posizione (px, py) per gestire sovrapposizioni â”€â”€
            from collections import defaultdict
            pos_to_rows = defaultdict(list)
            for _, row in panel_pts.iterrows():
                px = cache_map[row['cache_GB']]
                py = journal_map[row['journal_ms']]
                pos_to_rows[(px, py)].append(row)

            # â”€â”€ Disegna nodi (con offset orizzontale se sovrapposti) â”€â”€
            label_positions = {}   # step_num -> (lx, ly) per le frecce
            for (px, py), rows_at_pos in pos_to_rows.items():
                n = len(rows_at_pos)
                for k, row in enumerate(rows_at_pos):
                    step_num = int(row['step'])
                    is_best = (step_num == absolute_best_step)

                    # Offset orizzontale: distribuisce i punti sovrapposti
                    lx = px + (k - (n - 1) / 2) * 0.22 if n > 1 else px
                    ly = py
                    label_positions[step_num] = (lx, ly)

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

            # â”€â”€ Frecce tra step consecutivi (usa posizioni reali px/py, non offset) â”€â”€
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
    
    plt.savefig(f"{output_prefix}_Griglia_CS.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_CS.pdf",           bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_CS.svg",           bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN LOOP (COORDINATE SEARCH)
# ==========================================
def main():
    global evaluations_done
    evaluations_done = 0
    
    # âš ï¸ DEBUG: Verifica che le variabili d'ambiente siano impostate correttamente
    master_dir = get_master_dir()
    env_var = os.environ.get("BENCHMARK_MASTER_DIR")
    print(f"[DEBUG] BENCHMARK_MASTER_DIR (env var): {env_var}")
    print(f"[DEBUG] get_master_dir() ritorna: {master_dir}")
    
    # Calcola BASE_DIR DENTRO main() per usare il valore CORRETTO di BENCHMARK_MASTER_DIR
    BASE_DIR = os.path.join(get_master_dir(), "Coordinate Search")
    print(f"[DEBUG] BASE_DIR sarÃ : {BASE_DIR}")
    
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"results_Basic_CS_WL_{WORKLOAD_TYPE}_{ts}.csv")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation_id", "step", "cache_GB", "journal_ms", "compressor", "distribution", "repetition",
                        "duration_avg", "duration_min", "duration_max", "duration_std",
                        "throughput_avg", "throughput_min", "throughput_max", "throughput_std",
                        "is_path", "elapsed_minutes"])

    print(f"ðŸš€ PARTENZA ALGORITMO: COORDINATE SEARCH (Workload {WORKLOAD_TYPE})")

    start_time_global = time.time()

    # Per tenere traccia dei punti giÃ  valutati e dei risultati, utilizzo una struttura di memoization.
    visited_points = {}
    visited_eval_ids = {}   # (c_idx,j_idx,comp_idx,d_idx) â†’ evaluation_id originale
    final_path = []
    asse_nomi = ["Cache", "Journal", "Compressore", "Distribuzione"]
    
    # Partiamo dal centro
    curr_c = len(CACHE_SIZES) // 2       
    curr_j = len(JOURNAL_INTERVALS) // 2 
    curr_comp = len(COMPRESSORS) // 2    
    curr_d = len(DISTRIBUTIONS) // 2     
    
    step = 1
    is_restart_flag = False
    print(f"\nðŸŽ¯ VALUTAZIONE PUNTO DI PARTENZA")
    # Valuto il punto centrale e lo registro come primo punto del percorso (is_path=True)
    best_thr, s1_eval_id = evaluate_point(curr_c, curr_j, curr_comp, curr_d, visited_points, visited_eval_ids, csv_filename, step, start_time_global, is_center=True)
    # Registro il punto centrale come primo punto del percorso (is_path=True)
    final_path.append((step, CACHE_SIZES[curr_c], JOURNAL_INTERVALS[curr_j], COMPRESSORS[curr_comp], DISTRIBUTIONS[curr_d], is_restart_flag, s1_eval_id))
    
    # Il ciclo esterno continua finchÃ© non raggiungiamo il limite massimo di valutazioni (budget).
    while evaluations_done < EVALUATIONS:
        print(f"\n" + "="*50)
        print(f"ðŸ”„ INIZIO NUOVO CICLO SUGLI ASSI (Valutazioni: {evaluations_done}/{EVALUATIONS})")
        print("="*50)
         
        improvement_in_cycle = False

        for axis in range(4): # Cicla attraverso gli assi: 0=Cache, 1=Journal, 2=Comp, 3=Dist
            if evaluations_done >= EVALUATIONS:
                break
                
            print(f"\n   ðŸ” Esploro asse: {asse_nomi[axis]} (tenendo fissi gli altri)...")
            # Ottengo solo i vicini lungo QUESTO asse, tenendo fissi gli altri 3 parametri
            neighbors = get_axis_neighbors(curr_c, curr_j, curr_comp, curr_d, axis)
            
            # Per tenere traccia del miglior punto trovato su QUESTO asse specifico, inizializzo variabili dedicate.
            axis_best_thr = best_thr
            axis_best_coords = (curr_c, curr_j, curr_comp, curr_d)
            found_better_on_axis = False
            
            # Esploro i vicini lungo QUESTO asse specifico, valutando ognuno e confrontando con il miglior risultato trovato finora.
            for n_c, n_j, n_comp, n_d in neighbors:
                if evaluations_done >= EVALUATIONS:
                    print(f"âš ï¸ Raggiunto il limite massimo di valutazioni (Budget = {EVALUATIONS}). Interruzione forzata.")
                    break
                    
                # Valuto il punto vicino e ottengo la sua throughput media
                thr, _ = evaluate_point(n_c, n_j, n_comp, n_d, visited_points, visited_eval_ids, csv_filename, step, start_time_global, is_center=False)
                
                # Se trovo un punto migliore lungo questo specifico asse, lo segno
                if thr > axis_best_thr:
                    axis_best_thr = thr
                    axis_best_coords = (n_c, n_j, n_comp, n_d)
                    found_better_on_axis = True
            
            # --- MOVIMENTO ---
            # Se ho trovato di meglio su QUESTO asse, MI MUOVO. Altrimenti rimango dov'Ã¨.
            if found_better_on_axis:
                if evaluations_done < EVALUATIONS:
                    curr_c, curr_j, curr_comp, curr_d = axis_best_coords
                    best_thr = axis_best_thr
                    step += 1
                    # Registro il punto migliore trovato su QUESTO asse come parte del percorso (is_path=True)
                    move_eval_id = visited_eval_ids[axis_best_coords]
                    final_path.append((step, CACHE_SIZES[curr_c], JOURNAL_INTERVALS[curr_j], COMPRESSORS[curr_comp], DISTRIBUTIONS[curr_d], is_restart_flag, move_eval_id))
                    is_restart_flag = False
                    
                    print(f"   âœ… Miglioramento sull'asse {asse_nomi[axis]}! Mi sposto subito a: Cache={CACHE_SIZES[curr_c]}GB | Journal={JOURNAL_INTERVALS[curr_j]}ms | Comp={COMPRESSORS[curr_comp]} | Dist={DISTRIBUTIONS[curr_d].upper()}")
                    improvement_in_cycle = True

        # Se in un intero ciclo di tutti gli assi non abbiamo trovato alcun miglioramento, siamo in un ottimo locale. 
        if not improvement_in_cycle:
            print("\nðŸš« Nessun miglioramento. Ottimo locale raggiunto!")
            if evaluations_done < EVALUATIONS:
                print(f"ðŸ”€ RANDOM RESTART! Sfrutto il budget rimanente ({EVALUATIONS - evaluations_done} valutazioni)...")
                curr_c = random.randint(0, len(CACHE_SIZES) - 1)
                curr_j = random.randint(0, len(JOURNAL_INTERVALS) - 1)
                curr_comp = random.randint(0, len(COMPRESSORS) - 1)
                curr_d = random.randint(0, len(DISTRIBUTIONS) - 1)
                
                # Cerca una configurazione INEDITA per non sprecare il budget
                tentativi = 0
                while (curr_c, curr_j, curr_comp, curr_d) in visited_points and tentativi < 1000:
                    curr_c = random.randint(0, len(CACHE_SIZES) - 1)
                    curr_j = random.randint(0, len(JOURNAL_INTERVALS) - 1)
                    curr_comp = random.randint(0, len(COMPRESSORS) - 1)
                    curr_d = random.randint(0, len(DISTRIBUTIONS) - 1)
                    tentativi += 1
                
                step += 1
                is_restart_flag = True
                best_thr, restart_eval_id = evaluate_point(curr_c, curr_j, curr_comp, curr_d, visited_points, visited_eval_ids, csv_filename, step, start_time_global, is_center=True)
                final_path.append((step, CACHE_SIZES[curr_c], JOURNAL_INTERVALS[curr_j], COMPRESSORS[curr_comp], DISTRIBUTIONS[curr_d], is_restart_flag, restart_eval_id))
                is_restart_flag = False
            else:
                break

    # --- AGGIORNAMENTO DEL PATH FINALE NEL CSV ---
    # Usa evaluation_id come chiave univoca: impedisce che step successivi
    # con le stesse coordinate sovrascrivano step precedenti (es. S9 su S1).
    df = pd.read_csv(csv_filename)
    if 'is_restart' not in df.columns:
        df['is_restart'] = False
        
    for path_step, p_c, p_j, p_comp, p_dist, is_restart, p_eval_id in final_path:
        mask = df['evaluation_id'] == p_eval_id      # â† chiave univoca
        df.loc[mask, 'step']    = path_step
        df.loc[mask, 'is_path'] = True
        if is_restart:
            df.loc[mask, 'is_restart'] = True
            
    df.to_csv(csv_filename, index=False)

    # Calcoliamo il tempo totale impiegato per completare la Coordinate Search, per avere un'idea del tempo necessario per questo tipo di esplorazione sistematica.
    total_minutes = (time.time() - start_time_global) / 60.0

    print(f"\nâœ… Coordinate Search completata in {total_minutes:.1f} minuti.")
    print(f"ðŸ† OTTIMO LOCALE TROVATO: Cache={CACHE_SIZES[curr_c]}GB | Journal={JOURNAL_INTERVALS[curr_j]}ms | Comp={COMPRESSORS[curr_comp]} | Dist={DISTRIBUTIONS[curr_d].upper()}")
    
    print(f"\nðŸ“Š Generazione Grafico Matrice 3x3 in corso...")
    # (Generazione Heatmap spostata al termine del workload da master_plotter)

    # plot_cs_master(csv_filename, os.path.join(BASE_DIR, f"Analisi_WL_{WORKLOAD_TYPE}_{ts}"))
    print(f"âœ… Fatto! Trovi i risultati in: {BASE_DIR}")

if __name__ == "__main__":
    main()
