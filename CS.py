# ==========================================
# IMPORTAZIONI DI BASE E SISTEMA
# ==========================================
import os
import time
import datetime
import csv
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# LIBRERIE PER GRAFICI E MATEMATICA
# ==========================================
import numpy as np
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
BASE_DIR = os.path.join(master_dir, "Coordinate Search")

# ==========================================
# CONTATORE DI VALUTAZIONI PER LA STOP CONDITION
# ==========================================
evaluations_done = 0 

# ==========================================
# VALUTAZIONE CON MEMORIA E DELEGAZIONE ALLA MASTER API
# ==========================================
# Questa funzione è il cuore dell'esplorazione: valuta un punto specifico, ma prima controlla se è già stato valutato (memoization).
# Se è già stato valutato, recupera il risultato dalla memoria e lo registra nel CSV senza dover rieseguire il test.
# Altrimenti, esegue il test completo delegando al configuratore centrale, registra il risultato e lo memorizza.
def evaluate_point(c_idx, j_idx, comp_idx, d_idx, visited_points, csv_filename, step_num, start_time_global, is_center=False):
    global evaluations_done
    
    c_gb = CACHE_SIZES[c_idx]
    j_ms = JOURNAL_INTERVALS[j_idx]
    comp = COMPRESSORS[comp_idx]
    dist = DISTRIBUTIONS[d_idx]
    
    # --- MEMOIZATION ---
    if (c_idx, j_idx, comp_idx, d_idx) in visited_points:
        avg_thr, avg_dur = visited_points[(c_idx, j_idx, comp_idx, d_idx)]
        print(f"      -> ⏭️ Già esplorato! Recupero memoria: {avg_thr:.2f} ops/sec")
        elapsed_minutes = (time.time() - start_time_global) / 60.0
        
        # Registrazione nel CSV anche per i punti di memoria, con is_path=False (non è un punto del percorso, ma è comunque esplorato)
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([evaluations_done, step_num, c_gb, j_ms, comp, dist, "mean", avg_dur, avg_thr, False, round(elapsed_minutes, 2)])
        return avg_thr
    
    evaluations_done += 1
    
    # Indico se è il punto di partenza (PUNTO BASE) o un punto di esplorazione lungo un asse (ESPLORAZIONE ASSE).
    tipo_punto = "PUNTO BASE" if is_center else "ESPLORAZIONE ASSE"
    print(f"\n   [{tipo_punto} - Val. {evaluations_done}/{MAX_EVALUATIONS}] Testo: C={c_gb}GB, J={j_ms}ms, Comp={comp}, Dist={dist.upper()} ...", end="", flush=True)
    
    # Il configuratore si occuperà di eseguire i vari run, calcolare la media e restituire i risultati.
    avg_thr, avg_dur = execute_full_test(c_gb, j_ms, comp, dist)
    
    print(f"   🚀 THROUGHPUT FINALE MEDIO: {avg_thr:.2f} ops/sec\n")
    elapsed_minutes = (time.time() - start_time_global) / 60.0
    
    # Registrazione nel CSV (anche per i punti di memoria, con is_path=False)
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([evaluations_done, step_num, c_gb, j_ms, comp, dist, "mean", avg_dur, avg_thr, False, round(elapsed_minutes, 2)])

    # Memorizzo il risultato per evitare future valutazioni dello stesso punto (memoization)
    visited_points[(c_idx, j_idx, comp_idx, d_idx)] = (avg_thr, avg_dur)
    return avg_thr

# ==========================================
# LOGICA DI SPOSTAMENTO (UN SINGOLO ASSE ALLA VOLTA)
# ==========================================
# Questa funzione restituisce solo i vicini lungo l'asse specificato, tenendo fissi gli altri 3.
# In questo modo, quando esploro un asse, mi concentro solo su quel cambiamento specifico.
def get_axis_neighbors(c_idx, j_idx, comp_idx, d_idx, axis):
    """Restituisce solo i vicini muovendosi lungo l'asse specificato, tenendo fissi gli altri 3."""

    # Per ogni asse, posso muovermi in due direzioni (se non sono ai bordi). Restituisco solo quei vicini.
    neighbors = []
    if axis == 0:   # Asse Cache
        if c_idx > 0: neighbors.append((c_idx - 1, j_idx, comp_idx, d_idx))
        if c_idx < len(CACHE_SIZES) - 1: neighbors.append((c_idx + 1, j_idx, comp_idx, d_idx))
    elif axis == 1: # Asse Journal
        if j_idx > 0: neighbors.append((c_idx, j_idx - 1, comp_idx, d_idx))
        if j_idx < len(JOURNAL_INTERVALS) - 1: neighbors.append((c_idx, j_idx + 1, comp_idx, d_idx))
    elif axis == 2: # Asse Compressore
        if comp_idx > 0: neighbors.append((c_idx, j_idx, comp_idx - 1, d_idx))
        if comp_idx < len(COMPRESSORS) - 1: neighbors.append((c_idx, j_idx, comp_idx + 1, d_idx))
    elif axis == 3: # Asse Distribuzione
        if d_idx > 0: neighbors.append((c_idx, j_idx, comp_idx, d_idx - 1))
        if d_idx < len(DISTRIBUTIONS) - 1: neighbors.append((c_idx, j_idx, comp_idx, d_idx + 1))
    return neighbors

# ==========================================
# GENERAZIONE GRAFICI MATRICE (3x3)
# ==========================================
def plot_cs_master(csv_file, output_prefix):
    df = pd.read_csv(csv_file)
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(len(COMPRESSORS), len(DISTRIBUTIONS), figsize=(20, 15))
    fig.suptitle(f"Coordinate Search - Mappa Topografica (Workload {WORKLOAD_TYPE})\n(Punti grigi = Esplorati e scartati | Punti Bianchi = Percorso Vetta)", fontsize=20, fontweight='bold')

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
            
            df_discarded = df_plane[df_plane['is_path'] == False]
            if not df_discarded.empty:
                df_discarded_grouped = df_discarded.groupby(['cache_GB', 'journal_ms']).mean(numeric_only=True).reset_index()
                for _, row in df_discarded_grouped.iterrows():
                    ex = cache_map[row['cache_GB']]
                    ey = journal_map[row['journal_ms']]
                    ax.scatter(ex, ey, color='gray', s=50, alpha=0.6, zorder=4)

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
    
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"results_Basic_CS_WL_{WORKLOAD_TYPE}_{ts}.csv")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation_id", "step", "cache_GB", "journal_ms", "compressor", "distribution", "repetition", "duration", "throughput", "is_path", "elapsed_minutes"])

    print(f"🚀 PARTENZA ALGORITMO: COORDINATE SEARCH (Workload {WORKLOAD_TYPE})")

    start_time_global = time.time()

    # Per tenere traccia dei punti già valutati e dei risultati, utilizzo una struttura di memoization.
    visited_points = {}
    final_path = []
    asse_nomi = ["Cache", "Journal", "Compressore", "Distribuzione"]
    
    # Partiamo dal centro
    curr_c = len(CACHE_SIZES) // 2       
    curr_j = len(JOURNAL_INTERVALS) // 2 
    curr_comp = len(COMPRESSORS) // 2    
    curr_d = len(DISTRIBUTIONS) // 2     
    
    step = 1
    print(f"\n🎯 VALUTAZIONE PUNTO DI PARTENZA")
    # Valuto il punto centrale e lo registro come primo punto del percorso (is_path=True)
    best_thr = evaluate_point(curr_c, curr_j, curr_comp, curr_d, visited_points, csv_filename, step, start_time_global, is_center=True)
    # Registro il punto centrale come primo punto del percorso (is_path=True)
    final_path.append((step, CACHE_SIZES[curr_c], JOURNAL_INTERVALS[curr_j], COMPRESSORS[curr_comp], DISTRIBUTIONS[curr_d]))
    
    # Setto una variabile per tenere traccia se c'è stato un miglioramento in questo ciclo completo sugli assi
    improvement_in_cycle = True
    
    # Il ciclo esterno continua finché c'è un miglioramento in almeno uno degli assi.
    # Se in un ciclo completo sugli assi non c'è miglioramento, significa che abbiamo raggiunto un ottimo locale.
    while improvement_in_cycle and evaluations_done < MAX_EVALUATIONS:
        improvement_in_cycle = False
        print(f"\n" + "="*50)
        print(f"🔄 INIZIO NUOVO CICLO SUGLI ASSI (Valutazioni: {evaluations_done}/{MAX_EVALUATIONS})")
        print("="*50)
         
        for axis in range(4): # Cicla attraverso gli assi: 0=Cache, 1=Journal, 2=Comp, 3=Dist
            if evaluations_done >= MAX_EVALUATIONS:
                break
                
            print(f"\n   🔍 Esploro asse: {asse_nomi[axis]} (tenendo fissi gli altri)...")
            # Ottengo solo i vicini lungo QUESTO asse, tenendo fissi gli altri 3 parametri
            neighbors = get_axis_neighbors(curr_c, curr_j, curr_comp, curr_d, axis)
            
            # Per tenere traccia del miglior punto trovato su QUESTO asse specifico, inizializzo variabili dedicate.
            axis_best_thr = best_thr
            axis_best_coords = (curr_c, curr_j, curr_comp, curr_d)
            found_better_on_axis = False
            
            # Esploro i vicini lungo QUESTO asse specifico, valutando ognuno e confrontando con il miglior risultato trovato finora.
            for n_c, n_j, n_comp, n_d in neighbors:
                if evaluations_done >= MAX_EVALUATIONS:
                    print("⚠️ Limite di valutazioni raggiunto.")
                    break
                    
                # Valuto il punto vicino e ottengo la sua throughput media
                thr = evaluate_point(n_c, n_j, n_comp, n_d, visited_points, csv_filename, step, start_time_global, is_center=False)
                
                # Se trovo un punto migliore lungo questo specifico asse, lo segno
                if thr > axis_best_thr:
                    axis_best_thr = thr
                    axis_best_coords = (n_c, n_j, n_comp, n_d)
                    found_better_on_axis = True
            
            # --- MOVIMENTO IMMEDIATO ---
            # Appena ho finito di esplorare QUESTO asse, se ho trovato di meglio, MI MUOVO.
            if found_better_on_axis:
                curr_c, curr_j, curr_comp, curr_d = axis_best_coords
                best_thr = axis_best_thr
                improvement_in_cycle = True # Segnalo che il ciclo ha prodotto un miglioramento
                step += 1
                # Registro il punto migliore trovato su QUESTO asse come parte del percorso (is_path=True)
                final_path.append((step, CACHE_SIZES[curr_c], JOURNAL_INTERVALS[curr_j], COMPRESSORS[curr_comp], DISTRIBUTIONS[curr_d]))
                
                print(f"   ✅ Miglioramento sull'asse {asse_nomi[axis]}! Mi sposto subito a: Cache={CACHE_SIZES[curr_c]}GB | Journal={JOURNAL_INTERVALS[curr_j]}ms | Comp={COMPRESSORS[curr_comp]} | Dist={DISTRIBUTIONS[curr_d].upper()}")

    # --- AGGIORNAMENTO DEL PATH FINALE NEL CSV ---
    df = pd.read_csv(csv_filename)
    for path_step, p_c, p_j, p_comp, p_dist in final_path:
        mask = (df['cache_GB'] == p_c) & (df['journal_ms'] == p_j) & (df['compressor'] == p_comp) & (df['distribution'] == p_dist)
        df.loc[mask, 'step'] = path_step 
        df.loc[mask, 'is_path'] = True
    df.to_csv(csv_filename, index=False)

    if not improvement_in_cycle:
        print(f"\n🛑 NESSUN MIGLIORAMENTO IN NESSUN ASSE. L'algoritmo ha scoperto un Ottimo Locale!")

    # Calcoliamo il tempo totale impiegato per completare la Coordinate Search, per avere un'idea del tempo necessario per questo tipo di esplorazione sistematica.
    total_minutes = (time.time() - start_time_global) / 60.0

    print(f"\n✅ Coordinate Search completata in {total_minutes:.1f} minuti.")
    print(f"🏆 OTTIMO LOCALE TROVATO: Cache={CACHE_SIZES[curr_c]}GB | Journal={JOURNAL_INTERVALS[curr_j]}ms | Comp={COMPRESSORS[curr_comp]} | Dist={DISTRIBUTIONS[curr_d].upper()}")
    
    print(f"\n📊 Generazione Grafico Matrice 3x3 in corso...")
    plot_cs_master(csv_filename, os.path.join(BASE_DIR, f"Analisi_WL_{WORKLOAD_TYPE}_{ts}"))
    print(f"✅ Fatto! Trovi i risultati in: {BASE_DIR}")

if __name__ == "__main__":
    main()