# ==========================================
# IMPORTAZIONI DI BASE E SISTEMA
# ==========================================
import os
import time
import random
import csv
import datetime  
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# ==========================================
# LIBRERIE PER L'OTTIMIZZAZIONE BAYESIANA
# ==========================================
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

# ==========================================
# LIBRERIE PER GRAFICI 3x3 (HEATMAP)
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

# PARAMETRI BAYESIAN OPTIMIZATION
INIT_POINTS = 10      # Punti casuali esplorativi iniziali
OPT_STEPS = EVALUATIONS - INIT_POINTS       # Passi guidati dall'Intelligenza Artificiale
TOTAL_STEPS = EVALUATIONS

# ==========================================
# VALUTAZIONE CON MEMORIA E CLEAN CSV
# ==========================================
def evaluate_point(c_idx, j_idx, comp_idx, d_idx, visited_points, csv_filename, step_num, phase, start_time_global):
    c_gb = CACHE_SIZES[c_idx]
    j_ms = JOURNAL_INTERVALS[j_idx]
    comp = COMPRESSORS[comp_idx]
    dist = DISTRIBUTIONS[d_idx]
    
    # --- 1. MEMOIZATION (Evita di rifare test già fatti) ---
    # Prima di eseguire un test, controlliamo se abbiamo già valutato questa configurazione.
    # Se sì, recuperiamo il risultato dalla memoria e salviamo comunque nel CSV per tracciamento completo.
    if (c_idx, j_idx, comp_idx, d_idx) in visited_points:
        stats = visited_points[(c_idx, j_idx, comp_idx, d_idx)]
        avg_thr, min_thr, max_thr, std_thr, avg_dur, min_dur, max_dur, std_dur = stats
        print(f"      -> ⏭️ Punto già valutato! Recupero dalla memoria: {avg_thr:.2f} ops/sec")
        elapsed_minutes = (time.time() - start_time_global) / 60.0
        
        # Salviamo comunque il risultato nel CSV per avere un tracciamento completo, anche dei punti già visitati.
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step_num, phase, c_gb, j_ms, comp, dist, "mean",
                           avg_dur, round(min_dur, 3), round(max_dur, 3), round(std_dur, 3),
                           avg_thr, round(min_thr, 2), round(max_thr, 2), round(std_thr, 2),
                           round(elapsed_minutes, 2)])
        return avg_thr
    
    print(f"\n   [Step {step_num} | {phase}] Valuto : C={c_gb}GB, J={j_ms}ms, Comp={comp}, Dist={dist.upper()} ...", end="", flush=True)
    
    # ====================================================================
    # 🚀 LA MAGIA DELL'API: Deleghiamo TUTTO a config.py in una sola riga!
    # ====================================================================
    avg_thr, min_thr, max_thr, std_thr, avg_dur, min_dur, max_dur, std_dur = execute_full_test(c_gb, j_ms, comp, dist)
    
    print(f"   🚀 THROUGHPUT FINALE MEDIO: {avg_thr:.2f} ops/sec\n")
    
    # --- 2. SALVATAGGIO DEI RISULTATI ---
    elapsed_minutes = (time.time() - start_time_global) / 60.0
    
    # Salviamo il risultato nel CSV, includendo anche la durata e il tempo totale trascorso dall'inizio della ricerca.
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([step_num, phase, c_gb, j_ms, comp, dist, "mean",
                       avg_dur, round(min_dur, 3), round(max_dur, 3), round(std_dur, 3),
                       avg_thr, round(min_thr, 2), round(max_thr, 2), round(std_thr, 2),
                       round(elapsed_minutes, 2)])
        
    # Salviamo il risultato nella memoria, in modo da evitare di rifare lo stesso test in futuro.
    visited_points[(c_idx, j_idx, comp_idx, d_idx)] = (avg_thr, min_thr, max_thr, std_thr, avg_dur, min_dur, max_dur, std_dur)
    
    return avg_thr

# ==========================================
# GENERAZIONE GRAFICI MATRICE (3x3)
# ==========================================
def plot_bo_master(csv_file, output_prefix, global_vmin=None, global_vmax=None):
    df = pd.read_csv(csv_file)
    if 'throughput' in df.columns and 'throughput_avg' not in df.columns:
        df = df.rename(columns={'throughput': 'throughput_avg'})
    cache_map = {val: idx for idx, val in enumerate(CACHE_SIZES)}
    journal_map = {val: idx for idx, val in enumerate(JOURNAL_INTERVALS)}

    fig, axes = plt.subplots(len(COMPRESSORS), len(DISTRIBUTIONS), figsize=(20, 15))
    fig.suptitle("Bayesian Optimization - Mappa Topografica Globale\n(L'algoritmo ha navigato liberamente imparando dallo spazio)", fontsize=20, fontweight='bold')

    vmin = global_vmin if global_vmin is not None else df['throughput_avg'].min()
    vmax = global_vmax if global_vmax is not None else df['throughput_avg'].max()
    contour_plot = None

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
                        # smooth=0 → RBF passa ESATTAMENTE per i punti misurati.
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
            
            # --- CORREZIONE LOGICA GLOBALE ---
            df_global_path = df.sort_values('step').groupby(['step', 'cache_GB', 'journal_ms', 'compressor', 'distribution', 'phase']).mean(numeric_only=True).reset_index()
            path_steps = df_global_path['step'].tolist()
            path_c = df_global_path['cache_GB'].tolist()
            path_j = df_global_path['journal_ms'].tolist()
            path_comp = df_global_path['compressor'].tolist()
            path_dist = df_global_path['distribution'].tolist()
            path_phases = df_global_path['phase'].tolist()

            # Identifica lo step con il throughput assoluto massimo (per la stellina)
            best_step_row = df.loc[df['throughput_avg'].idxmax()]
            absolute_best_step = int(best_step_row['step'])

            for i in range(len(path_steps)):
                # Disegniamo il punto SOLO se appartiene al riquadro corrente
                if path_comp[i] == comp and path_dist[i] == dist:
                    px = cache_map[path_c[i]]
                    py = journal_map[path_j[i]]

                    is_best = (int(path_steps[i]) == absolute_best_step)

                    if is_best:
                        # Stella dorata — il testo "TN" viene disegnato con annotate
                        # ancorato alle coordinate del punto (xycoords='data') e NON
                        # aggiunto a `texts`, così adjust_text non lo sposta mai via
                        # dalla stella. fontsize piccolo (7) per stare dentro la stella.
                        ax.scatter(px, py, color='gold', marker='*', s=1200, zorder=12,
                                   edgecolors='black', linewidth=1.5)
                        ax.annotate(f"T{int(path_steps[i])}", xy=(px, py),
                                    ha='center', va='center',
                                    fontsize=7, fontweight='bold', color='black',
                                    xycoords='data', zorder=14,
                                    annotation_clip=False)
                        # NON aggiungiamo a texts → adjust_text non lo tocca
                    else:
                        pt_color = 'lightgray' if path_phases[i] == 'Init' else 'white'
                        ax.scatter(px, py, color=pt_color, s=150, zorder=8,
                                   edgecolors='black', linewidth=1.5)
                        t = ax.text(px, py, f"T{int(path_steps[i])}", ha='center', va='center',
                                    fontsize=10, fontweight='bold', color='black', zorder=10,
                                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black",
                                              lw=1, alpha=0.9))
                        texts.append(t)  # solo i punti normali vengono spostati da adjust_text
                    
                    # Disegniamo la freccia SOLO se anche il passo successivo (i+1) è caduto in questo STESSO riquadro
                    if i < len(path_steps) - 1 and path_comp[i+1] == comp and path_dist[i+1] == dist:
                        nx = cache_map[path_c[i+1]]
                        ny = journal_map[path_j[i+1]]
                        if px != nx or py != ny:
                            ax.annotate("", xy=(nx, ny), xytext=(px, py), 
                                        arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.4", 
                                                        color="blue", lw=2.0, shrinkA=12, shrinkB=12, 
                                                        connectionstyle="arc3,rad=0.2"), zorder=6)

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
    
    plt.savefig(f"{output_prefix}_Griglia_BO.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_BO.pdf",           bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Griglia_BO.svg",           bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN LOOP (BAYESIAN OPTIMIZATION)
# ==========================================
def get_normalized_coords(c_idx, j_idx, comp_idx, d_idx):
     return [
         c_idx / (len(CACHE_SIZES) - 1), 
         j_idx / (len(JOURNAL_INTERVALS) - 1),
         comp_idx / (len(COMPRESSORS) - 1) if len(COMPRESSORS) > 1 else 0,
         d_idx / (len(DISTRIBUTIONS) - 1) if len(DISTRIBUTIONS) > 1 else 0
    ]

def main():
    print(f"🧠 PARTENZA ALGORITMO: BAYESIAN OPTIMIZATION (Workload {WORKLOAD_TYPE})")

    # ⚠️ DEBUG: Verifica che le variabili d'ambiente siano impostate correttamente
    master_dir = get_master_dir()
    env_var = os.environ.get("BENCHMARK_MASTER_DIR")
    print(f"   [DEBUG] BENCHMARK_MASTER_DIR (env var): {env_var}")
    print(f"   [DEBUG] get_master_dir() ritorna: {master_dir}")
    
    # Calcola BASE_DIR DENTRO main() per usare il valore CORRETTO di BENCHMARK_MASTER_DIR
    BASE_DIR = os.path.join(get_master_dir(), "Bayesian Optimization")
    print(f"   [DEBUG] BASE_DIR sarà: {BASE_DIR}")

    # Creazione directory e file CSV per i risultati
    os.makedirs(BASE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = os.path.join(BASE_DIR, f"RESULTS_BO_WL_{WORKLOAD_TYPE}_{ts}.csv")

    # Creazione file CSV con intestazione
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "phase", "cache_GB", "journal_ms", "compressor", "distribution", "repetition",
                "duration_avg", "duration_min", "duration_max", "duration_std",
                "throughput_avg", "throughput_min", "throughput_max", "throughput_std",
                "elapsed_minutes"])
    
    start_time_global = time.time()
    
    # Kernel Matematico per la BO
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
    
    # Struttura dati per memorizzare i punti già visitati e i loro risultati, in modo da evitare di rifare test già fatti.
    visited_points = {}
    X_train = [] 
    y_train = [] 
    
    # Generiamo la lista completa di tutte le combinazioni di parametri, che rappresentano lo spazio di ricerca totale.
    # Questa lista ci servirà per tenere traccia dei punti ancora da visitare.
    unvisited = [(c, j, comp, d) for c in range(len(CACHE_SIZES)) 
                                 for j in range(len(JOURNAL_INTERVALS)) 
                                 for comp in range(len(COMPRESSORS)) 
                                 for d in range(len(DISTRIBUTIONS))]
    
    print(f"\n--- FASE 1: INIZIALIZZAZIONE ({INIT_POINTS} PUNTI CASUALI SU TUTTO LO SPAZIO) ---")
    # Selezioniamo casualmente un certo numero di punti dallo spazio totale per l'inizializzazione,
    # in modo da avere una base di dati su cui far apprendere l'IA.
    init_points = random.sample(unvisited, INIT_POINTS) 
    
    step = 1

    # Valutiamo i punti iniziali, salvando i risultati sia in memoria che nel CSV.
    # Questi punti serviranno come base di apprendimento per l'IA.
    for c_idx, j_idx, comp_idx, d_idx in init_points:
        thr = evaluate_point(c_idx, j_idx, comp_idx, d_idx, visited_points, csv_filename, step, "Init", start_time_global)
        X_train.append(get_normalized_coords(c_idx, j_idx, comp_idx, d_idx))
        y_train.append(thr)
        unvisited.remove((c_idx, j_idx, comp_idx, d_idx))
        step += 1
        
    print(f"\n--- FASE 2: OTTIMIZZAZIONE BAYESIANA ({OPT_STEPS} PUNTI GUIDATI DALL'IA) ---")
    # Ora che abbiamo i dati iniziali, l'IA può iniziare a guidare la ricerca verso le aree più promettenti dello spazio,
    # bilanciando esplorazione e sfruttamento.
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    
    # Per ogni passo di ottimizzazione, l'IA valuta tutti i punti non ancora visitati,
    # calcola l'UCB (Upper Confidence Bound, cioè il limite superiore della confidenza) e sceglie il punto con il valore più alto.
    for opt_step in range(OPT_STEPS):
        print(f"\n👉 Calcolo Processo Gaussiano - Step {step}...")
        gp.fit(X_train, y_train)
        
        best_ucb = -float('inf')
        best_point = None
        
        for c_idx, j_idx, comp_idx, d_idx in unvisited:
            x_test = np.array([get_normalized_coords(c_idx, j_idx, comp_idx, d_idx)])
            mu, sigma = gp.predict(x_test, return_std=True)
            
            kappa = 1.96 
            ucb = mu[0] + kappa * sigma[0]
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_point = (c_idx, j_idx, comp_idx, d_idx)
        
        c_idx, j_idx, comp_idx, d_idx = best_point
        print(f"   🤖 L'AI sceglie: Cache={CACHE_SIZES[c_idx]}GB | Journal={JOURNAL_INTERVALS[j_idx]}ms | Comp={COMPRESSORS[comp_idx]} | Dist={DISTRIBUTIONS[d_idx].upper()}")
        
        # Valutiamo il punto scelto dall'IA, salvando i risultati in memoria e nel CSV.
        thr = evaluate_point(c_idx, j_idx, comp_idx, d_idx, visited_points, csv_filename, step, "BO", start_time_global)
        
        # Aggiorniamo i dati di addestramento con il nuovo punto valutato,
        # in modo che l'IA possa apprendere da questo nuovo risultato per i passi successivi.
        X_train.append(get_normalized_coords(c_idx, j_idx, comp_idx, d_idx))
        y_train.append(thr)
        unvisited.remove((c_idx, j_idx, comp_idx, d_idx))
        step += 1
        
    # --- FINE: CALCOLO VINCITORE E GRAFICI ---
    best_idx = np.argmax(y_train)
    best_c_idx = int(round(X_train[best_idx][0] * (len(CACHE_SIZES) - 1)))
    best_j_idx = int(round(X_train[best_idx][1] * (len(JOURNAL_INTERVALS) - 1)))
    best_comp_idx = int(round(X_train[best_idx][2] * (len(COMPRESSORS) - 1)))
    best_d_idx = int(round(X_train[best_idx][3] * (len(DISTRIBUTIONS) - 1)))
    
    # Calcoliamo il tempo totale impiegato per completare la Bayesian Optimization, per avere un'idea del tempo necessario per questo tipo di esplorazione guidata dall'IA.
    total_minutes = (time.time() - start_time_global) / 60.0

    print(f"\n🏁 FINE RICERCA BAYESIANA. (Tempo totale: {total_minutes:.1f} minuti)")
    print(f"🏆 OTTIMO ASSOLUTO TROVATO: Cache={CACHE_SIZES[best_c_idx]}GB | Journal={JOURNAL_INTERVALS[best_j_idx]}ms | Comp={COMPRESSORS[best_comp_idx]} | Dist={DISTRIBUTIONS[best_d_idx].upper()}")

    print(f"\n📊 Generazione Grafico Matrice 3x3 in corso...")
    
    image_prefix = os.path.join(BASE_DIR, f"HEATMAP_WL_{WORKLOAD_TYPE}_{ts}")
    try:
        # (Generazione Heatmap spostata al termine del workload da master_plotter)

        # plot_bo_master(csv_filename, image_prefix)
        print(f"✅ Mappe salvate con successo in: {BASE_DIR}")
    except Exception as e:
        print(f"⚠️ Impossibile generare la heatmap: {e}")

if __name__ == "__main__":
    main()
