import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PALETTE COLORI CUSTOM
# Sostituisce tab10 con colori più distinti e leggibili anche in proiezione
# e stampa in bianco/nero. Assegnazione fissa per algoritmo.
# Grid Search rimane nera tratteggiata come baseline.
# ==========================================
ALGO_COLORS = {
    "Bayesian Optimization":  "#0077BB",  # Blu forte
    "Coordinate Search":      "#EE7733",  # Arancione
    "Evolutionary Algorithm": "#009988",  # Verde acqua
    "Hill Climbing":          "#CC3311",  # Rosso forte
    "Random Search":          "#AA3377",  # Magenta/viola
    "Simulated Annealing":    "#33BBEE",  # Azzurro/ciano
}
# Fallback nel caso arrivi un algoritmo non mappato
FALLBACK_COLORS = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#AA3377", "#33BBEE"]

# ==========================================
# RICONOSCIMENTO RIGOROSO ALGORITMI
# ==========================================
def get_algorithm_name(filename):
    fn = filename.upper()
    if "GS" in fn or "GRID_SEARCH" in fn: return "Grid Search"
    if "RS" in fn or "RANDOM_SEARCH" in fn: return "Random Search"
    if "EA" in fn or "EVOLUTIONARY" in fn: return "Evolutionary Algorithm"
    if "SA" in fn or "SIMULATED" in fn: return "Simulated Annealing"
    if "HC" in fn or "HILL" in fn: return "Hill Climbing"
    if "BO" in fn or "BAYES" in fn: return "Bayesian Optimization"
    if "CS" in fn or "COORDINATE" in fn: return "Coordinate Search"
    return "Unknown Algorithm"


def apply_clean_style(ax):
    """Applica lo stile pulito e moderno a un asse:
    - Griglia sottile con alpha=0.3 per non sovrastare le curve
    - Rimozione spine superiore e destra (look accademico moderno)
    """
    ax.grid(True, linestyle='-', alpha=0.5, linewidth=0.7, color='#bbbbbb')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def build_subtitle(algo_data, gs_max_throughput):
    """Costruisce il sottotitolo dinamico con la migliore configurazione trovata.
    Cerca l'algoritmo con il throughput finale più alto e ne estrae
    compressore, cache e journal interval dalla colonna best_config se disponibile.
    """
    if not algo_data:
        return ""

    # Trova l'algoritmo con il throughput finale più alto
    best_algo = max(algo_data.items(), key=lambda x: x[1]['y'][-1])
    best_name = best_algo[0]
    best_thr = best_algo[1]['y'][-1]
    best_cfg = best_algo[1].get('best_config', {})

    parts = [f"Miglior algoritmo: {best_name}"]

    # Aggiunge dettagli configurazione se disponibili nel CSV
    if best_cfg.get('compressor'):
        parts.append(f"Compressore: {str(best_cfg['compressor']).upper()}")
    if best_cfg.get('cache_gb'):
        parts.append(f"Cache: {best_cfg['cache_gb']} GB")
    if best_cfg.get('journal_ms'):
        parts.append(f"Journal: {best_cfg['journal_ms']} ms")

    parts.append(f"{best_thr:.0f} ops/sec")

    return "  |  ".join(parts)


def main():
    # Prende le direttive dal Master Runner
    master_dir = os.environ.get("BENCHMARK_MASTER_DIR")
    workload = os.environ.get("WORKLOAD_TYPE")

    # DEBUG: Verifica variabili d'ambiente
    print(f"\n[PLOTTER DEBUG] BENCHMARK_MASTER_DIR (env var): {master_dir}")
    print(f"[PLOTTER DEBUG] WORKLOAD_TYPE (env var): {workload}")

    if not master_dir or not workload:
        print("[PLOTTER ERROR] Variabili d'ambiente mancanti. Uso cartella locale per test.")
        master_dir = os.getcwd()
        # Aggiornato il fallback per supportare anche il Workload D
        if "Workload_A" in master_dir:
            workload = "A"
        elif "Workload_B" in master_dir:
            workload = "B"
        elif "Workload_D" in master_dir:
            workload = "D"
        else:
            workload = "Sconosciuto"

    print(f"[MASTER PLOTTER] Generazione dei 2 report (Steps & Time) per Workload {workload}...")

    # Cerca i CSV in tutte le sottocartelle (BO, HC, ecc.)
    all_csvs = glob.glob(os.path.join(master_dir, "**", "*.csv"), recursive=True)
    print(f"[PLOTTER DEBUG] Cercando CSV in: {master_dir}")
    print(f"[PLOTTER DEBUG] CSV trovati: {len(all_csvs)}")
    if all_csvs:
        print(f"[PLOTTER DEBUG] Primi 3 CSV: {all_csvs[:3]}")

    # Setup stile accademico
    plt.style.use('seaborn-v0_8-white')

    # Usiamo GridSpec per riservare una riga stretta in CIMA alla figura
    # esclusivamente per i riquadri informativi: in questo modo non si
    # sovrappongono MAI alle curve, indipendentemente dal contenuto del grafico.
    # height_ratios=[1, 8] → la riga header è ~11% dell'altezza totale.
    from matplotlib.gridspec import GridSpec

    def make_header_ax(fig, gs, slot):
        """Crea un asse header pulito: niente spine, tick, sfondo.
        Serve come tela per i riquadri informativi sopra il grafico.
        NON chiamare set_visible(False): nasconde anche i testi!"""
        ax = fig.add_subplot(gs[slot])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_visible(False)
        return ax

    fig_steps, ax_steps = plt.subplots(figsize=(12, 8.5))
    fig_time,  ax_time  = plt.subplots(figsize=(12, 8.5))

    algo_data = {}
    gs_max_throughput = None
    gs_best_min = None
    gs_best_max = None
    gs_best_abs_min = None  # Throughput MINIMO assoluto misurato per la miglior config
    gs_best_abs_max = None  # Throughput MASSIMO assoluto misurato per la miglior config
    gs_best_config = None  # Configurazione ottima della Grid Search (riquadro sinistra)
    fallback_idx = 0

    def pick_column(columns, preferred_names):
        for name in preferred_names:
            if name in columns:
                return name
        return None

    for csv_file in all_csvs:
        algo_name = get_algorithm_name(os.path.basename(csv_file))
        if algo_name == "Unknown Algorithm": continue

        try:
            # Carica il CSV (gestisce virgola o punto e virgola)
            df = pd.read_csv(csv_file, sep=None, engine='python')
            df.columns = df.columns.str.lower().str.strip()

            throughput_col = pick_column(df.columns, ['throughput_avg', 'throughput'])
            throughput_std_col = pick_column(df.columns, ['throughput_std'])
            throughput_min_col = pick_column(df.columns, ['throughput_min'])
            throughput_max_col = pick_column(df.columns, ['throughput_max'])

            if throughput_col is None:
                continue

            # Trova la colonna degli Step e quella del Tempo
            step_col = next((c for c in ['evaluation_id', 'step', 'generation', 'iteration'] if c in df.columns), None)
            time_col = next((c for c in ['elapsed_minutes', 'timestamp', 'elapsed_time', 'time', 'seconds'] if c in df.columns), None)

            if not step_col: continue
            df = df.sort_values(by=step_col)

            # Se è Grid Search, salviamo il massimo e la sua configurazione ottima
            if algo_name == "Grid Search":
                current_max = float(df[throughput_col].max())
                if gs_max_throughput is None or current_max > gs_max_throughput:
                    gs_max_throughput = current_max
                    gs_best_row = df.loc[df[throughput_col].idxmax()]
                    gs_best_config = {
                        'compressor': gs_best_row.get('compressor', None),
                        'cache_gb':   gs_best_row.get('cache_gb', None),
                        'journal_ms': gs_best_row.get('journal_ms', None),
                    }
                    # Fascia = media ± deviazione standard delle ripetizioni
                    # (convenzione accademica: cattura l'incertezza tipica della
                    # misura, meno sensibile agli outlier rispetto a min/max).
                    if throughput_std_col:
                        gs_best_avg = float(gs_best_row[throughput_col])
                        gs_best_std = float(gs_best_row[throughput_std_col])
                        gs_best_min = gs_best_avg - gs_best_std
                        gs_best_max = gs_best_avg + gs_best_std
                    # Min/max ASSOLUTI misurati sulle ripetizioni della miglior config
                    # (richiesti dal relatore: mostrano l'inviluppo reale della misura,
                    # utili per capire se un euristico supera davvero la Grid Search).
                    if throughput_min_col:
                        gs_best_abs_min = float(gs_best_row[throughput_min_col])
                    if throughput_max_col:
                        gs_best_abs_max = float(gs_best_row[throughput_max_col])
                continue

            # Calcolo della curva Anytime (massimo progressivo)
            df['cummax_throughput'] = df[throughput_col].cummax()

            # Estrazione migliore configurazione trovata dall'algoritmo
            # (usata per il sottotitolo dinamico del grafico)
            best_row = df.loc[df[throughput_col].idxmax()]
            best_config = {
                'compressor': best_row.get('compressor', None),
                'cache_gb':   best_row.get('cache_gb', None),
                'journal_ms': best_row.get('journal_ms', None),
            }

            # Calcolo tempo in minuti (parte da 0).
            # Per i cache hit (step rieseguiti da memoria in millisecondi), l'elapsed_minutes
            # è quasi uguale al passo precedente, comprimendo visivamente la curva.
            # Soluzione: per ogni step, usiamo la duration_avg dell'evaluation ORIGINALE
            # di quella configurazione (prima occorrenza nel CSV), poi ricostruiamo il
            # tempo come somma cumulativa delle durate effettive (in secondi → minuti).
            time_minutes = None
            if time_col:
                dur_col_name = next((c for c in df.columns if c == 'duration_avg'), None)
                cache_c   = next((c for c in df.columns if 'cache' in c and 'gb' in c.lower()), None)
                journal_c = next((c for c in df.columns if 'journal' in c), None)
                comp_c    = next((c for c in df.columns if 'compressor' in c), None)
                dist_c    = next((c for c in df.columns if 'distribut' in c), None)

                if dur_col_name and all([cache_c, journal_c, comp_c, dist_c]):
                    key_cols = [cache_c, journal_c, comp_c, dist_c]
                    first_dur = df.groupby(key_cols, sort=False)[dur_col_name].first()

                    def get_eff_dur(row):
                        try:
                            return first_dur[tuple(row[c] for c in key_cols)]
                        except Exception:
                            return row[dur_col_name]

                    eff_dur_sec = df.apply(get_eff_dur, axis=1)
                    time_minutes = eff_dur_sec.cumsum() / 60
                    time_minutes = time_minutes - time_minutes.iloc[0] + eff_dur_sec.iloc[0] / 60
                else:
                    if time_col == 'elapsed_minutes':
                        time_minutes = df[time_col] - df[time_col].min()
                    else:
                        time_minutes = (df[time_col] - df[time_col].min()) / 60

            algo_data[algo_name] = {
                'steps':       df[step_col].values,
                'time':        time_minutes,
                'y':           df['cummax_throughput'].values,
                'best_config': best_config,
            }
            print(f"Dati caricati per: {algo_name}")

        except Exception as e:
            print(f"ATTENZIONE: Errore nel file {csv_file}: {e}")

    # Disegno delle linee per ogni algoritmo
    for algo_name, data in sorted(algo_data.items()):
        # Colore fisso per algoritmo dalla palette custom, fallback se non mappato
        color = ALGO_COLORS.get(algo_name, FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)])
        if algo_name not in ALGO_COLORS:
            fallback_idx += 1

        # Sfumatura rimossa: era visivamente confusa senza aggiungere informazione.

        # 1. Grafico Performance vs Budget (Step)
        # Tutti gli step sono equidistanti sull'asse X, quindi tutti i marker sono visibili.
        ax_steps.plot(data['steps'], data['y'], label=algo_name,
                      linewidth=2.5, color=color, zorder=2)

        # 2. Grafico Performance vs Tempo (Minuti)
        # Il tempo è già corretto: i cache hit usano la duration_avg dell'evaluation
        # originale, quindi la curva è distribuita uniformemente. Solo linea pulita.
        if data['time'] is not None:
            ax_time.plot(data['time'], data['y'], linewidth=2.5,
                         color=color, label=algo_name, zorder=2)

    # Aggiunta della Baseline Grid Search (Linea Nera Tratteggiata e Ombreggiatura Min/Max)
    if gs_max_throughput is not None:
        if gs_best_min is not None and gs_best_max is not None:
            # Ombreggiatura a fascia (min-max) orizzontale
            ax_steps.axhspan(gs_best_min, gs_best_max, color='black', alpha=0.15, label='Grid Search Avg ± Std', zorder=1)
            ax_time.axhspan(gs_best_min, gs_best_max, color='black', alpha=0.15, label='Grid Search Avg ± Std', zorder=1)

        ax_steps.axhline(y=gs_max_throughput, color='black', linestyle='--',
                         linewidth=2.5, label='Grid Search', zorder=3)
        ax_time.axhline(y=gs_max_throughput, color='black', linestyle='--',
                        linewidth=2.5, label='Grid Search', zorder=3)

        # Righe MIN/MAX assoluti misurati per la miglior config (linee sottili
        # punteggiate). Mostrano l'inviluppo reale delle ripetizioni della Grid
        # Search: aiutano a giudicare se un euristico supera davvero la baseline
        # o resta dentro il rumore di misura. Si lascia tutto il resto invariato.
        if gs_best_abs_max is not None:
            ax_steps.axhline(y=gs_best_abs_max, color='black', linestyle=':',
                             linewidth=1.3, alpha=0.7, label='Grid Search Max (misurato)', zorder=3)
            ax_time.axhline(y=gs_best_abs_max, color='black', linestyle=':',
                            linewidth=1.3, alpha=0.7, label='Grid Search Max (misurato)', zorder=3)
        if gs_best_abs_min is not None:
            ax_steps.axhline(y=gs_best_abs_min, color='black', linestyle=':',
                             linewidth=1.3, alpha=0.7, label='Grid Search Min (misurato)', zorder=3)
            ax_time.axhline(y=gs_best_abs_min, color='black', linestyle=':',
                            linewidth=1.3, alpha=0.7, label='Grid Search Min (misurato)', zorder=3)

    # ── Riquadri informativi nella riga HEADER (al di sopra del grafico) ────
    # ax_header è un asse invisibile dedicato: occupa la riga superiore della
    # GridSpec e non tocca mai l'area delle curve. I riquadri vengono ancorati
    # in coordinate axes (0-1) dell'asse header, quindi sono sempre separati
    # fisicamente dal plot. Sinistra (bordo nero): Grid Search ottimo globale.
    # Destra (bordo colorato): miglior risultato degli euristici.
    def add_info_boxes(ax_header):
        # Riquadro sinistro — Grid Search (ottimo globale, bordo nero)
        if gs_max_throughput is not None:
            gs_cfg    = gs_best_config or {}
            gs_comp   = str(gs_cfg.get('compressor', 'N/A')).upper()
            gs_cache  = gs_cfg.get('cache_gb', 'N/A')
            gs_jrn    = gs_cfg.get('journal_ms', 'N/A')
            gs_text   = (f"\u25a1 GRID SEARCH (Ottimo Globale)\n"
                         f"Throughput: {gs_max_throughput:.0f} ops/sec\n"
                         f"Comp: {gs_comp} | Cache: {gs_cache}GB | Journal: {gs_jrn}ms")
            ax_header.text(0.01, 0.95, gs_text,
                           fontsize=9.5, verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                     alpha=0.95, edgecolor='black', linewidth=1.5),
                           transform=ax_header.transAxes, zorder=10)

        # Riquadro destro — Miglior algoritmo euristico (bordo nel colore dell'algoritmo)
        if algo_data:
            best_item  = max(algo_data.items(), key=lambda x: x[1]['y'][-1])
            best_name  = best_item[0]
            best_thr   = best_item[1]['y'][-1]
            best_cfg   = best_item[1].get('best_config', {})
            best_color = ALGO_COLORS.get(best_name, '#333333')
            b_comp     = str(best_cfg.get('compressor', 'N/A')).upper()
            b_cache    = best_cfg.get('cache_gb', 'N/A')
            b_jrn      = best_cfg.get('journal_ms', 'N/A')
            win_text   = (f"\u25a1 MIGLIOR ALGORITMO: {best_name}\n"
                          f"Throughput: {best_thr:.0f} ops/sec\n"
                          f"Comp: {b_comp} | Cache: {b_cache}GB | Journal: {b_jrn}ms")
            ax_header.text(0.99, 0.95, win_text,
                           fontsize=9.5, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                     alpha=0.95, edgecolor=best_color, linewidth=2.0),
                           transform=ax_header.transAxes, zorder=10)

    # Sottotitolo dinamico con la migliore configurazione trovata
    subtitle = build_subtitle(algo_data, gs_max_throughput)

    # ── Stile pulito e moderno (griglia sottile + spine rimosse) ────────────
    apply_clean_style(ax_steps)
    apply_clean_style(ax_time)

    from matplotlib.ticker import MultipleLocator
    ax_steps.yaxis.set_major_locator(MultipleLocator(100))
    ax_time.yaxis.set_major_locator(MultipleLocator(100))

    # --- Estetica Finale Grafico STEP ---
    ax_steps.set_xlabel("Numero di Configurazioni Testate", fontsize=13)
    ax_steps.set_ylabel("Throughput Medio Massimo Osservato (ops/sec)", fontsize=13)
    ax_steps.legend(loc='lower right', frameon=True, shadow=False,
                    framealpha=0.9, edgecolor='#cccccc')

    # ── Export multi-formato ─────────────────────────────────────────────────
    base_steps = os.path.join(master_dir, f"REPORT_STEPS_WL_{workload}")
    fig_steps.savefig(base_steps + ".png", dpi=300, bbox_inches='tight')
    fig_steps.savefig(base_steps + ".pdf",           bbox_inches='tight')
    fig_steps.savefig(base_steps + ".svg",           bbox_inches='tight')

    # --- Estetica Finale Grafico TIME ---
    ax_time.set_xlabel("Tempo Trascorso (Minuti)", fontsize=13)
    ax_time.set_ylabel("Throughput Medio Massimo Osservato (ops/sec)", fontsize=13)
    ax_time.legend(loc='lower right', frameon=True, shadow=False,
                   framealpha=0.9, edgecolor='#cccccc')

    # ── Export multi-formato ─────────────────────────────────────────────────
    base_time = os.path.join(master_dir, f"REPORT_TIME_WL_{workload}")
    fig_time.savefig(base_time + ".png", dpi=300, bbox_inches='tight')
    fig_time.savefig(base_time + ".pdf",           bbox_inches='tight')
    fig_time.savefig(base_time + ".svg",           bbox_inches='tight')

    plt.close('all')
    print(f"[PLOTTER] Grafici salvati in 3 formati (PNG/PDF/SVG) in: {master_dir}")

    # ==========================================
    # RIGENERAZIONE HEATMAP CON SCALA CROMATICA GLOBALE
    # ==========================================
    print(f"\n[MASTER PLOTTER] Generazione Heatmap algoritmi con scala cromatica globale per Workload {workload}...")
    
    # 1. Calcolo del min e max globale per il throughput
    global_vmin = float('inf')
    global_vmax = float('-inf')
    
    for csv_file in all_csvs:
        try:
            df_tmp = pd.read_csv(csv_file, sep=None, engine='python')
            df_tmp.columns = df_tmp.columns.str.lower().str.strip()
            t_col = pick_column(df_tmp.columns, ['throughput_avg', 'throughput'])
            if t_col:
                global_vmin = min(global_vmin, float(df_tmp[t_col].min()))
                global_vmax = max(global_vmax, float(df_tmp[t_col].max()))
        except Exception:
            pass

    if global_vmin == float('inf') or global_vmax == float('-inf'):
        print("ATTENZIONE: Nessun dato valido trovato per calcolare min/max globale. Uso scala locale per algoritmi.")
        global_vmin = None
        global_vmax = None
    else:
        print(f"   [SCALA] VMIN Globale={global_vmin:.2f}, VMAX Globale={global_vmax:.2f}")

    # 2. Richiamo le funzioni di plot per ogni file
    import sys
    import importlib.util

    def run_plot_func(py_file, func_name, csv_path, out_prefix):
        if not os.path.exists(py_file): return
        try:
            spec = importlib.util.spec_from_file_location("module.name", py_file)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["module.name"] = mod
            spec.loader.exec_module(mod)
            plot_func = getattr(mod, func_name, None)
            if plot_func:
                plot_func(csv_path, out_prefix, global_vmin=global_vmin, global_vmax=global_vmax)
        except Exception as e:
            print(f"ATTENZIONE: Errore nel generare la heatmap da {py_file} per il file {csv_path}: {e}")

    import datetime
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    for csv_file in all_csvs:
        base_dir_csv = os.path.dirname(csv_file)
        algo_name = get_algorithm_name(os.path.basename(csv_file))
        if algo_name == "Unknown Algorithm": continue
        
        # Mappatura algoritmi per avviare lo script giusto
        if algo_name == "Grid Search":
            run_plot_func(os.path.join(os.getcwd(), 'GS.py'), 'plot_gs_master', csv_file, os.path.join(base_dir_csv, f"HEATMAP_WL_{workload}_{ts}"))
        elif algo_name == "Random Search":
            run_plot_func(os.path.join(os.getcwd(), 'RS.py'), 'plot_rs_master', csv_file, os.path.join(base_dir_csv, f"HEATMAP_WL_{workload}_{ts}"))
        elif algo_name == "Evolutionary Algorithm":
            run_plot_func(os.path.join(os.getcwd(), 'EA.py'), 'plot_ea_master', csv_file, os.path.join(base_dir_csv, f"HEATMAP_WL_{workload}_{ts}"))
        elif algo_name == "Simulated Annealing":
            run_plot_func(os.path.join(os.getcwd(), 'SA.py'), 'plot_sa_master', csv_file, os.path.join(base_dir_csv, f"HEATMAP_WL_{workload}_{ts}"))
        elif algo_name == "Hill Climbing":
            run_plot_func(os.path.join(os.getcwd(), 'HC.py'), 'plot_steepest_hc_master', csv_file, os.path.join(base_dir_csv, f"Analisi_{ts}"))
        elif algo_name == "Bayesian Optimization":
            run_plot_func(os.path.join(os.getcwd(), 'BO.py'), 'plot_bo_master', csv_file, os.path.join(base_dir_csv, f"HEATMAP_WL_{workload}_{ts}"))
        elif algo_name == "Coordinate Search":
            run_plot_func(os.path.join(os.getcwd(), 'CS.py'), 'plot_cs_master', csv_file, os.path.join(base_dir_csv, f"Analisi_WL_{workload}_{ts}"))

if __name__ == "__main__":
    main()
