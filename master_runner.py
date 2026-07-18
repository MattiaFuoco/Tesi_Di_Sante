import os
import subprocess
import sys
import time
import datetime
from config import global_setup_database, full_teardown

# Nomi esatti dei tuoi script
ALGORITHMS = [
    ("Grid Search", "GS.py"),
    ("Random Search", "RS.py"),
    ("Evolutionary Algorithm", "EA.py"),
    ("Simulated Annealing", "SA.py"),
    ("Coordinate Search", "CS.py"),
    ("Hill Climbing", "HC.py"),
    ("Bayesian Optimization", "BO.py")
]

PLOTTER_SCRIPT = "master_plotter.py"

def run_automation():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*65)
    print("MONGODB TUNING - AUTOMAZIONE TOTALE 'ZERO-CLICK'")
    print("="*65)
    print(f"Inizio sessione: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Mettiti comodo, il sistema farà tutto da solo per i Workload A, B e D.\n")
    
    start_time_total = time.time()

    # [FASE 1] MEGA-LOAD GLOBALE: Prepariamo i volumi una volta per tutta la sessione
    global_setup_database()
    
    # [FASE 2] LOOP DEI WORKLOAD E ALGORITMI
    # Esegue prima tutto il blocco A, poi il blocco B, e infine il blocco D

    results_base = os.environ.get("RESULTS_DIR", os.path.join(os.getcwd(), "results"))
    if not os.path.exists(results_base):
        os.makedirs(results_base)

    for workload in ["A", "B", "D"]:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

        session_dir = os.path.join(results_base, f"RESULTS_Workload_{workload}_{ts}")

        os.makedirs(session_dir, exist_ok=True)
        
        # Imposta le variabili d'ambiente per farle leggere a config.py
        os.environ["BENCHMARK_MASTER_DIR"] = session_dir
        os.environ["WORKLOAD_TYPE"] = workload
        
        print(f"\n" + "#"*65)
        print(f"FASE WORKLOAD {workload} - Output: {os.path.basename(session_dir)}")
        print("#"*65)

        for name, script in ALGORITHMS:
            if os.path.exists(script):
                print(f"Esecuzione: {name}...")
                # Lancia lo script e aspetta che finisca
                # IMPORTANTE: Passa esplicitamente l'environment per garantire che le variabili
                # d'ambiente (BENCHMARK_MASTER_DIR, WORKLOAD_TYPE) vengano ereditate dal subprocess.
                # Questo è critico sul server Debian dove l'ereditarietà dell'environment
                # potrebbe non funzionare se non passata esplicitamente.
                subprocess.run([sys.executable, "-u", script], env=os.environ.copy())
                time.sleep(3) # Pausa di respiro per il PC tra un algoritmo e l'altro
            else:
                print(f"ATTENZIONE: Script non trovato: {script}")
        
        print(f"\nGenerazione automatica grafici Workload {workload}...")
        if os.path.exists(PLOTTER_SCRIPT):
            subprocess.run([sys.executable, "-u", PLOTTER_SCRIPT], env=os.environ.copy())
            print(f"Grafici generati in: {os.path.basename(session_dir)}")
        else:
            print(f"PLOTTER NON TROVATO: {PLOTTER_SCRIPT}")

        print(f"\nFASE WORKLOAD {workload} COMPLETATA CON SUCCESSO!")
        time.sleep(5)

    # [FASE 3] PULIZIA FINALE: Solo quando abbiamo finito TUTTI i workload
    full_teardown()

    duration = (time.time() - start_time_total) / 60.0
    print("\n" + "="*65)
    print(f"TUTTO COMPLETATO in {duration:.1f} minuti!")
    print(f"Controlla le cartelle appena create per i tuoi risultati e grafici.")
    print("="*65)

if __name__ == "__main__":
    run_automation()
