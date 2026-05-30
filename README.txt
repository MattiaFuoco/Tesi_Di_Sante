TESI: Ottimizzazione automatica delle prestazioni di MongoDB mediante tuning dei parametri di configurazione
(Performance Optimization of MongoDB Through Automated Configuration Tuning)


================================================================================
Descrizione dell'Architettura
================================================================================
Questo progetto implementa un framework avanzato di benchmarking e ottimizzazione
per database NoSQL (MongoDB). L'intero sistema gira all'interno di un singolo
container Docker privilegiato che contiene Python, MongoDB, YCSB e tutti gli
strumenti necessari. Lo script di avvio (start_benchmark.sh) si occupa di
costruire l'immagine e lanciare il container con un solo comando.


================================================================================
Architettura dello Storage: Il "Mega-Load" e i 3 Dataset Isolati
================================================================================
Una delle sfide principali nell'ottimizzazione di MongoDB è il tuning della
compressione (WiredTiger Block Compressor). Poiché la compressione dei dati
avviene a livello fisico sul disco, non è possibile testare le prestazioni in
lettura/scrittura cambiando semplicemente un parametro "al volo" su un database
già esistente.

Per garantire tempi di esecuzione efficienti e mantenere l'integrità strutturale,
il framework adotta una strategia basata su 3 Dataset Isolati:

-- Il "Mega-Load" Iniziale (Fase Preliminare):
   Prima che gli algoritmi inizino la loro esplorazione, il Master Runner esegue
   una preparazione massiva. Avvia in sequenza mongod tre volte, assegnando ad
   ogni istanza un compressore nativo diverso (none, snappy, zstd).
   Tramite YCSB, popola fisicamente questi tre database inserendo l'intero dataset
   (1.600.000 record, circa 10 KB cadauno: 10 campi YCSB da fieldlength=1000 byte,
   per un totale non compresso di ~16 GB per dataset).

-- Persistenza Dedicata (3 directory interne al container):
   I dati vengono salvati in directory separate all'interno del container,
   montate come volume sull'host per garantirne la persistenza:

     /data/db/none/    → dataset compresso con compressore "none"
     /data/db/snappy/  → dataset compresso con Snappy
     /data/db/zstd/    → dataset compresso con ZSTD

   Questo elimina il layer overlayfs interno di Docker, garantendo che le
   operazioni I/O di MongoDB colpiscano il filesystem direttamente,
   senza intermediari software che alterino le misurazioni.

-- Switch Dinamico (Fase di Run):
   Durante la fase di ottimizzazione, quando un algoritmo decide di valutare una
   specifica configurazione (es. Compressor: snappy, Cache: 2GB), l'orchestratore
   ferma mongod e lo riavvia puntando alla directory del compressore corretto,
   agganciando al volo il dataset pre-caricato corrispondente.

Questo approccio architetturale permette agli algoritmi di scambiare il
compressore in pochi secondi, simulando la presenza di tre giganteschi database
MongoDB paralleli, senza dover subire il pesantissimo overhead di ricaricare i
dati da zero ad ogni singola iterazione di test.


================================================================================
Gestione della OS Page Cache (Cold Start Garantito)
================================================================================
Per ottenere misurazioni I/O realistiche, prima di ogni run il framework forza
uno svuotamento completo della RAM cache del sistema operativo.

Il container viene lanciato con --privileged, che condivide il namespace del
kernel host. Questo permette di scrivere direttamente su:
  sync; echo 3 > /proc/sys/vm/drop_caches
colpendo il kernel host reale senza bisogno di container intermedi.

Livelli di controllo della cache attivi nel framework:

  1. nocache (wrappa mongod): impedisce a mongod di popolare la page cache
     durante ogni singola operazione I/O.
  2. drop_caches prima di ogni run: svuota tutto ciò che potrebbe essere
     rimasto in cache da processi di sistema o dal run precedente.

Nota: in questa versione del framework il Direct I/O di WiredTiger
(direct_io=[data,log]) è disabilitato per garantire la massima portabilità tra
WSL2 e bare-metal. La combinazione di nocache (a livello processo) e
drop_caches (a livello kernel) è sufficiente a ottenere un Cold Start
ripetibile su entrambi gli ambienti.


================================================================================
Rilevamento Automatico dell'Ambiente
================================================================================
Lo script start_benchmark.sh e il file config.py rilevano automaticamente
l'ambiente in cui stanno girando leggendo /proc/version:

- Se contiene "microsoft" → WSL2
- Altrimenti → Server Debian

Il rilevamento viene esportato nella variabile d'ambiente BENCHMARK_ENV e
serve attualmente solo a scopo informativo (stampa nei log, tracciamento
nelle sessioni di benchmark). L'architettura e il comportamento sono
identici nei due ambienti: il container è lo stesso, le directory dati sono
le stesse, e il Direct I/O è disabilitato in entrambi i casi (vedi sezione
precedente). Non è necessario modificare nulla manualmente: un solo comando
gestisce entrambi gli ambienti.


================================================================================
The "Single Source of Truth"
================================================================================
L'intero framework è governato da un file centrale (config.py). Per garantire
la massima equità accademica e un confronto "ad armi pari", tutti gli algoritmi
euristici (Random Search, Simulated Annealing, Evolutionary Algorithm, Bayesian
Optimization, Coordinate Search, Hill Climbing) condividono un budget di
valutazioni dinamico e centralizzato (EVALUATIONS = 90, circa la metà delle
189 configurazioni totali esplorate dalla Grid Search — pari a 7 valori di
cache × 3 intervalli di journal × 3 compressori × 3 distribuzioni).
Modificando questo singolo parametro, tutti gli algoritmi ricalcoleranno
automaticamente la propria termodinamica, i cicli evolutivi o le fasi esplorative.


================================================================================
Prerequisiti
================================================================================
Prima di lanciare l'esperimento, assicurati di avere a disposizione sul sistema:

- Docker (o Docker Desktop) installato e in esecuzione.
- YCSB (Yahoo Cloud Serving Benchmark): scaricato e configurato
  automaticamente dallo script di avvio se non presente.
  Non è necessario installarlo manualmente.
- Il Dockerfile incluso nel progetto: usato automaticamente per costruire
  l'immagine benchmark-all-in-one:latest che contiene MongoDB 7, Python,
  Java, nocache e tutte le dipendenze. Non richiede alcuna azione manuale.
- Il file .dockerignore incluso nel progetto: esclude automaticamente dal
  build Docker le cartelle inutili (risultati, YCSB, cache Python),
  velocizzando il build dell'immagine.

Nota: Java, Python e le dipendenze Python sono già inclusi nell'immagine
Docker — non è necessario installarli sul sistema host.


================================================================================
Esecuzione (Zero-Click) — UN SOLO COMANDO
================================================================================
Il processo è interamente automatizzato. Lo script di avvio:

  1. Rileva automaticamente l'ambiente (WSL2 o Server Debian).
  2. Scarica e configura YCSB automaticamente se non presente.
  3. Costruisce l'immagine Docker benchmark-all-in-one:latest dal Dockerfile
     incluso (solo se non già presente o se i file sono stati modificati —
     richiede ~3 minuti alla prima esecuzione, poi viene riutilizzata).
     La rebuild è automatica: se config.py o altri file Python vengono
     modificati, lo script rileva il cambiamento e ricostruisce l'immagine.
  4. Crea la cartella results/ sull'host per la persistenza dei risultati.
  5. Avvia il container unico con tutto dentro.

Il comando da eseguire è sempre e solo:

    sudo bash start_benchmark.sh

Questo vale sia su WSL2 (da terminale Ubuntu/Debian) che su Server Debian.
Non è necessario impostare variabili d'ambiente o modificare file di configurazione.


================================================================================
Cosa fa il container in background?
================================================================================
All'avvio il container esegue master_runner.py che:
  1. Esegue il Mega-Load iniziale: avvia mongod 3 volte (una per compressore)
     e popola i 3 dataset tramite YCSB.
  2. Per ogni workload (A, B, D) esegue in sequenza tutti e 7 gli algoritmi
     di ottimizzazione.
  3. Prima di ogni run, svuota la OS Page Cache per garantire il Cold Start.
  4. Al termine genera i grafici comparativi (Anytime Performance).
  5. Esegue il teardown finale.

I risultati sopravvivono al destroy del container perché vengono salvati
nella cartella results/ montata sull'host.


================================================================================
Cosa accade durante l'esecuzione?
================================================================================
Il sistema avvierà in sequenza i 7 algoritmi per i seguenti scenari (standard YCSB):

  Workload A (50/50):       50% Read / 50% Update
  Workload B (95/5):        95% Read / 5% Update
  Workload D (Read Latest): 95% Letture / 5% Inserimenti

mongod viene avviato e fermato dinamicamente per ogni configurazione testata.
Prima di ogni singola iterazione di test, viene forzato il drop delle cache in
RAM per garantire risultati stabili e non alterati dalle letture precedenti
(Cold Start garantito).


================================================================================
Risultati Generati
================================================================================
Al termine dell'esecuzione, tutti i file CSV e i grafici verranno salvati
automaticamente nella cartella results/ nella directory del progetto. Troverai:

- Cartelle results/RESULTS_Workload_X_...:
  Contengono i file CSV grezzi e le Heatmap (Grid/Topografiche) di ogni singolo
  algoritmo per i Workload A, B e D. Le mappe mostrano il percorso decisionale
  di ciascun algoritmo.

- File REPORT_STEPS_WL_X.{png,pdf,svg}:
  Grafico aggregato comparativo (Anytime Performance) basato sul budget di step.
  Esportato in tre formati: PNG (300 DPI) per visualizzazione rapida, PDF e
  SVG (vettoriali) per inclusione in tesi e stampa di qualita'.

- File REPORT_TIME_WL_X.{png,pdf,svg}:
  Grafico aggregato comparativo (Anytime Performance) basato sul tempo reale
  (fondamentale per valutare l'overhead introdotto dai processi decisionali
  dell'Intelligenza Artificiale). Esportato negli stessi tre formati.

Baseline Grid Search nei due grafici REPORT_*:
  - Linea nera tratteggiata: media delle ripetizioni della miglior configurazione
    trovata dalla Grid Search (= massimo tra le medie delle configurazioni).
  - Fascia grigia: media +- deviazione standard di quella stessa configurazione.
  - Linee nere punteggiate (sottili): minimo e massimo assoluti effettivamente
    osservati nelle ripetizioni della miglior configurazione. Permettono di
    distinguere un euristico che batte realmente la Grid Search da uno che
    resta dentro la variabilita' di misura.
