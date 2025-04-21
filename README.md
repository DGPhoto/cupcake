# Cupcake Photo Culling Library

Una potente e flessibile libreria per la selezione di fotografie destinata ai fotografi.

## Panoramica

Cupcake è una libreria completa per la selezione di fotografie progettata per aiutare i fotografi a selezionare in modo efficiente le migliori immagini dai loro scatti. Utilizza analisi avanzate per valutare le immagini in base a qualità tecnica, composizione, rilevamento del soggetto e preferenze dell'utente.

## Caratteristiche

- **Supporto completo di formati**: Gestisce formati standard e RAW di tutti i principali produttori di fotocamere
- **Analisi intelligente delle immagini**: Valuta messa a fuoco, esposizione, composizione e altri aspetti tecnici
- **Integrazione con machine learning**: Sistema di plugin supporta un LLM locale che apprende le preferenze dell'utente
- **Sistema di selezione flessibile**: Flusso di lavoro ottimizzato per selezionare e rifiutare immagini
- **Architettura di plugin estensibile**: Estendi facilmente le funzionalità attraverso il sistema di plugin

## Struttura del progetto

```
cupcake/
├── requirements.txt          # Dipendenze del progetto
├── environment.yml           # Ambiente conda per l'installazione
├── src/
│   ├── __init__.py           # Inizializzazione del pacchetto
│   ├── image_loader.py       # Caricamento immagini ed estrazione metadati
│   ├── image_formats.py      # Definizioni formati immagine supportati
│   ├── analysis_engine.py    # Analisi qualità delle immagini
│   ├── rating_system.py      # Algoritmi di rating e apprendimento preferenze
│   ├── selection_manager.py  # Sistema di selezione e organizzazione
│   ├── storage_manager.py    # Utilità di storage ed esportazione
│   ├── plugin_system.py      # Framework di plugin
│   └── gpu_utils.py          # Gestione accelerazione GPU
├── plugins/
│   ├── __init__.py           # Inizializzazione pacchetto plugin
│   └── llm_style_predictor.py # Plugin predizione stile basato su ML
└── tests/
    ├── __init__.py           # Inizializzazione pacchetto test
    ├── test_image_loader.py  # Test per image loader
    └── ...                   # Test per altri componenti
```

## Installazione

### Opzione 1: Utilizzo di Conda (consigliato)

Questo metodo è consigliato per gestire correttamente le dipendenze:

```bash
# Clona il repository
git clone https://github.com/username/cupcake.git
cd cupcake

# Crea l'ambiente conda
conda env create -f environment.yml

# Attiva l'ambiente
conda activate cupcake-gpu
```

### Opzione 2: Utilizzo di venv e pip

```bash
# Clona il repository
git clone https://github.com/username/cupcake.git
cd cupcake

# Crea un ambiente virtuale
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Installa le dipendenze
pip install -r requirements.txt
```

## Utilizzo

```python
from src.image_loader import ImageLoader
from src.analysis_engine import AnalysisEngine
from src.rating_system import RatingSystem
from src.selection_manager import SelectionManager
from src.storage_manager import StorageManager

# Inizializzazione dei componenti
image_loader = ImageLoader()
analysis_engine = AnalysisEngine()
rating_system = RatingSystem()
selection_manager = SelectionManager()
storage_manager = StorageManager("./output")

# Caricamento immagini
images = image_loader.load_from_directory("./photos")

# Analisi e classificazione
for path, image_data, metadata in images:
    image_id = metadata['filename']
    analysis = analysis_engine.analyze_image(image_data)
    rating = rating_system.calculate_score(analysis)
    
    # Selezione automatica basata sul rating
    if rating['overall_score'] > 80:
        selection_manager.mark_as_selected(image_id)
    else:
        selection_manager.mark_as_rejected(image_id)

# Esportazione immagini selezionate
storage_manager.export_selected(selection_manager, "./selected_photos")
```

## Componenti principali

### Image Loader
Gestisce il caricamento delle immagini da varie fonti ed estrae i metadati. Supporta un'ampia gamma di formati di immagine inclusi file RAW di tutti i principali produttori di fotocamere.

### Analysis Engine
Analizza le immagini per metriche di qualità tecnica tra cui nitidezza della messa a fuoco, esposizione, composizione e rilevamento dei volti.

### Rating System
Calcola punteggi di qualità complessiva basati sui risultati dell'analisi. Può apprendere dalle selezioni dell'utente per adattarsi alle preferenze individuali.

### Selection Manager
Gestisce lo stato di selezione delle immagini, organizzandole in raccolte e tracciando la cronologia delle selezioni.

### Storage Manager
Gestisce l'esportazione delle immagini selezionate e le organizza secondo vari criteri.

### Plugin System
Fornisce un framework estensibile per aggiungere nuove funzionalità. Include hook per tutte le principali fasi di elaborazione.

### GPU Utils
Gestisce l'accelerazione hardware quando disponibile, ottimizzando le operazioni di elaborazione delle immagini.

## Note sul supporto GPU

Per prestazioni ottimali con accelerazione hardware:

1. Il codice gestisce automaticamente sia la presenza che l'assenza di GPU
2. Non è necessaria alcuna configurazione aggiuntiva per l'uso base
3. Per installazioni avanzate con completo supporto CUDA:
   - Assicurarsi di avere installato i driver NVIDIA aggiornati
   - Installare CUDA Toolkit e cuDNN compatibili

## Licenza

MIT License