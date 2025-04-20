# complete_workflow.py

import os
import sys
import logging
import gc
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import psutil
import argparse
import time

# Aggiusta il path Python per trovare il modulo 'src'
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Importa i componenti Cupcake
from src.image_loader import ImageLoader
from src.analysis_engine import AnalysisEngine
from src.rating_system import RatingSystem, RatingCategory
from src.selection_manager import SelectionManager, SelectionStatus
from src.error_suppressor import ErrorSuppressor
from src.storage_manager import StorageManager, ExportFormat
from src.user_settings import UserSettings

# Configura il logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cupcake_workflow.log')
    ]
)

# Silenzia i logger troppo verbosi
logging.getLogger('exifread').setLevel(logging.ERROR)
logging.getLogger('cupcake.analysis').setLevel(logging.WARNING)

logger = logging.getLogger("cupcake.workflow")

# Funzione per il monitoraggio della memoria
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    return f"{memory_mb:.1f} MB"

def process_batch(image_files: List[str], 
                loader: ImageLoader,
                analysis_engine: AnalysisEngine,
                rating_system: RatingSystem,
                selection_manager: SelectionManager,
                rating_profile: str = "default",
                auto_culling: bool = True,
                culling_threshold: float = 75.0,
                pbar=None) -> Dict[str, int]:
    """
    Processa un batch di immagini.
    
    Args:
        image_files: Lista di file da processare
        loader: Istanza di ImageLoader
        analysis_engine: Istanza di AnalysisEngine
        rating_system: Istanza di RatingSystem
        selection_manager: Istanza di SelectionManager
        rating_profile: Nome del profilo di rating
        auto_culling: Se selezionare automaticamente le immagini
        culling_threshold: Soglia per selezione automatica
        pbar: Progress bar
        
    Returns:
        Statistiche del batch
    """
    stats = {
        "processed": 0,
        "selected": 0,
        "rejected": 0,
        "errors": 0
    }
    
    for file_path in image_files:
        try:
            # Carica e analizza l'immagine
            with ErrorSuppressor.suppress_stderr():
                image_data, metadata = loader.load_from_path(file_path)
                
                # Aggiorna la barra di progresso
                if pbar:
                    file_name = os.path.basename(file_path)
                    mem_usage = get_memory_usage()
                    pbar.set_postfix_str(f"File: {file_name} | Mem: {mem_usage}")
                
                # Registra immagine
                image_id = file_path
                selection_manager.register_image(image_id, metadata)
                
                # Analizza l'immagine
                analysis_result = analysis_engine.analyze_image(image_data, metadata)
                
                # Valuta l'immagine
                rating = rating_system.rate_image(
                    analysis_result,
                    image_id,
                    profile_name=rating_profile
                )
                
                # Selezione automatica
                if auto_culling:
                    if rating.overall_score >= culling_threshold:
                        selection_manager.mark_as_selected(image_id)
                        stats["selected"] += 1
                    else:
                        selection_manager.mark_as_rejected(image_id)
                        stats["rejected"] += 1
                
                stats["processed"] += 1
                
                # Libera memoria
                del image_data
                del analysis_result
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            stats["errors"] += 1
        
        # Forza il garbage collector dopo ogni immagine
        gc.collect()
        
        # Aggiorna la progress bar
        if pbar:
            pbar.update(1)
    
    return stats

def process_directory(
    directory_path: str,
    output_dir: Optional[str] = None,
    rating_profile: str = "default",
    export_format: ExportFormat = ExportFormat.ORIGINAL,
    auto_culling: bool = True,
    culling_threshold: float = 75.0,
    batch_size: int = 10,
    use_gpu: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Processa tutte le immagini in una directory con Cupcake.
    
    Args:
        directory_path: Percorso alla directory con le immagini
        output_dir: Directory di output (opzionale)
        rating_profile: Nome del profilo di rating da usare
        export_format: Formato di esportazione
        auto_culling: Se selezionare automaticamente le immagini in base al rating
        culling_threshold: Soglia di rating per la selezione automatica
        batch_size: Numero di immagini da processare in un batch
        use_gpu: Se usare l'accelerazione GPU (None = usa impostazione predefinita)
        
    Returns:
        Dizionario con statistiche di elaborazione
    """
    # Carica le impostazioni utente
    settings = UserSettings()
    
    # Determina se usare la GPU in base ai parametri e alle impostazioni utente
    if use_gpu is None:
        use_gpu = settings.get_setting("use_gpu", False)
    
    # Inizializza componenti
    loader = ImageLoader()
    
    # Configura l'uso della GPU 
    config = {"use_gpu": use_gpu}
    analysis_engine = AnalysisEngine(config)
    
    # Verifica se la GPU è effettivamente usata
    gpu_available = analysis_engine.gpu_available
    
    rating_system = RatingSystem()
    selection_manager = SelectionManager(f"Session-{os.path.basename(directory_path)}")
    
    logger.info(f"Inizializzazione elaborazione per: {directory_path}")
    logger.info(f"Utilizzo GPU: {'Sì' if gpu_available else 'No'}")
    logger.info(f"Memoria iniziale: {get_memory_usage()}")
    
    # Verifica directory
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        logger.error(f"La directory {directory_path} non esiste")
        return {"error": f"La directory {directory_path} non esiste"}
    
    # Trova tutte le immagini nella directory
    image_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower().lstrip('.')
            
            from src.image_formats import ImageFormats
            if ImageFormats.is_supported_format(ext):
                image_files.append(file_path)
    
    if not image_files:
        logger.error(f"Nessuna immagine trovata in {directory_path}")
        return {"error": "Nessuna immagine trovata"}
    
    logger.info(f"Trovate {len(image_files)} immagini in {directory_path}")
    
    # Registra la directory come recente nelle impostazioni
    settings.add_recent_directory(directory_path)
    
    # Statistiche totali
    total_images = len(image_files)
    processed = 0
    selected = 0
    rejected = 0
    errors = 0
    
    # Tempo di inizio
    start_time = time.time()
    
    # Crea la barra di progresso
    with tqdm(total=total_images, desc=f"Analisi immagini") as pbar:
        # Suddividi i file in batch
        for i in range(0, len(image_files), batch_size):
            # Estrai il batch corrente
            batch = image_files[i:i+batch_size]
            
            # Aggiorna la barra con informazioni sul batch
            batch_info = f"Batch {i//batch_size+1}/{(total_images+batch_size-1)//batch_size}"
            pbar.set_description(batch_info)
            
            # Processa il batch
            batch_stats = process_batch(
                batch,
                loader,
                analysis_engine,
                rating_system,
                selection_manager,
                rating_profile,
                auto_culling,
                culling_threshold,
                pbar
            )
            
            # Aggiorna le statistiche totali
            processed += batch_stats["processed"]
            selected += batch_stats["selected"]
            rejected += batch_stats["rejected"]
            errors += batch_stats["errors"]
            
            # Dopo ogni batch, libera completamente la memoria
            gc.collect()
    
    # Calcola il tempo totale
    total_time = time.time() - start_time
    images_per_second = processed / total_time if total_time > 0 else 0
    
    # Mostra statistiche finali
    stats = {
        "totale": total_images,
        "elaborati": processed,
        "selezionati": selected,
        "rifiutati": rejected,
        "errori": errors,
        "memoria_finale": get_memory_usage(),
        "tempo_totale": total_time,
        "immagini_al_secondo": images_per_second,
        "gpu_utilizzata": gpu_available
    }
    
    logger.info(f"Elaborazione completata:")
    logger.info(f"  Totale: {total_images} immagini")
    logger.info(f"  Elaborati: {processed} immagini")
    logger.info(f"  Selezionati: {selected} immagini")
    logger.info(f"  Rifiutati: {rejected} immagini")
    logger.info(f"  Errori: {errors} immagini")
    logger.info(f"  Tempo totale: {total_time:.1f} secondi ({images_per_second:.2f} img/s)")
    logger.info(f"  Memoria finale: {get_memory_usage()}")
    
    # Esporta risultati se richiesto
    if output_dir:
        logger.info(f"Esportazione in corso verso {output_dir}...")
        
        # Visualizza una nuova barra di progresso per l'esportazione
        with tqdm(total=selected, desc="Esportazione") as export_pbar:
            def progress_callback(current, total):
                export_pbar.update(1)
                export_pbar.set_postfix_str(f"Mem: {get_memory_usage()}")
            
            storage_manager = StorageManager(output_dir)
            export_stats = storage_manager.export_selected(
                selection_manager,
                output_dir,
                export_format=export_format,
                progress_callback=progress_callback
            )
        
        logger.info(f"Esportati {export_stats['exported']} file in {output_dir}")
        stats["esportati"] = export_stats['exported']
    
    # Rilascia memoria finale
    gc.collect()
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Cupcake Photo Culling Workflow")
    parser.add_argument("--input-dir", required=True, help="Directory with images to process")
    parser.add_argument("--output-dir", help="Output directory for selected images")
    parser.add_argument("--profile", default="default", help="Rating profile to use")
    parser.add_argument("--threshold", type=float, default=75.0, help="Auto-culling threshold")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--format", default="original", choices=["original", "jpeg", "tiff", "png"], 
                        help="Export format")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration if available")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    # Configura il livello di logging in base a verbose
    if args.verbose:
        logging.getLogger("cupcake").setLevel(logging.DEBUG)
    else:
        logging.getLogger("cupcake").setLevel(logging.INFO)
    
    # Converti il formato in enum
    export_format = ExportFormat.ORIGINAL
    if args.format == "jpeg":
        export_format = ExportFormat.JPEG
    elif args.format == "tiff":
        export_format = ExportFormat.TIFF
    elif args.format == "png":
        export_format = ExportFormat.PNG
    
    # Determina l'uso della GPU
    use_gpu = None  # Default alle impostazioni utente
    if args.use_gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False
    
    stats = process_directory(
        args.input_dir,
        args.output_dir,
        args.profile,
        export_format,
        True,  # auto_culling sempre attivo
        args.threshold,
        args.batch_size,
        use_gpu
    )
    
    print(f"\nElaborazione completata:")
    print(f"  Totale: {stats['totale']} immagini")
    print(f"  Elaborati: {stats['elaborati']} immagini")
    print(f"  Selezionati: {stats['selezionati']} immagini")
    print(f"  Errori: {stats['errori']} immagini")
    print(f"  Tempo totale: {stats['tempo_totale']:.1f} secondi ({stats['immagini_al_secondo']:.2f} img/s)")
    print(f"  GPU utilizzata: {'Sì' if stats['gpu_utilizzata'] else 'No'}")
    print(f"  Memoria finale: {stats['memoria_finale']}")
    
    if 'esportati' in stats:
        print(f"  Esportati: {stats['esportati']} immagini in {args.output_dir}")

if __name__ == "__main__":
    main()