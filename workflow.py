# workflow.py

import os
import sys
import logging
import gc
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import psutil

# Importa i componenti Cupcake
from src.image_loader import ImageLoader
from src.analysis_engine import AnalysisEngine
from src.rating_system import RatingSystem, RatingCategory
from src.selection_manager import SelectionManager, SelectionStatus
from src.error_suppressor import ErrorSuppressor

# Configura il logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cupcake_workflow.log')
    ]
)

logger = logging.getLogger("cupcake.workflow")

# Funzione di monitoraggio memoria
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    return f"{memory_mb:.1f} MB"

def process_directory(
    directory_path: str,
    output_dir: Optional[str] = None,
    rating_profile: str = "default",
    auto_culling: bool = True,
    culling_threshold: float = 75.0
) -> Dict[str, Any]:
    """
    Processa tutte le immagini in una directory con Cupcake.
    
    Args:
        directory_path: Percorso alla directory con le immagini
        output_dir: Directory di output (opzionale)
        rating_profile: Nome del profilo di rating da usare
        auto_culling: Se selezionare automaticamente le immagini in base al rating
        culling_threshold: Soglia di rating per la selezione automatica
        
    Returns:
        Dizionario con statistiche di elaborazione
    """
    # Inizializza componenti
    loader = ImageLoader()
    analysis_engine = AnalysisEngine()
    rating_system = RatingSystem()
    selection_manager = SelectionManager(f"Session-{os.path.basename(directory_path)}")
    
    logger.info(f"Inizializzazione elaborazione per: {directory_path}")
    logger.info(f"Uso memoria all'avvio: {print_memory_usage()}")
    
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
    
    logger.info(f"Trovate {len(image_files)} immagini")
    
    # Elabora le immagini
    total = len(image_files)
    processed = 0
    errors = 0
    selected = 0
    rejected = 0
    
    for idx, file_path in enumerate(image_files):
        try:
            print(f"Analisi: [{('#' * int(50 * idx / total)).ljust(50, '.')}] {idx+1}/{total}", end="\r")
            
            # Libera memoria all'inizio di ogni ciclo
            gc.collect()
            
            # Log della memoria ogni 10 immagini
            if idx % 10 == 0:
                logger.info(f"Memoria in uso: {print_memory_usage()}")
            
            # Carica immagine con gestione errori
            try:
                with ErrorSuppressor.suppress_stderr():
                    image_data, metadata = loader.load_from_path(file_path)
                    
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
                
                # Libera memoria esplicitamente
                del image_data
                del analysis_result
                gc.collect()
                
                # Selezione automatica basata su rating
                if auto_culling:
                    if rating.overall_score >= culling_threshold:
                        selection_manager.mark_as_selected(image_id)
                        selected += 1
                    else:
                        selection_manager.mark_as_rejected(image_id)
                        rejected += 1
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Errore nell'analisi di {file_path}: {e}")
                errors += 1
                
        except Exception as e:
            logger.error(f"Errore nel processamento di {file_path}: {e}")
            errors += 1
        
        # Libera memoria alla fine di ogni ciclo
        gc.collect()
    
    print("\n")  # Newline dopo la barra di avanzamento
    
    # Log statistiche finali
    stats = {
        "totale": total,
        "elaborati": processed,
        "errori": errors,
        "selezionati": selected,
        "rifiutati": rejected,
        "memoria_finale": print_memory_usage()
    }
    
    logger.info(f"Elaborazione completata:")
    logger.info(f"  Totale: {total} immagini")
    logger.info(f"  Elaborati: {processed} immagini")
    logger.info(f"  Errori: {errors} immagini")
    logger.info(f"  Selezionati: {selected} immagini")
    logger.info(f"  Rifiutati: {rejected} immagini")
    logger.info(f"  Memoria finale: {print_memory_usage()}")
    
    # Esporta risultati se richiesto
    if output_dir:
        from src.storage_manager import StorageManager, ExportFormat
        
        storage_manager = StorageManager(output_dir)
        export_stats = storage_manager.export_selected(
            selection_manager,
            output_dir,
            export_format=ExportFormat.ORIGINAL
        )
        
        logger.info(f"Esportati {export_stats['exported']} file in {output_dir}")
        stats["esportati"] = export_stats['exported']
    
    # Rilascia memoria finale
    gc.collect()
    
    return stats

def main():
    # Esempio di utilizzo
    import argparse
    
    parser = argparse.ArgumentParser(description="Cupcake Photo Culling Workflow")
    parser.add_argument("directory", help="Directory with images to process")
    parser.add_argument("--output", "-o", help="Output directory for selected images")
    parser.add_argument("--profile", "-p", default="default", help="Rating profile to use")
    parser.add_argument("--threshold", "-t", type=float, default=75.0, help="Auto-culling threshold")
    parser.add_argument("--no-auto", action="store_true", help="Disable auto-culling")
    
    args = parser.parse_args()
    
    stats = process_directory(
        args.directory,
        args.output,
        args.profile,
        not args.no_auto,
        args.threshold
    )
    
    print(f"\nElaborazione completata: {stats}")

if __name__ == "__main__":
    main()