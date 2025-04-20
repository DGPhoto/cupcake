#!/usr/bin/env python3
# examples/complete_workflow.py
#
# Questo script dimostra un flusso di lavoro completo con Cupcake:
# 1. Inizializzazione dell'applicazione con impostazioni personalizzate
# 2. Caricamento e analisi delle immagini RAW
# 3. Valutazione delle immagini usando profili personalizzati
# 4. Selezione automatica e manuale delle immagini
# 5. Esportazione dei risultati

import os
import sys
import logging
from pathlib import Path
import time
import argparse

# Aggiungiamo il percorso radice del progetto per l'importazione
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importazione dei componenti Cupcake
from src import get_settings, initialize_application
from src.rating_system import RatingCategory
from src.storage_manager import ExportFormat, NamingPattern, FolderStructure

def setup_arguments():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Cupcake Photo Culling Library - Workflow completo')
    
    parser.add_argument('--input-dir', '-i', type=str, 
                        help='Directory di input contenente le immagini')
    
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Directory di output per le immagini selezionate')
    
    parser.add_argument('--profile', '-p', type=str, default="black_and_white",
                        help='Profilo di rating da utilizzare (default: black_and_white)')
    
    parser.add_argument('--format', '-f', type=str, default="original",
                        choices=['original', 'jpeg', 'tiff', 'png'],
                        help='Formato di esportazione (default: original)')
                        
    parser.add_argument('--threshold', '-t', type=float, default=75.0,
                        help='Soglia di auto-selezione (default: 75.0)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Abilita output verboso')
    
    return parser.parse_args()

def print_section(title):
    """Print a section title."""
    width = 80
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")

def print_progress(current, total, message="Elaborazione"):
    """Print progress bar."""
    width = 50
    progress = int(width * current / total)
    sys.stdout.write("\r{0}: [{1}{2}] {3}/{4} ".format(
        message, "#" * progress, "." * (width - progress), current, total))
    sys.stdout.flush()

def main():
    """Main workflow demonstration."""
    # Parse arguments
    args = setup_arguments()
    
    # Setup logging
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=logging_level, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("cupcake.workflow")
    
    print_section("CUPCAKE PHOTO CULLING LIBRARY - WORKFLOW COMPLETO")
    
    # 1. Inizializzazione applicazione con impostazioni personalizzate
    print("Inizializzazione applicazione...")
    
    # Ottieni impostazioni e inizializza applicazione
    settings = get_settings()
    
    # Configura directory di output da args o usa impostazioni
    if args.output_dir:
        output_dir = args.output_dir
        settings.set_setting("output_directory", output_dir)
    else:
        output_dir = settings.get_output_directory()
    
    print(f"Directory di output: {output_dir}")
    
    # Crea profilo specializzato per bianco e nero se non esiste
    profile_name = args.profile
    if profile_name not in settings.profiles and profile_name == "black_and_white":
        print(f"Creazione profilo specializzato per bianco e nero...")
        settings.create_specialized_profile("black_and_white", "black_and_white")
    
    # Inizializza componenti applicazione
    components = initialize_application()
    image_loader = components["image_loader"]
    analysis_engine = components["analysis_engine"]
    rating_system = components["rating_system"]
    selection_manager = components["selection_manager"]
    storage_manager = components["storage_manager"]
    
    # 2. Determina directory input
    if args.input_dir:
        input_dir = args.input_dir
    else:
        # Chiedi directory se non specificata
        input_dir = input("Inserisci il percorso della directory con le immagini: ")
    
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print(f"Errore: directory non valida '{input_dir}'")
        return
    
    # Aggiungi directory ai recenti
    settings.add_recent_directory(input_dir)
    
    print_section("CARICAMENTO IMMAGINI")
    
    # Carica immagini dalla directory
    print(f"Caricamento immagini da: {input_dir}")
    start_time = time.time()
    
    try:
        image_results = image_loader.load_from_directory(input_dir)
        load_time = time.time() - start_time
        
        print(f"Caricamento completato in {load_time:.2f} secondi")
        print(f"Trovate {len(image_results)} immagini")
        
        # Mostra alcuni esempi di immagini caricate
        if image_results:
            print("\nEsempio delle prime immagini caricate:")
            for i, (path, _, metadata) in enumerate(image_results[:3]):
                filename = os.path.basename(path)
                print(f"  {i+1}. {filename} - {metadata.get('extension', '?').upper()}")
                if 'camera_make' in metadata and 'camera_model' in metadata:
                    print(f"     {metadata['camera_make']} {metadata['camera_model']}")
    except Exception as e:
        logger.error(f"Errore nel caricamento delle immagini: {e}")
        return
    
    if not image_results:
        print("Nessuna immagine trovata nella directory specificata.")
        return
    
    # Registra immagini con il selection manager
    selection_manager.register_images_from_loader(image_results)
    
    print_section("ANALISI E VALUTAZIONE")
    
    # 3. Analisi e valutazione delle immagini
    print(f"Analisi delle immagini con profilo: {profile_name}")
    
    # Strutture dati per tenere traccia dei risultati
    analysis_results = {}
    image_ratings = {}
    
    # Analizza ogni immagine
    total_images = len(image_results)
    for i, (path, image_data, metadata) in enumerate(image_results):
        # Mostra progresso
        print_progress(i+1, total_images, "Analisi")
        
        try:
            # Analizza l'immagine
            analysis_result = analysis_engine.analyze_image(image_data, metadata)
            analysis_results[path] = analysis_result
            
            # Valuta l'immagine con il profilo specificato
            image_rating = rating_system.rate_image(
                analysis_result, path, profile_name, 
                apply_preferences=True  # Applica preferenze se disponibili
            )
            image_ratings[path] = image_rating
            
            # Logging dettagliato se verboso
            if args.verbose and i < 5:  # Dettagli solo per le prime 5 immagini
                logger.debug(f"Valutazione di {os.path.basename(path)}:")
                logger.debug(f"  Technical: {image_rating.technical_score:.1f}")
                logger.debug(f"  Composition: {image_rating.composition_score:.1f}")
                logger.debug(f"  Overall: {image_rating.overall_score:.1f}")
                logger.debug(f"  Rating: {image_rating.rating_category.name}")
                
        except Exception as e:
            logger.error(f"Errore nell'analisi di {path}: {e}")
    
    print()  # Newline after progress bar
    
    # Calcola statistiche delle valutazioni
    rating_counts = {category: 0 for category in RatingCategory}
    for rating in image_ratings.values():
        rating_counts[rating.rating_category] += 1
    
    print("\nDistribuzione delle valutazioni:")
    for category, count in rating_counts.items():
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"  {category.name}: {count} ({percentage:.1f}%)")
    
    print_section("SELEZIONE AUTOMATICA")
    
    # 4. Selezione automatica delle immagini
    threshold = args.threshold
    print(f"Selezione automatica delle immagini (soglia: {threshold})")
    
    # Conta selezioni e rifiuti
    selected_count = 0
    rejected_count = 0
    
    # Seleziona immagini in base al punteggio
    for path, rating in image_ratings.items():
        if rating.overall_score >= threshold:
            selection_manager.mark_as_selected(path)
            selected_count += 1
        else:
            selection_manager.mark_as_rejected(path)
            rejected_count += 1
    
    print(f"Immagini selezionate automaticamente: {selected_count}")
    print(f"Immagini rifiutate automaticamente: {rejected_count}")
    
    # Stampare alcune delle immagini selezionate
    selected_images = selection_manager.get_selected_images()
    if selected_images:
        print("\nEsempio di immagini selezionate:")
        for i, path in enumerate(selected_images[:5]):
            rating = image_ratings[path]
            filename = os.path.basename(path)
            print(f"  {i+1}. {filename} - Punteggio: {rating.overall_score:.1f}")
    
    # 5. Opzione per selezione manuale interattiva
    interactive = input("\nVuoi visualizzare e selezionare manualmente le immagini? (s/n): ").lower()
    
    if interactive.startswith('s'):
        print_section("SELEZIONE MANUALE")
        print("Modalità selezione manuale. Per ogni immagine, puoi:")
        print("  s - Seleziona")
        print("  r - Rifiuta")
        print("  n - Prossima immagine (mantieni stato attuale)")
        print("  q - Termina selezione manuale")
        
        # Ottieni tutte le immagini
        all_images = selection_manager.get_collection_images()
        
        # Interfaccia di selezione manuale semplice
        for i, path in enumerate(all_images):
            rating = image_ratings[path]
            image_info = selection_manager.get_image_info(path)
            status = image_info["status"]
            
            filename = os.path.basename(path)
            
            # Mostra informazioni sull'immagine
            print(f"\nImmagine {i+1}/{len(all_images)}: {filename}")
            print(f"  Rating: {rating.rating_category.name} (Punteggio: {rating.overall_score:.1f})")
            print(f"  Stato attuale: {status.name}")
            
            # Chiedi azione utente
            action = input("Azione [s/r/n/q]: ").lower()
            
            if action == 'q':
                break
            elif action == 's':
                selection_manager.mark_as_selected(path)
                print(f"  → Immagine selezionata")
            elif action == 'r':
                selection_manager.mark_as_rejected(path)
                print(f"  → Immagine rifiutata")
    
    print_section("ESPORTAZIONE")
    
    # 6. Esportazione delle immagini selezionate
    # Determina formato di esportazione
    format_map = {
        "original": ExportFormat.ORIGINAL,
        "jpeg": ExportFormat.JPEG,
        "tiff": ExportFormat.TIFF,
        "png": ExportFormat.PNG
    }
    export_format = format_map.get(args.format, ExportFormat.ORIGINAL)
    
    # Aggiorna conteggio di immagini selezionate
    selected_images = selection_manager.get_selected_images()
    selected_count = len(selected_images)
    
    if selected_count == 0:
        print("Nessuna immagine selezionata da esportare.")
        return
    
    print(f"Esportazione di {selected_count} immagini nel formato {args.format}...")
    print(f"Directory di output: {output_dir}")
    
    # Esecuzione esportazione
    try:
        start_time = time.time()
        
        export_result = storage_manager.export_selected(
            selection_manager,
            output_dir,
            export_format=export_format,
            naming_pattern=NamingPattern.ORIGINAL,
            folder_structure=FolderStructure.DATE,
            jpeg_quality=90 if export_format == ExportFormat.JPEG else None,
            include_xmp=True,
            overwrite=True
        )
        
        export_time = time.time() - start_time
        
        print(f"Esportazione completata in {export_time:.2f} secondi")
        print(f"Risultati esportazione:")
        print(f"  Esportate: {export_result['exported']}")
        print(f"  Saltate: {export_result['skipped']}")
        print(f"  Errori: {export_result['errors']}")
        
        if export_result['exported'] > 0:
            print(f"\nLe immagini selezionate sono state esportate in:")
            print(f"  {os.path.abspath(output_dir)}")
    
    except Exception as e:
        logger.error(f"Errore nell'esportazione: {e}")
    
    print_section("COMPLETATO")
    print("Workflow Cupcake completato con successo!")

if __name__ == "__main__":
    main()