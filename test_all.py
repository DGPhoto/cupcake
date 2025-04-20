import os
import sys
from src.image_loader import ImageLoader
from src.analysis_engine import AnalysisEngine
from src.rating_system import RatingSystem, RatingProfile
from src.selection_manager import SelectionManager
from src.storage_manager import StorageManager, ExportFormat, NamingPattern, FolderStructure

# Configurazione dei percorsi
input_directory = "F:/RAW/test"  # Sostituisci con il percorso reale
output_directory = "F:/RAW/testoutput"  # Sostituisci con il percorso di output desiderato

# Inizializzazione dei componenti
image_loader = ImageLoader()
analysis_engine = AnalysisEngine()
rating_system = RatingSystem()
selection_manager = SelectionManager("Sessione Fotografica")
storage_manager = StorageManager(output_directory)

# 1. Caricamento delle immagini
print(f"Caricamento immagini da {input_directory}...")
loaded_images = image_loader.load_from_directory(input_directory)
print(f"Caricate {len(loaded_images)} immagini")

# Registrazione delle immagini nel selection manager
selection_manager.register_images_from_loader(loaded_images)

# 2. Analisi delle immagini e valutazione
analysis_results = {}
print("\nAnalisi e valutazione delle immagini...")

for i, (path, image_data, metadata) in enumerate(loaded_images):
    print(f"Analisi immagine {i+1}/{len(loaded_images)}: {os.path.basename(path)}", end='\r')
    
    # Analisi dell'immagine
    analysis_result = analysis_engine.analyze_image(image_data, metadata)
    analysis_results[path] = analysis_result
    
    # Valutazione dell'immagine
    rating = rating_system.rate_image(analysis_result, path)
    
    # Applicazione automatica della selezione basata sul rating
    if rating.rating_category.value >= 4:  # GOOD o EXCELLENT
        selection_manager.mark_as_selected(path)
    elif rating.rating_category.value <= 2:  # REJECT o BELOW_AVERAGE
        selection_manager.mark_as_rejected(path)
    
    # Imposta anche un rating a stelle basato sul punteggio complessivo
    stars = min(5, max(0, int(rating.overall_score / 20)))
    selection_manager.set_rating(path, stars)

print("\nAnalisi e valutazione completate!")

# 3. Statistiche di selezione
stats = selection_manager.get_statistics()
print("\nStatistiche di selezione:")
print(f"Immagini totali: {stats['total']}")
print(f"Selezionate: {stats['selected']} ({stats['selected_percent']:.1f}%)")
print(f"Rifiutate: {stats['rejected']} ({stats['rejected_percent']:.1f}%)")
print(f"Non valutate: {stats['unrated']} ({stats['unrated_percent']:.1f}%)")

# 4. Esportazione delle immagini selezionate
export_path = os.path.join(output_directory, "selected_images")
print(f"\nEsportazione delle immagini selezionate in {export_path}...")

export_stats = storage_manager.export_selected(
    selection_manager,
    export_path,
    export_format=ExportFormat.JPEG,
    naming_pattern=NamingPattern.SEQUENCE,
    folder_structure=FolderStructure.DATE,
    jpeg_quality=90
)

print(f"Esportazione completata: {export_stats['exported']} immagini esportate.")

# 5. Esportazione dei metadati (compatibilità con Lightroom)
xmp_path = os.path.join(output_directory, "lightroom_metadata")
print(f"\nEsportazione dei metadati per Lightroom in {xmp_path}...")
selection_manager.export_lightroom_metadata(xmp_path)

# 6. Esportazione dei metadati come CSV
csv_path = os.path.join(output_directory, "metadata.csv")
print(f"\nEsportazione dei metadati in CSV: {csv_path}...")
storage_manager.export_with_metadata(
    selection_manager,
    metadata_fields=["camera_make", "camera_model", "focal_length", "f_number", "exposure_time", "iso"],
    output_file=csv_path,
    format="csv"
)

# 7. Salvataggio dello stato di selezione
session_path = os.path.join(output_directory, "session.json")
print(f"\nSalvataggio dello stato della sessione in {session_path}...")
selection_manager.save_to_json(session_path)

print("\nTest completato con successo!")