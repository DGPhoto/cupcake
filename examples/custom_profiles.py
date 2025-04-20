# examples/custom_profiles.py
#
# Questo esempio mostra come:
# 1. Creare e gestire profili di rating personalizzati
# 2. Utilizzare profili per la fotografia in bianco e nero
# 3. Implementare profili specializzati per diversi generi fotografici
# 4. Allenare i modelli di preferenza utente per i diversi profili
# 5. Esportare e importare impostazioni tra sessioni

import os
import sys
import logging
from pathlib import Path

# Aggiungiamo il percorso radice del progetto per l'importazione
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import get_settings, UserSettings, RatingSystem, ImageLoader, AnalysisEngine
from src.rating_system import RatingCategory, RatingProfile

# Configurazione del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("example")

def main():
    """Esempio di utilizzo del sistema di impostazioni utente e profili personalizzati."""
    print("=== Cupcake Photo Culling Library - Esempio Profili Personalizzati ===\n")
    
    # Otteniamo l'istanza delle impostazioni
    settings = get_settings()
    print(f"Impostazioni caricate da: {settings.settings_dir}")
    
    # Configurazione directory input/output
    settings.set_setting("output_directory", os.path.expanduser("~/Pictures/CupcakeOutput"))
    output_dir = settings.get_output_directory()
    print(f"Directory di output: {output_dir}")
    
    # 1. Creazione di un profilo base personalizzato
    print("\n=== Creazione Profilo Personalizzato ===")
    
    # Verifico profili esistenti
    existing_profiles = settings.load_profiles()
    print(f"Profili disponibili: {', '.join(existing_profiles.keys())}")
    
    # Creo un nuovo profilo personalizzato
    custom_profile = RatingProfile(
        name="il_mio_profilo",
        description="Il mio profilo personalizzato",
        # Pesonalizzo pesi
        technical_weight=0.55,
        composition_weight=0.45,
        # Maggiore enfasi su nitidezza e contrasto
        sharpness_weight=0.4,
        exposure_weight=0.3,
        contrast_weight=0.2,
        noise_weight=0.1,
        # Configurazione composizione
        rule_of_thirds_weight=0.5,
        symmetry_weight=0.2,
        subject_position_weight=0.3,
        # Soglie personalizzate
        excellent_threshold=80.0,
        good_threshold=70.0,
        average_threshold=55.0,
        below_average_threshold=40.0
    )
    
    # Salvo il profilo
    if settings.save_profile(custom_profile):
        print(f"Profilo '{custom_profile.name}' salvato con successo")
    else:
        print(f"Errore nel salvare il profilo")
    
    # 2. Creazione di un profilo specializzato per fotografia in bianco e nero
    print("\n=== Creazione Profilo Bianco e Nero ===")
    
    bw_profile = settings.create_specialized_profile("bianco_e_nero", "black_and_white")
    if bw_profile:
        print(f"Profilo creato: {bw_profile.name} - {bw_profile.description}")
        print(f"Pesi tecnici: sharpness={bw_profile.sharpness_weight}, "
              f"contrast={bw_profile.contrast_weight}, "
              f"exposure={bw_profile.exposure_weight}")
    else:
        print("Errore nella creazione del profilo bianco e nero")
    
    # 3. Creazione di profili per diversi generi fotografici
    print("\n=== Creazione Profili Specializzati ===")
    
    # Creo alcuni profili specializzati
    specializations = ["portrait", "landscape", "wildlife", "street", "night"]
    
    for specialization in specializations:
        profile_name = f"{specialization}_profile"
        profile = settings.create_specialized_profile(profile_name, specialization)
        if profile:
            print(f"Profilo {profile_name} creato con successo")
    
    # Verifichiamo i profili creati
    all_profiles = settings.load_profiles()
    print(f"\nTotale profili disponibili: {len(all_profiles)}")
    
    # 4. Impostazione del profilo predefinito
    print("\n=== Impostazione Profilo Predefinito ===")
    
    settings.set_setting("default_rating_profile", "bianco_e_nero")
    print(f"Profilo predefinito impostato a: bianco_e_nero")
    
    # 5. Simulazione di addestramento delle preferenze utente
    print("\n=== Addestramento Modello Preferenze ===")
    
    # Otteniamo il modello di preferenze per il profilo B&N
    preference_model = settings.get_preference_model("bianco_e_nero")
    
    # Simuliamo l'apprendimento da feedback utente con alcuni aggiustamenti
    print("Simulazione addestramento modello...")
    
    # Modifichiamo manualmente i fattori per simulare un apprendimento
    preference_model.contrast_factor = 1.3  # Preferenza per alto contrasto in B&W
    preference_model.sharpness_factor = 1.1
    preference_model.exposure_factor = 0.9
    preference_model.rule_of_thirds_factor = 1.1
    preference_model.symmetry_factor = 1.2  # Maggiore enfasi su simmetria in B&W
    
    # Salviamo il modello modificato
    settings.save_preference_model("bianco_e_nero", preference_model)
    print("Modello di preferenze aggiornato e salvato")
    
    # 6. Test di applicazione di un profilo
    print("\n=== Applicazione di un Profilo per Valutazione ===")
    
    # Inizializziamo i componenti necessari
    rating_system = RatingSystem()
    
    # Aggiungiamo i nostri profili personalizzati
    for name, profile in all_profiles.items():
        if name != "default":  # 'default' è già caricato da RatingSystem
            rating_system.add_profile(profile)
    
    # Simuliamo l'analisi di un'immagine (senza caricare realmente)
    from src.analysis_engine import ImageAnalysisResult
    
    # Creiamo un risultato di analisi fittizio
    analysis_result = ImageAnalysisResult()
    analysis_result.sharpness_score = 85.0
    analysis_result.exposure_score = 75.0
    analysis_result.contrast_score = 90.0  # Alto contrasto, buono per B&W
    analysis_result.noise_score = 60.0
    analysis_result.rule_of_thirds_score = 65.0
    analysis_result.symmetry_score = 80.0  # Buona simmetria
    analysis_result.overall_technical_score = 80.0
    analysis_result.overall_composition_score = 70.0
    
    # Valutiamo l'immagine con diversi profili
    image_id = "immagine_test.jpg"
    
    # Con profilo predefinito
    default_rating = rating_system.rate_image(analysis_result, image_id, 'default', False)
    
    # Con profilo B&W
    bw_rating = rating_system.rate_image(analysis_result, image_id, 'bianco_e_nero', False)
    
    # Con profilo personalizzato
    custom_rating = rating_system.rate_image(analysis_result, image_id, 'il_mio_profilo', False)
    
    # Con modello di preferenze
    bw_pref_rating = rating_system.rate_image(analysis_result, image_id, 'bianco_e_nero', True)
    
    # Visualizziamo i risultati
    print("Valutazione con diversi profili:")
    print(f"Default: {default_rating.rating_category.name} ({default_rating.overall_score:.1f})")
    print(f"Bianco e Nero: {bw_rating.rating_category.name} ({bw_rating.overall_score:.1f})")
    print(f"Personalizzato: {custom_rating.rating_category.name} ({custom_rating.overall_score:.1f})")
    print(f"B&W con preferenze: {bw_pref_rating.rating_category.name} ({bw_pref_rating.overall_score:.1f})")
    
    # 7. Esportazione e importazione delle impostazioni
    print("\n=== Esportazione e Importazione Impostazioni ===")
    
    # Esportiamo le impostazioni
    export_path = os.path.join(output_dir, "cupcake_settings_export.json")
    if settings.export_settings(export_path):
        print(f"Impostazioni esportate in: {export_path}")
    
    # Simuliamo l'importazione (in una sessione diversa si userebbe un nuovo UserSettings)
    print("Simulazione importazione impostazioni...")
    new_settings = UserSettings(os.path.join(os.path.expanduser("~"), ".cupcake", "temp_settings"))
    
    if new_settings.import_settings(export_path):
        print("Impostazioni importate con successo")
        
        # Verifichiamo i profili importati
        imported_profiles = new_settings.load_profiles()
        print(f"Profili importati: {', '.join(imported_profiles.keys())}")
    
    print("\n=== Esempio completato ===")

if __name__ == "__main__":
    main()