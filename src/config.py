# CCONFIGS                        ← dictionnaire principal
#  ├── 'rafdb'    → { ... }       ← clé : string, valeur : dictionnaire
#  └── 'fer2013'  → { ... }       ← clé : string, valeur : dictionnaire

CONFIGS = {
'rafdb': {
    'zip_path'     : 'data/raw/archive.zip',  # chemin du zip dans Drive
    'dataset_path' : '/content/archive',             # où il sera extrait dans Colab
    'input_shape'  : (75, 75, 1),
    'num_classes'  : 7,
    'batch_size'   : 64,
    'model_path'   : 'models/best_model_rafdb.keras',
    'image_size'   : (75, 75),
    'color_mode'   : 'grayscale',
    'label_mode'   : 'categorical',   
} , 
  'fer2013' : {
    'dataset_path' : 'data/processed' , 
    'input_shape' : (48, 48, 1) , 
    'num_classes'  : 7,
    'batch_size'   : 64,
    'model_path'   : 'models/best_model_fer2013.keras'
  }
}