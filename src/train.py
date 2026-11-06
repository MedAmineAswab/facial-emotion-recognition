import sys , os
import pickle , zipfile
import tensorflow as tf

base_path = '/content/drive/MyDrive/emotion-detection'

# Ajouter le chemin du dossier src à la liste des chemins 
# où Python cherche lorsqu'on fait import
sys.path.append(os.path.join(base_path, 'src') ) 


from config import CONFIGS
from model import build_model

sys.path.append(os.path.join(base_path, 'src/utils') )
from visualization import plot_training_history

def get_pickle_file(file_path) :
  with open(os.path.join(base_path ,file_path) , "rb") as f :
      return pickle.load(f)

def train_model_fer2013(dataset_path):
    X_train = get_pickle_file(os.path.join(dataset_path, "X_train.pkl"))
    Y_train = get_pickle_file(os.path.join(dataset_path, "Y_train.pkl"))
    X_val   = get_pickle_file(os.path.join(dataset_path, "X_val.pkl"))
    Y_val   = get_pickle_file(os.path.join(dataset_path, "Y_val.pkl"))

    for split, (X, Y) in [("Train", (X_train, Y_train)),
                           ("Validation", (X_val, Y_val))]:
        print(f"[{split}]  X : {X.shape}  |  Y : {Y.shape}")

    return (X_train, Y_train), (X_val, Y_val)

def get_callbacks(model_path):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(base_path, model_path) , 
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(os.path.join(base_path , zip_path), 'r') as zip_ref:
        zip_ref.extractall(extract_path) 
  
def train_model_rafdb(zip_path , dataset_path ,image_size , color_mode , batch_size , label_mode) :
  extract_zip( zip_path, '/content/')

  train_ds = tf.keras.utils.image_dataset_from_directory(
    f'{dataset_path}/train',
    image_size = image_size ,
    color_mode = color_mode,
    batch_size = batch_size ,
    shuffle =True ,
    label_mode = label_mode
)
  val_ds = tf.keras.utils.image_dataset_from_directory(
    f'{dataset_path}/val',
     image_size = image_size ,
    color_mode = color_mode,
    batch_size = batch_size ,
    shuffle = False ,
    label_mode = label_mode
  )
    # Normalisation et optimisation 
  normalization_layer = tf.keras.layers.Rescaling(1./255)
  AUTOTUNE = tf.data.AUTOTUNE
    
  train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(AUTOTUNE)
  val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(AUTOTUNE)
    
  return train_ds, val_ds


def train_model(dataset_type) : 

  dataset_path = CONFIGS[dataset_type]['dataset_path']
  input_shape = CONFIGS[dataset_type]['input_shape']
  batch_size = CONFIGS[dataset_type]['batch_size']
  num_classes = CONFIGS[dataset_type]['num_classes']
  model_path = CONFIGS[dataset_type]['model_path']

  # Construction et compilation du modèle
  model = build_model(input_shape, num_classes)

  if dataset_type == 'fer2013' : 
        # Chargement des données depuis les fichiers pickle
        (X_train, Y_train), (X_val, Y_val) = train_model_fer2013(dataset_path)
        # Entraînement
        history = model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=60,
            validation_data=(X_val, Y_val),
            callbacks=get_callbacks(model_path) )
        
  elif dataset_type == 'rafdb' :
    zip_path = CONFIGS[dataset_type]['zip_path']
    image_size = CONFIGS[dataset_type]['image_size']
    color_mode = CONFIGS[dataset_type]['color_mode']
    label_mode = CONFIGS[dataset_type]['label_mode']
    train_ds , val_ds = train_model_rafdb(zip_path, dataset_path, image_size, color_mode, batch_size, label_mode)

    # Entraînement
    history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=60,
            callbacks=get_callbacks(model_path) )
    
  plot_training_history(history)


train_model('fer2013')
train_model('rafdb')
