import sys , os , numpy as np
import pickle , zipfile
import tensorflow as tf
from tensorflow.keras.models import load_model

base_path = '/content/drive/MyDrive/emotion-detection'

# Ajouter le chemin du dossier src à la liste des chemins 
# où Python cherche lorsqu'on fait import
sys.path.append(os.path.join(base_path, 'src') )
from config import CONFIGS 

sys.path.append(os.path.join(base_path, 'src/utils') )
from visualization import plot_confusion_matrix

def get_pickle_file(file_path) :
  with open(os.path.join(file_path) , "rb") as f :
       return pickle.load(f)


def load_fer2013_test(dataset_path) :
  X_test = get_pickle_file(os.path.join(dataset_path , "X_test.pkl"))
  Y_test = get_pickle_file(os.path.join(dataset_path , "Y_test.pkl"))

  print("X_test shape : " , X_test.shape)
  print("Y_test shape : " , Y_test.shape)

  return X_test , Y_test

def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(os.path.join(base_path , zip_path), 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def normalize(x, y):
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    return normalization_layer(x), y

def load_rafdb_test(zip_path , dataset_path ,image_size , color_mode , batch_size , label_mode) : 

  extract_zip( zip_path, '/content/')

  test_ds = tf.keras.utils.image_dataset_from_directory(
    f'{dataset_path}/test',
    image_size = image_size ,
    color_mode = color_mode,
    batch_size = batch_size ,
    shuffle =False ,
    label_mode = label_mode
     )
  
  print("Classes :", test_ds.class_names)

  # Normalisation et optimisation 
  AUTOTUNE = tf.data.AUTOTUNE
  test_ds = test_ds.map(normalize)
  test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
  

  return test_ds
  
def evaluate_fer2013() : 
  dataset_type = 'fer2013'
  # Chargement du modèle 
  model_fer2013 = load_model(os.path.join(base_path , CONFIGS[dataset_type]["model_path"]))


  # Chargement des données d'évaluation
  X_test , Y_test = load_fer2013_test( os.path.join( base_path,CONFIGS[dataset_type]["dataset_path"]))

  # Evaluation du modèle 
  loss , accuracy = model_fer2013.evaluate(X_test , Y_test)
  print("Loss : " , loss)
  print("Accuracy : " , accuracy)

  # Matrice de confusion 
  y_true = np.argmax(Y_test, axis=1)
  y_pred = np.argmax(model_fer2013.predict(X_test), axis=1)
  plot_confusion_matrix(y_true , y_pred)

def evaluate_rafdb() : 
  dataset_type = 'rafdb'
  
  zip_path = CONFIGS[dataset_type]["zip_path"]
  dataset_path = CONFIGS[dataset_type]["dataset_path"]
  image_size = CONFIGS[dataset_type]["image_size"]
  color_mode = CONFIGS[dataset_type]["color_mode"]
  batch_size = CONFIGS[dataset_type]["batch_size"]
  label_mode = CONFIGS[dataset_type]["label_mode"]


  # Chargement du modèle 
  model_rafdb = load_model(os.path.join(base_path , CONFIGS[dataset_type]["model_path"]))

  # Chargement des données d'évaluation
  test_ds = load_rafdb_test(zip_path , dataset_path ,image_size , color_mode , batch_size , label_mode)

  # Evaluation du modèle 
  loss , accuracy = model_rafdb.evaluate(test_ds)
  print("Loss : " , loss)
  print("Accuracy : " , accuracy)

  # Matrice de confusion 
  y_true = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for _, labels in test_ds
  ])
  y_pred = np.argmax(model_rafdb.predict(test_ds), axis=1)

  plot_confusion_matrix(y_true , y_pred)

evaluate_fer2013()
evaluate_rafdb()
