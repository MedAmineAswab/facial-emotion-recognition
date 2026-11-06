# Détection des émotions faciales en temps réel 

Système de vision par ordinateur capable de **détecter et classifier les émotions humaines**
à partir d'un flux vidéo en temps réel. Le système combine la capture et l'affichage du flux
vidéo via **OpenCV**, la détection de visage via **MediaPipe** et la classification d'émotions
par un **réseau de neurones convolutif (CNN)** entraîné sur des données faciales annotées.


## Aperçu du projet

Ce projet explore la détection des émotions faciales en temps réel à travers deux
datasets de référence : **FER-2013** et **RAF-DB**, avec pour chacun un pipeline
complet allant du prétraitement jusqu'au déploiement.

### Datasets

| Dataset  |Particularité                                                                                             |
|----------|----------------------------------------------------------------------------------------------------------|
| FER-2013 | Normalisation, reshape, one-hot encoding, split train/val/test — dataset déséquilibré entre les classes  |
| RAF-DB   | Version déjà équilibrée entre les classes, augmentée et divisée en train/val/test                        | 

### Architecture CNN (identique pour les deux modèles)

Deux modèles CNN construits **from scratch** avec la même architecture :

- 4 blocs de convolution : 32 → 64 → 128 → 256 filtres
- Chaque bloc : `Conv2D (same, stride=1)` → `BatchNorm` → `ReLU` → `MaxPooling(2×2)` → `Dropout`
- `Flatten` → `Dense(512)` → `ReLU` → `BatchNorm` → `Dropout(0.5)` → `Dense(7, softmax)`

## Structure du projet 

emotion-detection/
├── app/
│   ├── ai_env/                    # Environnement virtuel Python 3.10.11
│   │                              # (TensorFlow 2.17.0, Keras 3.4.0, OpenCV 4.10.0.82,
│   │                              #  MediaPipe 0.10.9, NumPy 1.26.4)
│   ├── camera.py                  # Flux caméra temps réel, détection de visage via MediaPipe,
│   │                              # inférence des deux modèles, affichage des émotions
|   |
|   ├── requirements.txt           # Dépendances du script caméra
|   |
│   └── README.md                  # Documentation de l'environnement virtuel et commandes
│
├── data/
│   ├── raw/
│   │   ├── fer2013.csv            # Dataset FER-2013 brut (déséquilibré entre classes)
│   │   └── archive.zip            # RAF-DB équilibré (compressé)
│   └── processed/                 # Données FER-2013 prétraitées stockées en pickle
│       ├── X_train.pkl / Y_train.pkl
│       ├── X_val.pkl   / Y_val.pkl
│       └── X_test.pkl  / Y_test.pkl
│
├── models/                        # Meilleurs modèles sauvegardés (checkpoint sur val)
│   ├── best_model_fer2013.h5           # Modèle FER-2013 (format .h5, utilisé dans camera.py)
│   ├── best_model_fer2013.keras        # Modèle FER-2013 (format .keras)
│   ├── best_model_rafdb.h5             # Modèle RAF-DB (format .h5, utilisé dans camera.py)
│   └── best_model_rafdb.keras          # Modèle RAF-DB (format .keras)
│
├── notebooks/
│   ├── 01_setup.ipynb              # Création de la structure Drive + import fer2013.csv
│   ├── 02_eda.ipynb                # Exploration FER-2013 (distribution des classes, visualisation)
│   ├── 03_preprocessing_test.ipynb # Prétraitement FER-2013 (normalisation, reshape, one-hot, pickle)
│   ├── 04_train_cnn_fer2013.ipynb  # CNN FER-2013 : filtres manuels, construction, entraînement,
│   │                               # évaluation, visualisation filtres, matrice de confusion
│   └── 05_train_cnn_rafdb.ipynb    # CNN RAF-DB : lecture par batch, entraînement, évaluation,
│                                   # comparaison des résultats avec FER-2013
│
├── src/
│   ├── data/
│   │   ├── load_data.py           # Chargement des données FER-2013
│   │   ├── preprocess.py          # Normalisation, reshape, affichage des formats
│   │   └── split.py               # Division en train/val/test et enregistrement des splits en format pickle
│   ├── prepare_data.py            # Pipeline complet de prétraitement FER-2013 et one-hot encoding des labels
│   ├── config.py                  # Paramètres globaux et chemins (dictionnaires)
│   ├── model.py                   # Construction de l'architecture CNN
│   ├── train.py                   # Entraînement et sauvegarde du meilleur modèle
│   ├── evaluate.py                # Évaluation sur les données de test
│   └── utils/
│       └── visualization.py       # Courbes loss/accuracy + heatmap de confusion
│
└── README.md                      # Documentation principale du projet

## Installation

### Notebooks et scripts (Google Colab)

Tous les notebooks et scripts s'exécutent directement sur Google Colab sans installation
de dépendances, les bibliothèques nécessaires étant déjà disponibles dans l'environnement Colab.

### Script caméra (local — Windows)

Le script `app/camera.py` est le seul composant nécessitant une exécution locale
avec accès à la caméra.

**Prérequis :** Python 3.10.11

> Pour créer l'environnement virtuel depuis zéro, consulter `app/README.md`.


## Utilisation 

### Lancer le script caméra (local)

```bash
cd app
ai_env\Scripts\activate
python camera.py
```

Le script capture le flux vidéo via **OpenCV**, détecte les visages en temps réel
via **MediaPipe**, applique les deux modèles (FER-2013 et RAF-DB) et affiche
les émotions détectées au-dessus du visage.

### Entraînement et Évaluation

**Via les scripts** — alternative aux notebooks, pour entraîner ou évaluer
le modèle séparément. Les fichiers `src/train.py` et `src/evaluate.py` nécessitent
un environnement comme Google Colab. Ouvrir un notebook Colab et connecter le Drive :

```python
from google.colab import drive
drive.mount('/content/drive')
```

Puis dans une nouvelle cellule :
```python
base_path = '/content/drive/MyDrive/emotion-detection'
import os

train_path = os.path.join(base_path, 'src/train.py')
evaluate_path = os.path.join(base_path, 'src/evaluate.py')
%run $train_path
%run $evaluate_path
```

## Résultats 

| Dataset  | Epochs              | Test Accuracy | Test Loss |
|----------|---------------------|---------------|-----------|
| FER-2013 | 60 (arrêt anticipé) | 63.97%        | 1.0062    |
| RAF-DB   | 60                  | 93.01%        | 0.2426    |

L'arrêt anticipé sur FER-2013 est déclenché par un callback qui surveille la loss
de validation — l'entraînement s'arrête automatiquement après 10 epochs consécutives
sans amélioration.

La différence significative de performance entre les deux modèles s'explique principalement
par le déséquilibre des classes dans FER-2013, contrairement à RAF-DB qui est équilibré
et mieux structuré.

## Technologies utilisées 

### Google Colab

| Technologie  | Version | Usage                                                                            |
|--------------|---------|----------------------------------------------------------------------------------|
| Python       | 3.12.13 | Langage principal                                                                |
| TensorFlow   | 2.20.0  | Framework deep learning                                                          |
| Keras        | 3.13.2  | Utilisé via tf.keras pour la construction et l'entraînement des modèles CNN      |
| NumPy        | 2.0.2   | Manipulation des arrays et des images                                            |
| Pandas       | 2.2.2   | Chargement et exploration de FER-2013 (DataFrame)                                |
| Matplotlib   | 3.10.0  | Visualisation des images, filtres et courbes                                     |
| Seaborn      | 0.13.2  | Heatmap de la matrice de confusion                                               |
| Scikit-learn | 1.6.1   | Calcul de la matrice de confusion                                                |
| Pickle       |   -     | Enregistrement des splits prétraités                                             |
| Zipfile      |   -     | Décompression de l'archive RAF-DB                                                |

zipfile et pickle font partie de la bibliothèque standard Python, pas de version à afficher.

### Local (Windows)

| Technologie | Version   | Usage                               |
|-------------|-----------|-------------------------------------|
| Python      | 3.10.11   | Langage principal                   |
| TensorFlow  | 2.17.0    | Chargement et inférence des modèles |
| Keras       | 3.4.0     | Chargement des modèles              |
| OpenCV      | 4.10.0.82 | Capture du flux vidéo               |
| MediaPipe   | 0.10.9    | Détection de visage en temps réel   |
| NumPy       | 1.26.4    | Manipulation des arrays             |
| Protobuf    | 3.20.3    | Compatibilité MediaPipe/TensorFlow  |


## Auteurs

|------------------|-----------------------------------------------------------------------------|
| Noms             | Mohamed Amine Aswab & Mehdi Ammal                                           |
| Formation        | 4ème année Ingénierie — Spécialité Intelligence Artificielle & Data Science |
| École            | EMSI — École Marocaine des Sciences de l'Ingénieur                          |
| Année académique | 2025/2026                                                                   |