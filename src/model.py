import sys 

# Ajouter le chemin du dossier src à la liste des chemins 
# où Python cherche lorsqu'on fait import
sys.path.append('/content/drive/MyDrive/emotion-detection/src')

from tensorflow.keras import layers , models

def build_emotion_cnn(input_shape, num_classes=7):
    """
    Construit le modèle CNN pour la classification d'émotions.

    Args:
        input_shape (tuple): Dimensions d'une image d'entrée (hauteur, largeur, canaux).
                             (75, 75, 1) pour RAF-DB, (48, 48, 1) pour FER-2013.
        num_classes (int):   Nombre d'émotions à classifier. Par défaut 7.

    Returns:
        model (tf.keras.Sequential): Modèle prêt à être compilé.
    """

    model = models.Sequential([
        # Nouvelle façon recommandée par Keras
        layers.Input(shape=input_shape),
        # ─────────────────────────────────────────────────────────────────
        # BLOC 1 — Détection de features bas niveau (bords, coins, textures)
        # ─────────────────────────────────────────────────────────────────
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ─────────────────────────────────────────────────────────────────
        # BLOC 2 — Features intermédiaires (combinaisons de bords, formes)
        # ─────────────────────────────────────────────────────────────────
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ─────────────────────────────────────────────────────────────────
        # BLOC 3 — Features de haut niveau (parties du visage)
        # ─────────────────────────────────────────────────────────────────
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ─────────────────────────────────────────────────────────────────
        # BLOC 4 — Features très abstraites (représentation globale du visage)
        # ─────────────────────────────────────────────────────────────────
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ─────────────────────────────────────────────────────────────────
        # CLASSIFIEUR — Décision finale basée sur toutes les features
        # ─────────────────────────────────────────────────────────────────
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def build_model(input_shape, num_classes , summary = False) : 

  model = build_emotion_cnn(input_shape, num_classes)

  if summary : 
    model.summary()

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  return model
