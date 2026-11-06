import cv2
import mediapipe as mp
import numpy as np
import time
import mediapipe as mp
import tensorflow as tf , keras
import os 

# Récupérer le chemin du dossier emotion-detection
# quel que soit l'endroit depuis lequel il est lancé
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Patch pour ignorer quantization_config
class PatchedDense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# Recharger le modèle entrainé sur FER2013
model_fer2013 = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "models", "best_model_fer2013.h5"),
    custom_objects={'Dense': PatchedDense},
    compile=False
)

# Recharger le modèle entrainé sur RAF-DB
model_rafdb = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "models", "best_model_rafdb.h5"),
    custom_objects={'Dense': PatchedDense},
    compile=False
)

# La liste des émotions 
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# MediaPipe est une bibliothèque de Google basée sur des modèles de machine learning pré-entraînés.
# Elle permet de détecter des visages en temps réel dans une image (frame) ou une vidéo.
# Le modèle retourne les coordonnées des visages détectés ainsi qu'un score de confiance.
# Le paramètre min_detection_confidence (ex: 0.5) permet de filtrer les détections :
# seules les prédictions avec une confiance ≥ 50% sont conservées.

# Init MediaPipe
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)


# OpenCV est une bibliothèque de vision par ordinateur utilisée pour manipuler les images.
# Dans ce projet, elle sert à :
# - accéder à la webcam et lire les frames en temps réel
# - convertir les images (BGR → RGB ou gris)
# - dessiner des rectangles autour des visages détectés
# - découper la région du visage (ROI)
# - redimensionner les images pour le modèle CNN
# - afficher le résultat final à l'écran

# Camera
# cv2.VideoCapture() : sert à ouvrir une source de vidéo 
# 0 : l'index pour designer la caméra par défaut(principale)
cap = cv2.VideoCapture(0)


# last_capture_time stocke le timestamp de la dernière frame traitée
# CAPTURE_INTERVAL définit le temps minimum entre deux traitements (0.5s)
# Cela permet de limiter le nombre de frames analysées par seconde
# afin de réduire la charge CPU/GPU et stabiliser les prédictions

last_capture_time = 0
CAPTURE_INTERVAL = 0.5 # secondes


while True:
    # cap.read() récupère une frame depuis la webcam
    # ret indique si la lecture a réussi (True) ou échoué (False)
    # frame contient l'image capturée
    # si ret == False, cela signifie que la caméra ne fournit plus de frames
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Récupération des dimensions de l'image
    # h = hauteur, w = largeur , _ : canaux (BGR)
    h, w, _ = frame.shape

    # OpenCV utilise BGR tandis que MediaPipe nécessite du RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Détection des visages dans l'image
    # MediaPipe retourne des bounding boxes en coordonnées relatives (0 à 1)
    results = face_detection.process(rgb)
    
    # Liste pour stocker les visages détectés par le modèle 
    faces = []

    # -----------------------------
    # DETECTION DES VISAGES
    # -----------------------------
    if results.detections:
        # Vérifie si MediaPipe a détecté au moins un visage.
        # results.detections est une liste d'objets Detection.
        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box
            
            # Conversion des coordonnées relatives en pixels.
            # On passe du repère normalisé au repère réel de l'image.
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Sécurité : si le visage déborde à gauche ou en haut,
            # on force les coordonnées à rester dans l'image.
            x = max(0, x)
            y = max(0, y)
            
            # Marge à ajouter autour du visage (en pixels)
            padding = 10

            # Agrandir le rectangle avec le padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + width  + padding)
            y2 = min(h, y + height + padding)
            
            # Dessin de la bounding box du visage sur le frame
            # afin de visualiser la détection en temps réel.
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        
            # Extraction de la région du visage (ROI = Region Of Interest).
            # On découpe uniquement la zone délimitée par la bounding box.
            face = frame[y1:y2, x1:x2]

        # Vérifie que la région extraite contient bien des pixels.
        # Cela évite des erreurs si la boîte est invalide ou vide.
        if face.size != 0:

            # Ajout du visage extrait dans la liste des visages détectés.
            faces.append(face)

    # Vérifie qu'au moins un visage a été détecté.
    if len(faces) > 0:

        # Sélection du plus grand visage détecté.
        # Cela permet généralement de suivre le visage principal
        # et d'obtenir une image plus stable pour la prédiction.
        face = max(faces, key=lambda f: f.shape[0] * f.shape[1])

        # Temps actuel en secondes 
        current_time = time.time()

        # Vérifie que l'intervalle minimal entre deux captures
        # a été respecté afin d'éviter de traiter toutes les frames.
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            
            # ── FER2013 (48×48, grayscale) ──────────────────────────────

            # Redimensionnement à 48×48 pixels (taille attendue par FER2013)
            face_resized_fer = cv2.resize(face, (48, 48))

            # Conversion en niveaux de gris (FER2013 est entraîné sur des images grayscale)
            face_gray = cv2.cvtColor(face_resized_fer, cv2.COLOR_BGR2GRAY)

            # Normalisation des pixels entre 0 et 1
            face_normalized = face_gray / 255.0

            # Reshape : ajout des dimensions batch et canal → (1, 48, 48, 1)
            face_final_fer = np.reshape(face_normalized, (1, 48, 48, 1))

            # Prédiction : récupère l'index de l'émotion la plus probable
            emotion_index_fer = np.argmax(model_fer2013.predict(face_final_fer), axis=1)[0]

            # Récupère le label correspondant à l'index
            emotion_label_fer = EMOTION_LABELS[emotion_index_fer]
            
            # ── RAF-DB (75×75, grayscale) ────────────────────────────────

            # Redimensionnement à 75×75 pixels (taille attendue par RAF-DB)
            face_resized_raf = cv2.resize(face, (75, 75))

            # Conversion en niveaux de gris
            face_gray_raf = cv2.cvtColor(face_resized_raf, cv2.COLOR_BGR2GRAY)

            # Normalisation des pixels entre 0 et 1
            face_normalized_raf = face_gray_raf / 255.0

            # Reshape : ajout des dimensions batch et canal → (1, 75, 75, 1)
            face_final_raf = np.reshape(face_normalized_raf, (1, 75, 75, 1))

            # Prédiction : récupère l'index de l'émotion la plus probable
            emotion_index_raf = np.argmax(model_rafdb.predict(face_final_raf), axis=1)[0]

            # Récupère le label correspondant à l'index
            emotion_label_raf = EMOTION_LABELS[emotion_index_raf]
            
            # ── Affichage FER2013 (rouge, au-dessus du visage) ───────────

            text_fer = f"FER: {emotion_label_fer}"

            # Calcul de la largeur du texte pour centrage horizontal
            (text_width_fer, _), _ = cv2.getTextSize(text_fer, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # Position centrée : milieu du visage - moitié de la largeur du texte
            text_x_fer = x + (width // 2) - (text_width_fer // 2)

            cv2.putText(frame, text_fer, (text_x_fer, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # ── Affichage RAF-DB (jaune, juste au-dessus du visage) ───────

            text_raf = f"RAF-DB: {emotion_label_raf}"

            # Calcul de la largeur du texte pour centrage horizontal
            (text_width_raf, _), _ = cv2.getTextSize(text_raf, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # Position centrée
            text_x_raf = x + (width // 2) - (text_width_raf // 2)

            cv2.putText(frame, text_raf, (text_x_raf, y - 10),  
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Mise à jour du temps de la dernière capture.
            last_capture_time = current_time

    # Affichage du flux vidéo avec les rectangles des visages.
    cv2.imshow("Face Detection", frame)
    
    # attendre 1 milliseconde
    # Quitter le programme si l'utilisateur appuie sur la touche 'q'.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération de la caméra.
cap.release()

# Fermeture de toutes les fenêtres OpenCV.
cv2.destroyAllWindows()