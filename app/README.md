# Environnement de fonctionnement — Détection d'émotions en temps réel

## Création et activation de l'environnement virtuel

py -3.10 -m venv ai_env
ai_env\Scripts\activate

## Mettre à jour pip

pip install --upgrade pip

## Versions des bibliothèques

| Bibliothèque | Version   |
|--------------|-----------|
| Python       | 3.10.11   |
| TensorFlow   | 2.17.0    |
| Keras        | 3.4.0     |
| OpenCV       | 4.10.0.82 |
| MediaPipe    | 0.10.9    |
| NumPy        | 1.26.4    |
| Protobuf     | 3.20.3    |

## Installation

```bash
pip install -r requirements.txt
```

## Notes de compatibilité

- **NumPy** doit rester en `1.26.4` — NumPy 2.x est incompatible avec OpenCV 4.10 et MediaPipe 0.10.9
- **Protobuf** doit rester en `3.20.3` — Les versions 4.x+ sont incompatibles avec MediaPipe 0.10.9
- **Keras** doit être en `3.4.0` pour supporter `DTypePolicy` utilisé dans les modèles FER2013/RAF-DB
- **L'ordre d'installation est important** — `numpy` et `protobuf` doivent être installés avant `tensorflow` et `mediapipe`
