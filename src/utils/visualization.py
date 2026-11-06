import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS
    )
    plt.title('Matrice de confusion')
    plt.xlabel('Classe prédite')
    plt.ylabel('Classe réelle')
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    train_loss     = history.history['loss']
    val_loss       = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy   = history.history['val_accuracy']
    epochs         = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss,     label='Train',      color='blue')
    axes[0].plot(epochs, val_loss,       label='Validation', color='orange', linestyle='--')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_accuracy, label='Train',      color='blue')
    axes[1].plot(epochs, val_accuracy,   label='Validation', color='orange', linestyle='--')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle('Évolution loss et accuracy par epoch', fontsize=14)
    plt.tight_layout()
    plt.show()
