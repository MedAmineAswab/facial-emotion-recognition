
import numpy as np 

def preprocess_data(images) : 
    """
    Prétraite les images pour un modèle CNN. 
    Cette fonction convertit des images au format string (pixels séparés par des espaces) 
    en tenseurs numpy normalisés et compatibles avec un CNN.

    Etapes : 
    1. Conversion des chaînes de caractères en tableaux numpy
    2. Reshape en images 48x48
    3. Normalisation des pixels (entre 0 et 1)
    4. Ajout de la dimension du canal (grayscale → 1 canal)

    Parameters
    ----------
      images : list or array-like
          Liste de chaînes représentant les pixels des images.

    Returns
    -------
      numpy.ndarray
          Tableau de forme (N, 48, 48, 1) prêt pour un CNN.

    """
    images = np.array([np.fromstring(image, dtype=int, sep=' ') 
                        for image in images])

    images = images.reshape(-1 , 48 , 48 )

    images = images / 255.0 

    images = np.expand_dims(images , -1)

    print("Final shape:", images.shape)
    
    return images
