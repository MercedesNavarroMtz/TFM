from datetime import datetime
import numpy as np



def handle_nan_values(metrics):
    """Procesa un diccionario de métricas para convertir todos los NaN en None."""
    return {key: None if np.isnan(value) else value for key, value in metrics.items()}