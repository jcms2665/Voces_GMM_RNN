'''
Funciones para GMM
JC

V-1: 25-10-2024
V-2: 01-11-2024

'''


import numpy as np
from pyAudioAnalysis import audioBasicIO
import Audio_Feature_Extraction as AFE
from sklearn.mixture import GaussianMixture
import joblib
import os

'''
Verifica las dimensiones de las características
'''
def check_feature_dimensions(features_1, features_2):
    if features_1.shape[1] != features_2.shape[1]:
        raise ValueError(f"Desajuste en el número de características: "
                         f"primer archivo ({features_1.shape[1]} características), "
                         f"segundo archivo ({features_2.shape[1]} características).")

'''
Extrae MFCC y características adicionales (delta y double delta)
'''

def extract_features(signal, Fs, window_size, overlap):
    MFCC13_F = AFE.stFeatureExtraction(signal, Fs, int(Fs * window_size), int(Fs * overlap))

    # (primera derivada)
    delta_MFCC = np.zeros(MFCC13_F.shape)
    for t in range(delta_MFCC.shape[1]):
        index_t_minus_one, index_t_plus_one = t - 1, t + 1
        index_t_minus_one = max(index_t_minus_one, 0)
        index_t_plus_one = min(index_t_plus_one, delta_MFCC.shape[1] - 1)
        delta_MFCC[:, t] = 0.5 * (MFCC13_F[:, index_t_plus_one] - MFCC13_F[:, index_t_minus_one])

    # (segunda derivada)
    double_delta_MFCC = np.zeros(MFCC13_F.shape)
    for t in range(double_delta_MFCC.shape[1]):
        index_t_minus_one, index_t_plus_one, index_t_plus_two, index_t_minus_two = t - 1, t + 1, t + 2, t - 2
        index_t_minus_one = max(index_t_minus_one, 0)
        index_t_plus_one = min(index_t_plus_one, delta_MFCC.shape[1] - 1)
        index_t_minus_two = max(index_t_minus_two, 0)
        index_t_plus_two = min(index_t_plus_two, double_delta_MFCC.shape[1] - 1)
        double_delta_MFCC[:, t] = 0.1 * (2 * MFCC13_F[:, index_t_plus_two] + MFCC13_F[:, index_t_plus_one]
                                         - MFCC13_F[:, index_t_minus_one] - 2 * MFCC13_F[:, index_t_minus_two])

    # Combinar todo
    Combined_MFCC_F = np.concatenate((MFCC13_F, delta_MFCC, double_delta_MFCC), axis=1)
    
    return Combined_MFCC_F

'''
Función principal
Procesar los archivos de audio
'''

def procesar_audios(file_1, file_2, folder_path):
    file_1_path = os.path.join(folder_path, file_1)
    file_2_path = os.path.join(folder_path, file_2)

    # Leer audio
    [Fs, x] = audioBasicIO.read_audio_file(file_1_path)
    if x is None:
        raise ValueError(f"Error al cargar el archivo de audio {file_1}")

    # Convertir a mono si el audio tiene más de un canal
    # Esto hay que validarlo con Fernanda/Sofía

    if x.ndim > 1:
        x = np.mean(x, axis=1)

    # Parámetros (los mismos del github)
    window_size = 0.030  # Tamaño de la ventana en segundos
    overlap = 0.015      # Sobreposición entre ventanas
    VTH_Multiplier = 0.05
    VTH_range = 100
    n_mixtures = 16
    max_iterations = 75

    # Energía de la señal
    energy = [s**2 for s in x]
    Voiced_Threshold = VTH_Multiplier * np.mean(energy)

    # Filtrar muestras con voz
    clean_samples = []
    for sample_set in range(0, len(x) - VTH_range, VTH_range):
        sample_set_th = np.mean(energy[sample_set:sample_set + VTH_range])
        if sample_set_th > Voiced_Threshold:
            clean_samples.extend(x[sample_set:sample_set + VTH_range])

    if len(clean_samples) == 0:
        raise ValueError(f"No se encontraron muestras con voz en el archivo de audio {file_1}")

    Combined_MFCC_F = extract_features(np.array(clean_samples), Fs, window_size, overlap)

    gmm = GaussianMixture(n_components=n_mixtures, covariance_type='diag', max_iter=max_iterations)
    gmm.fit(Combined_MFCC_F)

    joblib.dump(gmm, 'be_mix.pkl')

    [Fs2, x2] = audioBasicIO.read_audio_file(file_2_path)
    if x2 is None:
        raise ValueError(f"Error al cargar el archivo de audio {file_2}")

    if x2.ndim > 1:
        x2 = np.mean(x2, axis=1)

    # Calcular la energía de la señal (validar)
    energy2 = [s**2 for s in x2]
    Voiced_Threshold2 = VTH_Multiplier * np.mean(energy2)

    # Filtrar muestras con voz
    clean_samples2 = []
    for sample_set in range(0, len(x2) - VTH_range, VTH_range):
        sample_set_th = np.mean(energy2[sample_set:sample_set + VTH_range])
        if sample_set_th > Voiced_Threshold2:
            clean_samples2.extend(x2[sample_set:sample_set + VTH_range])

    if len(clean_samples2) == 0:
        raise ValueError(f"No se encontraron muestras con voz en el archivo de audio {file_2}")

    Combined_MFCC_F2 = extract_features(np.array(clean_samples2), Fs2, window_size, overlap)

    check_feature_dimensions(Combined_MFCC_F, Combined_MFCC_F2)

    log_likelihood_2 = gmm.score(Combined_MFCC_F2)

    print(f"Log likelihood del segundo audio comparado con el modelo del primero: {log_likelihood_2}")
