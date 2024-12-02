import os
import numpy as np
from pyAudioAnalysis import audioBasicIO
import Audio_Feature_Extraction as AFE
from sklearn.mixture import GaussianMixture
import joblib
import pandas as pd
# valida nuevo 

# 
folder_path = "........./9 Audios/"  
audio_file_1 = "Ivana_LEC_T1.wav"  
audio_file_2 = "Andrea_Itzel_HESP.wav"  


# Código ajustado con base en:

# https://github.com/genzen2103/Speaker-Recognition-System-using-GMM
# https://stackoverflow.com/questions/49040208/gaussian-mixture-model-gives-negative-value-scores
# https://stackoverflow.com/questions/48465737/how-to-convert-log-probability-into-simple-probability-between-0-and-1-values-us


log_messages = []

def log_message(message):
    """Agrega un mensaje a la lista de mensajes de registro."""
    log_messages.append(message)

def check_feature_dimensions(features_1, features_2):
    if features_1.shape[1] != features_2.shape[1]:
        raise ValueError(f"Desajuste en el número de características: "
                         f"primer archivo ({features_1.shape[1]} características), "
                         f"segundo archivo ({features_2.shape[1]} características).")

def extract_features(signal, Fs, window_size, overlap):
    MFCC13_F = AFE.stFeatureExtraction(signal, Fs, int(Fs * window_size), int(Fs * overlap))
    delta_MFCC = np.zeros(MFCC13_F.shape)
    for t in range(delta_MFCC.shape[1]):
        index_t_minus_one, index_t_plus_one = t - 1, t + 1
        index_t_minus_one = max(index_t_minus_one, 0)
        index_t_plus_one = min(index_t_plus_one, delta_MFCC.shape[1] - 1)
        delta_MFCC[:, t] = 0.5 * (MFCC13_F[:, index_t_plus_one] - MFCC13_F[:, index_t_minus_one])

    double_delta_MFCC = np.zeros(MFCC13_F.shape)
    for t in range(double_delta_MFCC.shape[1]):
        index_t_minus_one, index_t_plus_one, index_t_plus_two, index_t_minus_two = t - 1, t + 1, t + 2, t - 2
        index_t_minus_one = max(index_t_minus_one, 0)
        index_t_plus_one = min(index_t_plus_one, delta_MFCC.shape[1] - 1)
        index_t_minus_two = max(index_t_minus_two, 0)
        index_t_plus_two = min(index_t_plus_two, double_delta_MFCC.shape[1] - 1)
        double_delta_MFCC[:, t] = 0.1 * (2 * MFCC13_F[:, index_t_plus_two] + MFCC13_F[:, index_t_plus_one]
                                         - MFCC13_F[:, index_t_minus_one] - 2 * MFCC13_F[:, index_t_minus_two])

    Combined_MFCC_F = np.concatenate((MFCC13_F, delta_MFCC, double_delta_MFCC), axis=1)
    return Combined_MFCC_F

def procesar_audios(file_1, file_2, folder_path):
    file_1_path = os.path.join(folder_path, file_1)
    file_2_path = os.path.join(folder_path, file_2)

    [Fs, x] = audioBasicIO.read_audio_file(file_1_path)
    if x is None:
        raise ValueError(f"Error al cargar el archivo de audio {file_1}")
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    window_size = 0.030
    overlap = 0.015
    VTH_Multiplier = 0.05
    VTH_range = 100
    n_mixtures = 16
    max_iterations = 500

    energy = [s**2 for s in x]
    Voiced_Threshold = VTH_Multiplier * np.mean(energy)
    clean_samples = []
    for sample_set in range(0, len(x) - VTH_range, VTH_range):
        sample_set_th = np.mean(energy[sample_set:sample_set + VTH_range])
        if sample_set_th > Voiced_Threshold:
            clean_samples.extend(x[sample_set:sample_set + VTH_range])

    if len(clean_samples) == 0:
        raise ValueError(f"No se encontraron muestras con voz en el archivo de audio {file_1}")

    Combined_MFCC_F = extract_features(np.array(clean_samples), Fs, window_size, overlap)
    #gmm = GaussianMixture(n_components=n_mixtures, covariance_type='diag', max_iter=max_iterations)
    gmm = GaussianMixture(n_components=n_mixtures, covariance_type='diag', max_iter=max_iterations, tol=1e-3)
    gmm.fit(Combined_MFCC_F)

    joblib.dump(gmm, 'be_mix.pkl')

    [Fs2, x2] = audioBasicIO.read_audio_file(file_2_path)
    if x2 is None:
        raise ValueError(f"Error al cargar el archivo de audio {file_2}")
    if x2.ndim > 1:
        x2 = np.mean(x2, axis=1)

    energy2 = [s**2 for s in x2]
    Voiced_Threshold2 = VTH_Multiplier * np.mean(energy2)
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
    probability = 1 / (1 + np.exp(-log_likelihood_2))  # Aplicar función sigmoide para obtener un valor entre 0 y 1

    #log_message(f"LLR (log likelihood): {log_likelihood_2}")
    #log_message(f"Probability (scaled to [0, 1]): {probability}")
    #print(f"Probabilidad: {probability}")
    probability_decimal = f"{probability:.20f}"
    return probability_decimal

procesar_audios(audio_file_1, audio_file_2, folder_path)

'''
log_file_path = os.path.join(folder_path, "LLR_usando_GMM_prueba_2.txt")
with open(log_file_path, 'w') as f:
    for message in log_messages:
        f.write(message + '\n')

print(f"Archivo de registro guardado en: {log_file_path}")

prob = procesar_audios(audio_file_1, audio_file_2, folder_path)
print(f"Probabilidad: {prob}")
'''


#----------------------------------------------------------------------------------------------------------------

reference_audio = "Ivana_LEC_T1.wav"  # Archivo de referencia
comparison_audios = [
    "Ivana_LEC_T1.wav",
    "Ivana_LEC_T2.wav",
    "Dafne_LEC.wav",
    "Dafne_HESP.wav",
    "Andrea_Itzel_LEC_T2.wav",
    "Emiliano_LEC_T2.wav",
    "Emiliano_LEC.wav",
    "Andrea_Itzel_HESP_T2.wav",
    "CDMN1E1SF001_LEC.wav",
    "Emiliano_HESP.wav",
    "Andrea_Itzel_LEC.wav",
    "Andrea_Itzel_HESP.wav",
    "Emiliano_HESP_T2.wav"
]
output_excel = os.path.join(folder_path, "comparaciones_probabilidades.xlsx")
resultados = pd.DataFrame(columns=["Audio 1", "Audio 2", "Probabilidad"])

# Procesar comparaciones
for compare_audio in comparison_audios:
    try:
        probabilidad = procesar_audios(reference_audio, compare_audio, folder_path)
        nuevo_resultado = pd.DataFrame({
            "Audio 1": [reference_audio],
            "Audio 2": [compare_audio],
            "Probabilidad": [probabilidad]
        })
        resultados = pd.concat([resultados, nuevo_resultado], ignore_index=True)
    except Exception as e:
        print(f"Error procesando {compare_audio}: {e}")

# Guardar resultados en el archivo Excel
resultados.to_excel(output_excel, index=False)
print(f"Resultados guardados en {output_excel}")








