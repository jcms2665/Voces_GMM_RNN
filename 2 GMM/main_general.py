
import os
from funciones import procesar_audios

os.chdir("G:/Mi unidad/1 Proyectos/12 Acústica ejercicios/5 GMM")


# Uso de la función principal
audio_file_1 = "Ivana_LEC_T2.wav"

audio_file_2 = "Ivana_LEC_T1.wav"


folder_path = "G:/Mi unidad/1 Proyectos/12 Acústica ejercicios/2 Código/Código/Comparación/Audios"

# Procesa los archivos de audio
procesar_audios(audio_file_1, audio_file_2, folder_path)



