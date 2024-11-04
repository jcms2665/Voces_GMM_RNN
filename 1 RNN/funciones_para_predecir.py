'''
Funciones para Redes Neuronales
JC

V-1: 24-10-2024
V-2: 02-11-2024

'''


import numpy as np


"""
Función del github de ruslanmv
Calcula la probabilidad de similitud para un ID especifico
"""

def mas_parecido(id_index, model, data, all_data=None):

    X_test, _ = data
    probabilities = model.predict(X_test, verbose=0)
    # Validar si es bueno traerlo así 
    if all_data is not None:
        mask = all_data['ID'] == id_index
        if mask.any():
            specific_probs = probabilities[mask]
            return np.mean(specific_probs[:, id_index])
    
    return probabilities[id_index][id_index]


"""
Calcula el Logaritmo de la Razon de Verosimilitud y encuentra el ID mas similar
Regresa el indice_id_similar, puntaje_llr, p_h0, p_h1
Validar con Fer/Sofía

"""

def calcula_llr(test_id, model, data, id_mapping):

    test_index = (list(id_mapping.keys())[list(id_mapping.values()).index(test_id)] 
                 if isinstance(test_id, str) else test_id)
    
    X_test, _ = data
    probabilities = model.predict(X_test, verbose=0)
    
    p_h0 = mas_parecido(test_index, model, data)
    print(f"Probabilidad de similitud para el ID cuestionado (p_h0): {p_h0}")
    
    other_probs = []
    for i in range(len(id_mapping)):
        if i != test_index:
            prob = mas_parecido(i, model, data)
            other_probs.append(prob)
    
    p_h1 = np.mean(other_probs) if other_probs else 0
    print(f"Probabilidad promedio de similitud para otros IDs (p_h1): {p_h1}")
    
    llr = np.log(p_h0 / p_h1) if p_h1 > 0 else float('inf')
    
    similar_id_index = np.argmax([mas_parecido(i, model, data) 
                                 for i in range(len(id_mapping))])
    
    return similar_id_index, llr, p_h0, p_h1