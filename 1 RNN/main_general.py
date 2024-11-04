import matplotlib.pyplot as plt
import seaborn as sns
from modelo_de_rnn import train_model
from funciones_para_predecir import calcula_llr
import numpy as np


def main():
    model, data, id_mapping, history = train_model(
        include_label=True,
        save_path="saved_model/model.keras" 
    )
    #  ID de prueba
    prueba = "CDMN3E1SM007_LEC"
    
    similar_id_index, llr_score, p_h0, p_h1 = calcula_llr(prueba, model, data, id_mapping)
    similar_id = id_mapping[similar_id_index]
    
    print("\nResultados para ID:", prueba)
    print("ID mas similar:", similar_id)

    print(f"\nProbabilidades:")
    print(f"H0 (ID de prueba): {p_h0:.6f}")
    print(f"H1 (Promedio de otros IDs): {p_h1:.6f}")

    print(f"\nLLR: {llr_score:.4f}")


if __name__ == "__main__":
    main()