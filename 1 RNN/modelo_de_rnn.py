import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os


"""
Datos
Primera prueba
"""

def load_data(include_label=False):

    data = pd.read_csv("data/General_mean_0422.csv")
    
    unique_ids = data['ID'].unique()
    num_individuals = len(unique_ids)
    
    id_mapping = {idx: id_val for idx, id_val in enumerate(unique_ids)}
    
    features = data[['Mean_pitch', 'F1', 'F2', 'F3', 'F4', 'Duration']]
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    
    if include_label:
        features.loc[:, 'Labels'] = pd.factorize(data['Labels'])[0]
    
    labels = pd.Series([list(id_mapping.keys())[list(id_mapping.values()).index(id)] 
                       for id in data['ID']])
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return (X_train, X_test, y_train, y_test), id_mapping, num_individuals

"""
Construcción del modelo
Se guarda en  la carpeta "saved_model" para reusarlo 

"""

def build_model(input_shape, num_individuals):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(num_individuals, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

"""
Entrenar o cargar
Cuando hagamos más, aquí podemos modificar y ocuparlos según necesidades
"""

def train_model(include_label=False, save_path="saved_model/model.keras"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    (X_train, X_test, y_train, y_test), id_mapping, num_individuals = load_data(include_label)
    
    if os.path.exists(save_path):
        print(f"Cargar nuevo modelo '{save_path}'")
        model = tf.keras.models.load_model(save_path)
        history = None
    else:
        print("Entrenando nevo modelo...")
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                save_path,
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3
            )
        ]
        
        # Build and train model
        model = build_model(input_shape=X_train.shape[1], num_individuals=num_individuals)
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"Modelo guardado en: '{save_path}'")
    
    return model, (X_test, y_test), id_mapping, history