import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from typing import List, Tuple, Dict, Any

def load_cgm_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Carica i dati CGM da tutti i file Excel nella cartella data_dir.
    Restituisce un dizionario {patient_id: dataframe}
    """
    data = {}
    for fname in os.listdir(data_dir):
        if fname.endswith('.xlsx'):
            patient_id = os.path.splitext(fname)[0]
            df = pd.read_excel(os.path.join(data_dir, fname))
            data[patient_id] = df
    return data

def load_static_features(static_features_path: str) -> pd.DataFrame:
    """
    Carica le feature statiche dei pazienti da un file Excel/CSV.
    """
    if static_features_path.endswith('.xlsx'):
        return pd.read_excel(static_features_path)
    else:
        return pd.read_csv(static_features_path)

def preprocess_cgm(df: pd.DataFrame, col_time: str, col_glucose: str, max_gap: int = 3) -> pd.DataFrame:
    """
    Gestione dati mancanti e ordinamento temporale.
    max_gap: massimo numero di valori consecutivi imputabili (gap piccolo)
    """
    df = df.sort_values(col_time).reset_index(drop=True)
    # Identifica gap
    mask = df[col_glucose].isna()
    # Imputazione solo per gap piccoli
    df[col_glucose] = df[col_glucose].interpolate(method='linear', limit=max_gap, limit_direction='both')
    return df

def create_windows(
    series: np.ndarray, 
    window_size: int, 
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea finestre temporali (input/output) dalla serie CGM.
    """
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size+horizon-1])
    return np.array(X), np.array(y)

def scale_data(train: np.ndarray, val: np.ndarray, test: np.ndarray, scaler_type: str = 'zscore'):
    """
    Normalizza/standardizza i dati CGM.
    """
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(train.reshape(-1, 1))
    train_scaled = scaler.transform(train.reshape(-1, 1)).reshape(train.shape)
    val_scaled = scaler.transform(val.reshape(-1, 1)).reshape(val.shape)
    test_scaled = scaler.transform(test.reshape(-1, 1)).reshape(test.shape)
    return train_scaled, val_scaled, test_scaled, scaler

def encode_static_features(df: pd.DataFrame, categorical: List[str], numerical: List[str], scaler_type: str = 'zscore'):
    """
    Esegue encoding e scaling delle feature statiche.
    """
    encoders = {}
    scalers = {}
    df_encoded = df.copy()
    for col in categorical:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le
    for col in numerical:
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        df_encoded[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    return df_encoded, encoders, scalers

def split_patients(patients: List[str], val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Suddivide i pazienti in train/val/test (cross-paziente).
    """
    np.random.seed(seed)
    patients_array = np.array(patients)
    np.random.shuffle(patients_array)
    n = len(patients_array)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val_patients = patients_array[:n_val]
    test_patients = patients_array[n_val:n_val+n_test]
    train_patients = patients_array[n_val+n_test:]
    return list(train_patients), list(val_patients), list(test_patients)

def prepare_dataset_for_training(cgm_data_dict, static_features_df, window_size=12, horizon=6, 
                                train_patients=None, val_patients=None, test_patients=None,
                                categorical_cols=['Gender', 'Race'], 
                                numerical_cols=['Age', 'Hemoglobin A1C'],
                                scaler_type='zscore'):
    """
    Prepara i dataset completi per training, validation e test.
    Combina CGM e feature statiche seguendo la scheda tecnica.
    
    Returns:
        dict con train/val/test splits contenenti X_seq, X_static, y, scalers, encoders
    """
    
    # 1. Encoding delle feature statiche
    static_encoded, encoders, scalers = encode_static_features(
        static_features_df, categorical_cols, numerical_cols, scaler_type
    )
    
    # 2. Creazione finestre per ogni paziente
    all_windows = {'train': [], 'val': [], 'test': []}
    all_static = {'train': [], 'val': [], 'test': []}
    all_targets = {'train': [], 'val': [], 'test': []}
    
    for patient_id, cgm_df in cgm_data_dict.items():
        # Determina il set di appartenenza
        if patient_id in train_patients:
            split = 'train'
        elif patient_id in val_patients:
            split = 'val'
        elif patient_id in test_patients:
            split = 'test'
        else:
            continue
        
        # Preprocessing CGM per questo paziente
        cgm_processed = preprocess_cgm(cgm_df, 'date', 'glucose', max_gap=3)
        glucose_values = np.array(cgm_processed['glucose'].dropna().values)
        
        if len(glucose_values) < window_size + horizon:
            print(f"Paziente {patient_id} ha troppo pochi dati CGM, skip")
            continue
        
        # Crea finestre temporali
        X_windows, y_windows = create_windows(glucose_values, window_size, horizon)
        
        # Feature statiche per questo paziente
        patient_static = static_encoded[static_encoded['SubjectID'] == patient_id]
        if patient_static.empty:
            print(f"Feature statiche mancanti per {patient_id}, skip")
            continue
        
        # Prendi le feature statiche (escluso SubjectID)
        static_features = patient_static.drop('SubjectID', axis=1).values[0]
        
        # Replica le feature statiche per ogni finestra
        static_repeated = np.tile(static_features, (len(X_windows), 1))
        
        all_windows[split].append(X_windows)
        all_static[split].append(static_repeated)
        all_targets[split].append(y_windows)
    
    # 3. Concatena tutti i dati per split
    dataset = {}
    for split in ['train', 'val', 'test']:
        if all_windows[split]:
            X_seq = np.vstack(all_windows[split])
            X_static = np.vstack(all_static[split])
            y = np.concatenate(all_targets[split])
            
            dataset[split] = {
                'X_seq': X_seq,
                'X_static': X_static,
                'y': y
            }
        else:
            dataset[split] = {'X_seq': None, 'X_static': None, 'y': None}
    
    # 4. Scaling dei dati CGM
    if dataset['train']['X_seq'] is not None:
        train_seq = dataset['train']['X_seq']
        val_seq = dataset['val']['X_seq'] if dataset['val']['X_seq'] is not None else np.array([])
        test_seq = dataset['test']['X_seq'] if dataset['test']['X_seq'] is not None else np.array([])
        
        # Scaling
        train_scaled, val_scaled, test_scaled, cgm_scaler = scale_data(
            train_seq, val_seq, test_seq, scaler_type
        )
        
        # Scaling targets
        y_train = dataset['train']['y']
        y_val = dataset['val']['y'] if dataset['val']['y'] is not None else np.array([])
        y_test = dataset['test']['y'] if dataset['test']['y'] is not None else np.array([])
        
        y_train_scaled, y_val_scaled, y_test_scaled, y_scaler = scale_data(
            y_train, y_val, y_test, scaler_type
        )
        
        # Aggiorna dataset con dati scalati
        dataset['train'].update({
            'X_seq': train_scaled.reshape(-1, window_size, 1),
            'y': y_train_scaled
        })
        
        if dataset['val']['X_seq'] is not None:
            dataset['val'].update({
                'X_seq': val_scaled.reshape(-1, window_size, 1),
                'y': y_val_scaled
            })
        
        if dataset['test']['X_seq'] is not None:
            dataset['test'].update({
                'X_seq': test_scaled.reshape(-1, window_size, 1),
                'y': y_test_scaled
            })
        
        # Aggiungi scalers
        dataset['scalers'] = {
            'cgm_scaler': cgm_scaler,
            'y_scaler': y_scaler,
            'static_encoders': encoders,
            'static_scalers': scalers
        }
    
    return dataset

def load_static_features_from_excel(file_path, subject_col='SubjectID'):
    """
    Carica feature statiche da Excel e standardizza formato SubjectID
    """
    df = pd.read_excel(file_path)
    
    # Standardizza SubjectID
    if 'Subject' in df.columns and subject_col not in df.columns:
        df['SubjectID'] = 'subject' + df['Subject'].astype(str)
    elif subject_col in df.columns:
        df['SubjectID'] = 'subject' + df[subject_col].astype(str)
    
    return df

def inverse_transform_predictions(predictions, scaler):
    """
    Converte le predizioni dalla scala normalizzata ai valori originali
    """
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

def create_patient_level_data(cgm_data_dict, static_features_df, window_size=12, horizon=6):
    """
    Crea dataset mantenendo la struttura per-paziente (per analisi XAI)
    """
    patient_data = {}
    
    for patient_id, cgm_df in cgm_data_dict.items():
        # Preprocessing CGM
        cgm_processed = preprocess_cgm(cgm_df, 'date', 'glucose', max_gap=3)
        glucose_values = np.array(cgm_processed['glucose'].dropna().values)
        
        if len(glucose_values) < window_size + horizon:
            continue
        
        # Crea finestre
        X_windows, y_windows = create_windows(glucose_values, window_size, horizon)
        
        # Feature statiche
        patient_static = static_features_df[static_features_df['SubjectID'] == patient_id]
        if patient_static.empty:
            continue
        
        patient_data[patient_id] = {
            'X_seq': X_windows,
            'y': y_windows,
            'static_features': patient_static.drop('SubjectID', axis=1).values[0],
            'timestamps': cgm_processed['date'].values[:len(X_windows)]
        }
    
    return patient_data