# DiaTrend-Forecasting Framework

## 🎯 Obiettivo

Questo framework sviluppa e confronta modelli di deep learning per il **forecasting della glicemia** utilizzando dati CGM (Continuous Glucose Monitoring) dal dataset DiaTrend. L'obiettivo non è solo raggiungere la massima accuratezza predittiva, ma anche fornire **spiegabilità clinicamente rilevante** per supportare decisioni mediche informate.

## 🔬 Approccio Metodologico

Il framework implementa un confronto sistematico tra diverse architetture di deep learning, valutando sia:
- **Performance predittiva**: Accuratezza, MAE, RMSE, metriche cliniche
- **Interpretabilità**: Meccanismi di explainability per comprendere le decisioni del modello

### Modelli Implementati

1. **LSTM/GRU Stacked**: Reti ricorrenti ottimizzate per serie temporali
2. **Transformer Encoder**: Architettura attention-based per pattern temporali complessi
3. **Modelli Ibridi**: Combinazioni di architetture (future implementazioni)

### Meccanismi di Explainability

1. **SHAP (SHapley Additive exPlanations)**: Model-agnostic, applicabile a tutti i modelli
2. **Attention Weights**: Intrinseci ai Transformer, mostrano pattern temporali
3. **Feature Importance**: Analisi dell'impatto delle feature statiche del paziente

## 📁 Struttura del Framework

### 🏗️ Core Models

#### `stacked_RNNs.py`
**Funzione principale**: `make_stacked_RNNs()`

Implementa reti ricorrenti ottimizzate per forecasting:
- **LSTM/GRU stacked** con configurazione ottimale per time series
- **Ottimizzazioni cuDNN**: `dropout=0.0`, `unroll=False` per performance GPU
- **Architettura modulare**: Supporto per bidirectional e multi-layer
- **Output multi-step**: Predizione diretta di `forecast_horizon` timesteps

```python
# Esempio utilizzo
model = make_stacked_RNNs(
    input_shape=(12, 1),      # 12 timesteps, 1 feature (glucose)
    forecast_horizon=6,       # Predici prossimi 6 timesteps
    type_model='LSTM',        # 'LSTM' o 'GRU'
    num_layers=3,            # Numero layer stacked
    hidden_units=64,         # Unità per layer
    bidirectional=True,      # Processamento bidirezionale
    dropout=0.0              # Ottimizzazione cuDNN
)
```

#### `ForecastEncoder.py`
**Classe principale**: `TransformerForecaster`

Transformer encoder ottimizzato per forecasting con interpretabilità:
- **TimeSeriesEmbedding**: CNN 1D con skip connections per embedding temporali
- **Multi-Head Attention**: Con salvataggio pesi per interpretabilità
- **Aggregazione intelligente**: Diversi metodi (attention, pooling, last)
- **Interpretabilità nativa**: `get_attention_maps()`, `interpret_prediction()`

```python
# Esempio utilizzo
transformer = TransformerForecaster(
    num_layers=3,
    d_model=64,
    num_heads=8,
    dff=256,
    input_features=1,
    forecast_horizon=6,
    aggregation='attention'   # Aggregazione learnable
)

# Estrai attention maps per interpretabilità
attention_maps = transformer.get_attention_maps(x)
interpretation = transformer.interpret_prediction(x)
```

### 🔍 Explainability Framework

#### `explainability.py`
**Classe principale**: `XAIFramework`

Framework unificato per spiegabilità model-agnostic:
- **SHAP Integration**: Support per DeepExplainer, GradientExplainer, KernelExplainer
- **Analisi multi-livello**: Feature statiche + pattern temporali CGM
- **Visualizzazioni cliniche**: Orientate al dominio medico
- **Analisi comparativa**: Confronto interpretabilità tra modelli

```python
# Setup framework XAI
xai = XAIFramework(model, dataset, feature_names)
xai.setup_shap_explainer(explainer_type='gradient')

# Calcola e visualizza spiegazioni
shap_values = xai.calculate_shap_values(max_samples=50)
xai.plot_global_importance()
xai.plot_sample_explanation(sample_idx=0)
```

**Funzionalità chiave**:
- `plot_global_importance()`: Importanza media features e timesteps
- `plot_sample_explanation()`: Spiegazione dettagliata per singolo campione
- `analyze_feature_interactions()`: Interazioni feature statiche ↔ pattern CGM
- `generate_explanation_summary()`: Report testuale automatico

### 📊 Main Framework

#### `framework_main.ipynb`
Notebook principale che orchestr tutto il processo:

1. **Data Loading & Preprocessing**: Caricamento dataset DiaTrend
2. **Model Training**: Training sistematico di tutti i modelli
3. **Performance Evaluation**: Metriche predittive e cliniche
4. **Explainability Analysis**: Analisi SHAP e attention per tutti i modelli
5. **Comparative Results**: Confronto performance vs interpretabilità

**Pipeline completa**:
- Setup dataset con split train/val/test
- Training modelli con hyperparameter optimization
- Valutazione metriche cliniche (time in range, hypoglycemia risk)
- Analisi explainability con SHAP
- Generazione report comparativo

### 🧪 Utilities & Legacy

#### `Trasformers.py`
**Status**: Legacy - Transformer encoder+decoder per text generation
- Progettato originariamente per NLP
- Non ottimizzato per forecasting time series
- Sostituito da `ForecastEncoder.py`

#### `test_framework.py`
Utility per testing rapido dei modelli durante sviluppo.

#### `funcs.py`
Funzioni di utilità per preprocessing e analisi dati.

## 🚀 Quick Start

### 1. Setup Ambiente
```bash
pip install tensorflow pandas numpy matplotlib seaborn shap
```

### 2. Training Modelli
```python
# Nel notebook framework_main.ipynb
from stacked_RNNs import make_stacked_RNNs
from ForecastEncoder import TransformerForecaster
from explainability import XAIFramework

# Definisci modelli da confrontare
models = {
    'LSTM': make_stacked_RNNs(...),
    'Transformer': TransformerForecaster(...)
}

# Training e valutazione
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    # Evaluate
    predictions = model.predict(X_test)
    
    # Explainability
    xai = XAIFramework(model, dataset)
    shap_values = xai.calculate_shap_values()
    xai.plot_global_importance()
```

### 3. Analisi Risultati
Il framework genera automaticamente:
- **Performance Report**: Metriche quantitative per ogni modello
- **Explainability Dashboard**: Visualizzazioni SHAP e attention
- **Clinical Insights**: Analisi orientata al dominio medico
- **Model Comparison**: Confronto sistematico accuratezza vs interpretabilità

## 📈 Metriche di Valutazione

### Performance Predittiva
- **MAE/RMSE**: Errori standard
- **Time in Range (TIR)**: % predizioni in range 70-180 mg/dL
- **Hypoglycemia Detection**: Sensibilità rilevamento <70 mg/dL
- **Clarke Error Grid**: Analisi errori clinicamente significativi

### Interpretabilità
- **Feature Importance**: Ranking feature più influenti
- **Temporal Patterns**: Quali timesteps sono più importanti
- **Attention Analysis**: Pattern appresi dai Transformer
- **Clinical Relevance**: Coerenza con conoscenza medica

## 🔮 Roadmap

### Versione Corrente (v1.0)
- [x] LSTM/GRU ottimizzati con cuDNN
- [x] Transformer encoder per forecasting
- [x] Framework SHAP completo
- [x] Attention-based interpretability
- [x] Pipeline training automatizzata

### Prossime Features (v1.1)
- [ ] Ensemble methods (stacking, voting)
- [ ] Hyperparameter optimization automatico
- [ ] Real-time forecasting pipeline
- [ ] Clinical validation metrics
- [ ] Model deployment tools

### Future Enhancements (v2.0)
- [ ] Multi-modal inputs (insulin, meals, exercise)
- [ ] Personalized models per paziente
- [ ] Federated learning capabilities
- [ ] Mobile deployment
- [ ] Clinical trial integration

## 📝 Output del Framework

Al completamento, il framework produce:

1. **Model Performance Report**
   - Tabella comparativa accuratezza
   - Grafici performance temporali
   - Analisi errori per range glicemici

2. **Explainability Dashboard**
   - Heatmap attention weights (Transformer)
   - SHAP importance plots (tutti i modelli)
   - Feature interaction analysis

3. **Clinical Insights Report**
   - Raccomandazioni modello ottimale
   - Analisi interpretabilità clinica
   - Risk assessment capabilities

## 🎓 Contributi Scientifici

Questo framework contribuisce alla ricerca in:
- **Explainable AI in Healthcare**: Metodologie per spiegabilità in ambito medico
- **Time Series Forecasting**: Ottimizzazioni architetturali per dati CGM
- **Clinical Decision Support**: Strumenti interpretativi per medici
- **Comparative ML**: Framework sistematico per confronto modelli

---

*Framework sviluppato per ricerca accademica in Explainable AI e Medical Forecasting*
