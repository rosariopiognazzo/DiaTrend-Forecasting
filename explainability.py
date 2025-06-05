import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

class XAIFramework:
    """
    Framework per l'explainability del modello di forecasting glicemico.
    Integra SHAP per spiegare l'impatto delle sequenze CGM e feature statiche.
    """
    
    def __init__(self, model, dataset, feature_names=None):
        """
        Inizializza il framework XAI.
        
        Args:
            model: Modello Keras/TensorFlow allenato
            dataset: Dataset con train/val/test splits
            feature_names: Nomi delle feature statiche
        """
        self.model = model
        self.dataset = dataset
        self.feature_names = feature_names or [f"static_feat_{i}" for i in range(dataset['train']['X_static'].shape[1])]
        self.explainer = None
        self.shap_values = None
        
    def setup_shap_explainer(self, background_size=100, explainer_type='deep'):
        """
        Configura l'explainer SHAP.
        
        Args:
            background_size: Numero di campioni per il background dataset
            explainer_type: 'deep', 'kernel' o 'gradient'
        """
        # Seleziona campioni random per il background
        idx = np.random.choice(len(self.dataset['train']['X_seq']), 
                              min(background_size, len(self.dataset['train']['X_seq'])), 
                              replace=False)
        
        background_seq = self.dataset['train']['X_seq'][idx]
        background_static = self.dataset['train']['X_static'][idx]
        
        if explainer_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, [background_seq, background_static])
        elif explainer_type == 'gradient':
            self.explainer = shap.GradientExplainer(self.model, [background_seq, background_static])
        else:  # kernel
            def model_predict(inputs):
                seq_data, static_data = inputs[:, :self.dataset['train']['X_seq'].shape[1]*self.dataset['train']['X_seq'].shape[2]], inputs[:, self.dataset['train']['X_seq'].shape[1]*self.dataset['train']['X_seq'].shape[2]:]
                seq_reshaped = seq_data.reshape(-1, self.dataset['train']['X_seq'].shape[1], self.dataset['train']['X_seq'].shape[2])
                return self.model.predict([seq_reshaped, static_data])
            
            # Flatten dei dati per KernelExplainer
            combined_background = np.hstack([
                background_seq.reshape(background_seq.shape[0], -1),
                background_static
            ])
            self.explainer = shap.KernelExplainer(model_predict, combined_background)
        
        print(f"SHAP {explainer_type} explainer configurato con {background_size} campioni di background")
    
    def calculate_shap_values(self, test_indices=None, max_samples=50):
        """
        Calcola i valori SHAP per il test set.
        
        Args:
            test_indices: Indici specifici del test set da analizzare
            max_samples: Numero massimo di campioni da analizzare
        """
        if self.explainer is None:
            raise ValueError("Configura prima l'explainer con setup_shap_explainer()")
        
        # Seleziona campioni del test set
        if test_indices is None:
            n_samples = min(max_samples, len(self.dataset['test']['X_seq']))
            test_indices = np.random.choice(len(self.dataset['test']['X_seq']), n_samples, replace=False)
        
        test_seq = self.dataset['test']['X_seq'][test_indices]
        test_static = self.dataset['test']['X_static'][test_indices]
        
        print(f"Calcolando valori SHAP per {len(test_indices)} campioni...")
        
        if isinstance(self.explainer, (shap.DeepExplainer, shap.GradientExplainer)):
            self.shap_values = self.explainer.shap_values([test_seq, test_static])
        else:  # KernelExplainer
            combined_test = np.hstack([
                test_seq.reshape(test_seq.shape[0], -1),
                test_static
            ])
            shap_vals = self.explainer.shap_values(combined_test)
            
            # Separa i valori SHAP per sequenza e features statiche
            seq_size = test_seq.shape[1] * test_seq.shape[2]
            self.shap_values = [
                shap_vals[:, :seq_size].reshape(test_seq.shape),
                shap_vals[:, seq_size:]
            ]
        
        self.test_indices = test_indices
        print("Valori SHAP calcolati!")
        
        return self.shap_values
    
    def plot_global_importance(self, top_k=20):
        """
        Visualizza l'importanza globale delle feature.
        """
        if self.shap_values is None:
            raise ValueError("Calcola prima i valori SHAP con calculate_shap_values()")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Importanza delle feature statiche
        static_shap = self.shap_values[1]
        static_importance = np.mean(np.abs(static_shap), axis=0)
        
        axes[0].barh(self.feature_names, static_importance)
        axes[0].set_title('Importanza Globale Feature Statiche')
        axes[0].set_xlabel('|Valore SHAP Medio|')
        
        # 2. Importanza temporale della sequenza CGM
        seq_shap = self.shap_values[0]
        seq_importance = np.mean(np.abs(seq_shap), axis=(0, 2))  # Media su samples e features
        timesteps = [f"t-{len(seq_importance)-i}" for i in range(len(seq_importance))]
        
        axes[1].plot(timesteps, seq_importance, marker='o')
        axes[1].set_title('Importanza Temporale Sequenza CGM')
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('|Valore SHAP Medio|')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_sample_explanation(self, sample_idx=0):
        """
        Visualizza la spiegazione per un singolo campione.
        """
        if self.shap_values is None:
            raise ValueError("Calcola prima i valori SHAP")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Dati del campione
        seq_data = self.dataset['test']['X_seq'][self.test_indices[sample_idx]]
        static_data = self.dataset['test']['X_static'][self.test_indices[sample_idx]]
        true_value = self.dataset['test']['y'][self.test_indices[sample_idx]]
        pred_value = self.model.predict([seq_data.reshape(1, -1, 1), static_data.reshape(1, -1)])[0][0]
        
        # 1. Serie temporale CGM con importanza SHAP
        seq_shap_sample = np.array(self.shap_values[0])[sample_idx, :, 0]
        timesteps = list(range(len(seq_data)))
        
        axes[0, 0].plot(timesteps, seq_data.flatten(), 'b-', label='Valori CGM', linewidth=2)
        axes[0, 0].set_title(f'Serie CGM - Campione {sample_idx}')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Valore CGM')
        axes[0, 0].legend()
        
        # 2. Importanza SHAP per timestep
        colors = ['red' if x > 0 else 'blue' for x in seq_shap_sample]
        axes[0, 1].bar(timesteps, seq_shap_sample, color=colors, alpha=0.7)
        axes[0, 1].set_title('Importanza SHAP per Timestep')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Valore SHAP')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. Feature statiche con importanza SHAP
        static_shap_sample = self.shap_values[1][sample_idx]
        colors_static = ['red' if x > 0 else 'blue' for x in static_shap_sample]
        
        axes[1, 0].barh(self.feature_names, static_data, alpha=0.7)
        axes[1, 0].set_title('Valori Feature Statiche')
        axes[1, 0].set_xlabel('Valore')
        
        axes[1, 1].barh(self.feature_names, static_shap_sample, color=colors_static, alpha=0.7)
        axes[1, 1].set_title('Importanza SHAP Feature Statiche')
        axes[1, 1].set_xlabel('Valore SHAP')
        axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.suptitle(f'Spiegazione Campione {sample_idx}\nVero: {true_value:.2f}, Predetto: {pred_value:.2f}', 
                     fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_interactions(self):
        """
        Analizza le interazioni tra feature statiche e pattern CGM.
        """
        if self.shap_values is None:
            raise ValueError("Calcola prima i valori SHAP")
        
        # Raggruppa i campioni per valori delle feature statiche
        static_data = self.dataset['test']['X_static'][self.test_indices]
        seq_shap = self.shap_values[0]
        static_shap = self.shap_values[1]
        
        # Analisi per feature categoriali (es. Gender)
        for feat_idx, feat_name in enumerate(self.feature_names):
            if feat_name in ['Gender', 'Race']:  # Feature categoriali
                unique_values = np.unique(static_data[:, feat_idx])
                
                plt.figure(figsize=(12, 6))
                
                for i, value in enumerate(unique_values):
                    mask = static_data[:, feat_idx] == value
                    if np.sum(mask) > 0:
                        # Importanza media della sequenza per questo gruppo
                        group_seq_importance = np.mean(np.abs(seq_shap[mask]), axis=(0, 2))
                        timesteps = [f"t-{len(group_seq_importance)-j}" for j in range(len(group_seq_importance))]
                        
                        plt.plot(timesteps, group_seq_importance, 
                                label=f'{feat_name}={value} (n={np.sum(mask)})', 
                                marker='o')
                
                plt.title(f'Importanza Sequenza CGM per {feat_name}')
                plt.xlabel('Timestep')
                plt.ylabel('|Valore SHAP Medio|')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
    
    def generate_explanation_summary(self, patient_id=None):
        """
        Genera un summary testuale delle spiegazioni.
        """
        if self.shap_values is None:
            raise ValueError("Calcola prima i valori SHAP")
        
        # Importanza globale feature statiche
        static_importance = np.mean(np.abs(self.shap_values[1]), axis=0)
        top_static_features = np.argsort(static_importance)[-3:][::-1]
        
        # Importanza temporale
        seq_importance = np.mean(np.abs(self.shap_values[0]), axis=(0, 2))
        most_important_timesteps = np.argsort(seq_importance)[-3:][::-1]
        
        summary = f"""
        === SUMMARY EXPLAINABILITY ===
        
        TOP 3 FEATURE STATICHE PIÙ INFLUENTI:
        """
        
        for i, feat_idx in enumerate(top_static_features):
            feat_name = self.feature_names[feat_idx]
            importance = static_importance[feat_idx]
            summary += f"{i+1}. {feat_name}: {importance:.4f}\n        "
        
        summary += f"""
        
        TOP 3 TIMESTEPS CGM PIÙ INFLUENTI:
        """
        
        for i, timestep in enumerate(most_important_timesteps):
            importance = seq_importance[timestep]
            summary += f"{i+1}. t-{len(seq_importance)-timestep}: {importance:.4f}\n        "
        
        summary += f"""
        
        PATTERN IDENTIFICATI:
        - Le feature statiche contribuiscono in media per {np.mean(static_importance):.4f}
        - I timesteps più recenti (ultimi 5) hanno importanza media: {np.mean(seq_importance[-5:]):.4f}
        - I timesteps più lontani (primi 5) hanno importanza media: {np.mean(seq_importance[:5]):.4f}
        """
        
        print(summary)
        return summary

def create_clinical_explanation(model, X_seq, X_static, feature_names, 
                               threshold_low=70, threshold_high=180):
    """
    Crea spiegazioni orientate al dominio clinico.
    """
    prediction = model.predict([X_seq, X_static])[0][0]
    
    # Analisi della tendenza nella sequenza CGM
    recent_trend = np.mean(X_seq[0, -3:, 0]) - np.mean(X_seq[0, -6:-3, 0])
    
    explanation = f"Predizione: {prediction:.1f} mg/dL\n\n"
    
    if prediction < threshold_low:
        explanation += "⚠️ RISCHIO IPOGLICEMIA\n"
    elif prediction > threshold_high:
        explanation += "⚠️ RISCHIO IPERGLICEMIA\n"
    else:
        explanation += "✅ RANGE NORMALE\n"
    
    explanation += f"\nTendenza recente: "
    if recent_trend > 5:
        explanation += "📈 In aumento"
    elif recent_trend < -5:
        explanation += "📉 In diminuzione"
    else:
        explanation += "➡️ Stabile"
    
    explanation += f" ({recent_trend:+.1f} mg/dL)\n"
    
    # Analisi feature statiche (esempio)
    if len(X_static[0]) > 0:
        explanation += f"\nFattori del paziente:\n"
        for i, value in enumerate(X_static[0]):
            if i < len(feature_names):
                explanation += f"- {feature_names[i]}: {value:.2f}\n"
    
    return explanation

# Funzioni di utilità per la visualizzazione
def plot_prediction_vs_actual(y_true, y_pred, title="Predizioni vs Valori Reali"):
    """
    Plot scatter delle predizioni vs valori reali.
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Linea perfetta
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predizione Perfetta')
    
    # Zone cliniche
    plt.axhspan(0, 70, alpha=0.2, color='red', label='Ipoglicemia (<70)')
    plt.axhspan(70, 180, alpha=0.2, color='green', label='Range Normale (70-180)')
    plt.axhspan(180, max_val, alpha=0.2, color='orange', label='Iperglicemia (>180)')
    
    plt.xlabel('Valori Reali (mg/dL)')
    plt.ylabel('Predizioni (mg/dL)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calcola R²
    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_time_series_with_predictions(timestamps, y_true, y_pred, title="Serie Temporale con Predizioni"):
    """
    Plot della serie temporale con predizioni.
    """
    plt.figure(figsize=(15, 6))
    
    plt.plot(timestamps, y_true, 'b-', label='Valori Reali', linewidth=2)
    plt.plot(timestamps, y_pred, 'r--', label='Predizioni', linewidth=2)
    
    # Zone cliniche
    plt.axhspan(0, 70, alpha=0.2, color='red', label='Ipoglicemia')
    plt.axhspan(70, 180, alpha=0.2, color='green', label='Range Normale')
    plt.axhspan(180, 400, alpha=0.2, color='orange', label='Iperglicemia')
    
    plt.xlabel('Tempo')
    plt.ylabel('Glicemia (mg/dL)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
