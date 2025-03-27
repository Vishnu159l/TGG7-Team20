import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import scipy.signal as signal
import scipy.stats
import plotly.express as px
import plotly.graph_objs as go

class AdvancedVisualizationTechniques:
    @staticmethod
    def risk_level_sankey_diagram(neural_network_predictions, random_forest_predictions):
        """Create a Sankey diagram showing model prediction alignments"""
        alignment_counts = {
            (0, 0): np.sum((neural_network_predictions == 0) & (random_forest_predictions == 0)),
            (0, 1): np.sum((neural_network_predictions == 0) & (random_forest_predictions == 1)),
            (0, 2): np.sum((neural_network_predictions == 0) & (random_forest_predictions == 2)),
            (1, 0): np.sum((neural_network_predictions == 1) & (random_forest_predictions == 0)),
            (1, 1): np.sum((neural_network_predictions == 1) & (random_forest_predictions == 1)),
            (1, 2): np.sum((neural_network_predictions == 1) & (random_forest_predictions == 2)),
            (2, 0): np.sum((neural_network_predictions == 2) & (random_forest_predictions == 0)),
            (2, 1): np.sum((neural_network_predictions == 2) & (random_forest_predictions == 1)),
            (2, 2): np.sum((neural_network_predictions == 2) & (random_forest_predictions == 2))
        }
        
        nodes = [
            'NN Low', 'NN Moderate', 'NN High',
            'RF Low', 'RF Moderate', 'RF High'
        ]
        
        links = {
            'source': [0, 0, 0, 1, 1, 1, 2, 2, 2],
            'target': [3, 4, 5, 3, 4, 5, 3, 4, 5],
            'value': [
                alignment_counts.get((0, 0), 0),
                alignment_counts.get((0, 1), 0),
                alignment_counts.get((0, 2), 0),
                alignment_counts.get((1, 0), 0),
                alignment_counts.get((1, 1), 0),
                alignment_counts.get((1, 2), 0),
                alignment_counts.get((2, 0), 0),
                alignment_counts.get((2, 1), 0),
                alignment_counts.get((2, 2), 0)
            ]
        }
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=nodes
            ),
            link=dict(
                source=links['source'],
                target=links['target'],
                value=links['value']
            )
        )])
        
        fig.update_layout(title_text='Model Prediction Alignment', font_size=10)
        return fig

    @staticmethod
    def multi_dimensional_vulnerability_heatmap(attack_characteristics):
        """Create a multi-dimensional heatmap of vulnerability characteristics"""
        df = pd.DataFrame(attack_characteristics)
        features = ['signal_complexity', 'leakage_potential', 'anomaly_count']
        df_normalized = (df[features] - df[features].min()) / (df[features].max() - df[features].min())
        
        fig = px.imshow(
            df_normalized.T, 
            labels=dict(x='Trace Index', y='Vulnerability Metric', color='Normalized Value'),
            title='Multi-Dimensional Vulnerability Heatmap',
            color_continuous_scale='YlOrRd'
        )
        
        return fig

    @staticmethod
    def probabilistic_risk_distribution(neural_network_probabilities):
        """Create a probabilistic risk distribution visualization"""
        prob_df = pd.DataFrame(neural_network_probabilities, columns=['Low', 'Moderate', 'High'])
        
        fig = go.Figure()
        for column in prob_df.columns:
            fig.add_trace(go.Box(
                y=prob_df[column],
                name=column,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title='Probabilistic Risk Level Distributions',
            yaxis_title='Probability',
            xaxis_title='Risk Levels'
        )
        
        return fig

class AdvancedSideChannelAnalyzer:
    def __init__(self, num_samples=3000, trace_length=250):
        """
        Advanced Side-Channel Vulnerability Analyzer
        
        Enhanced with more comprehensive vulnerability modeling
        
        Args:
        - num_samples: Number of synthetic traces to generate
        - trace_length: Length of each trace
        """
        self.num_samples = num_samples
        self.trace_length = trace_length
        
        # Enhanced data storage
        self.traces = None
        self.labels = None
        self.attack_characteristics = None
        self.preprocessed_traces = None
        
        # Models
        self.neural_network_model = None
        self.random_forest_model = None
        
        # Enhanced analysis results
        self.analysis_report = {}
        
        # Attack-specific vulnerability weights
        self.attack_vulnerability_matrix = {
            'power': {
                'base_risk_multiplier': 1.2,
                'key_leakage_sensitivity': 0.8,
                'frequency_vulnerability': 0.7
            },
            'timing': {
                'base_risk_multiplier': 1.0,
                'execution_variance_sensitivity': 0.9,
                'algorithmic_complexity_impact': 0.6
            },
            'electromagnetic': {
                'base_risk_multiplier': 1.5,
                'signal_leakage_sensitivity': 0.9,
                'emission_pattern_complexity': 0.8
            }
        }

    def generate_synthetic_data(self, attack_type='multi'):
        """
        Generate advanced synthetic side-channel traces with multi-dimensional vulnerability indicators
        
        Args:
        - attack_type: Type of attack scenario ('multi', 'power', 'timing', 'electromagnetic')
        
        Returns:
        - traces: Synthetic power traces
        - labels: Corresponding vulnerability labels
        """
        np.random.seed(42)
        
        # Generate base traces with advanced complexity
        traces = np.random.randn(self.num_samples, self.trace_length)
        labels = np.zeros(self.num_samples, dtype=int)
        attack_characteristics = []
        
        # Enhanced vulnerability generation
        trace_sums = np.sum(traces, axis=1)
        low_threshold = np.percentile(trace_sums, 60)
        high_threshold = np.percentile(trace_sums, 90)
        
        # Multi-dimensional risk scoring
        labels[trace_sums > low_threshold] = 1  # Moderate risk
        labels[trace_sums > high_threshold] = 2  # High risk
        
        # Attack-type specific trace generation
        for i in range(self.num_samples):
            attack_char = {
                'signal_complexity': 0,
                'leakage_potential': 0,
                'anomaly_count': 0
            }
            
            # Base trace modifications
            if labels[i] == 2:
                # High-risk trace enhancements
                traces[i] += np.sin(np.linspace(0, 20, self.trace_length)) * 3
                traces[i] += np.random.normal(0, 1.5, self.trace_length)
                
                # Significant anomalies
                spike_indices = np.random.choice(self.trace_length, 5, replace=False)
                traces[i, spike_indices] += np.random.uniform(5, 10, 5)
                attack_char['anomaly_count'] = len(spike_indices)
            
            # Attack-type specific modifications
            if attack_type in ['multi', 'power']:
                # Power analysis specific variations
                power_variation = np.sin(np.linspace(0, np.pi, self.trace_length)) + 1
                traces[i] *= power_variation
                traces[i] += np.random.normal(0, 0.5, self.trace_length)
                attack_char['signal_complexity'] += np.mean(np.abs(power_variation))
            
            if attack_type in ['multi', 'timing']:
                # Timing-based variation
                timing_variation = np.linspace(0.8, 1.2, self.trace_length)
                traces[i] *= timing_variation
                attack_char['leakage_potential'] += np.std(timing_variation)
            
            if attack_type in ['multi', 'electromagnetic']:
                # Electromagnetic emission simulation
                chirp_signal = signal.chirp(
                    np.linspace(0, 1, self.trace_length), 
                    0.1, 1, labels[i] * 10
                )
                traces[i] += chirp_signal
                attack_char['signal_complexity'] += np.mean(np.abs(chirp_signal))
            
            attack_characteristics.append(attack_char)
        
        self.traces = traces
        self.labels = labels
        self.attack_characteristics = attack_characteristics
        
        return traces, labels, attack_characteristics

    def preprocess_data(self, test_size=0.2):
        """
        Advanced preprocessing with comprehensive feature extraction
        
        Returns:
        - Preprocessed train and test splits
        """
        # Standardization
        scaler = StandardScaler()
        traces_scaled = scaler.fit_transform(self.traces)
        
        def extract_advanced_features(traces):
            features = []
            for trace in traces:
                # Comprehensive statistical features
                trace_features = [
                    np.mean(trace),
                    np.std(trace),
                    np.max(trace),
                    np.min(trace),
                    np.percentile(trace, 25),
                    np.percentile(trace, 75),
                    # Advanced spectral features
                    np.mean(np.abs(np.fft.fft(trace)[:len(trace)//2])),
                    np.max(np.abs(np.fft.fft(trace)[:len(trace)//2])),
                    # Non-linear features
                    np.mean(np.diff(trace)**2),
                    scipy.stats.skew(trace),
                    scipy.stats.kurtosis(trace)
                ]
                features.append(trace_features)
            
            return np.array(features)
        
        # Extract advanced features
        traces_features = extract_advanced_features(traces_scaled)
        
        # Stratified split with advanced sampling
        X_train, X_test, y_train, y_test = train_test_split(
            traces_features, self.labels, 
            test_size=test_size, 
            random_state=42,
            stratify=self.labels
        )
        
        self.preprocessed_traces = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        return X_train, X_test, y_train, y_test

    def create_neural_network_model(self, input_shape):
        """
        Advanced neural network architecture for vulnerability detection
        """
        # Dynamic class weight computation
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(self.preprocessed_traces['y_train']), 
            y=self.preprocessed_traces['y_train']
        )
        class_weights = dict(enumerate(class_weights))
        
        # Enhanced weight balancing
        class_weights[1] *= 1.5  # Boost moderate risk class
        class_weights[2] *= 2.0  # Significantly boost high-risk class
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        return model, class_weights

    def train_models(self):
        """
        Comprehensive model training with advanced evaluation
        """
        # Ensure data is preprocessed
        if self.preprocessed_traces is None:
            self.generate_synthetic_data()
            self.preprocess_data()
        
        X_train, X_test, y_train, y_test = (
            self.preprocessed_traces['X_train'],
            self.preprocessed_traces['X_test'],
            self.preprocessed_traces['y_train'],
            self.preprocessed_traces['y_test']
        )
        
        # Neural Network Training
        self.neural_network_model, class_weights = self.create_neural_network_model((X_train.shape[1],))
        nn_history = self.neural_network_model.fit(
            X_train, y_train, 
            validation_split=0.2,
            epochs=100, 
            batch_size=64, 
            verbose=0,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=15, 
                    restore_best_weights=True
                )
            ]
        )
        
        # Random Forest Training
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        
        self.random_forest_model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42,
            class_weight=class_weights,
            min_samples_leaf=2,
            max_depth=10
        )
        self.random_forest_model.fit(X_train, y_train)
        
        # Comprehensive Evaluation
        nn_predictions = np.argmax(self.neural_network_model.predict(X_test), axis=1)
        rf_predictions = self.random_forest_model.predict(X_test)
        
        self.analysis_report = {
            'Neural Network': {
                'Accuracy': self.neural_network_model.evaluate(X_test, y_test)[1],
                'Classification Report': classification_report(y_test, nn_predictions, output_dict=True)
            },
            'Random Forest': {
                'Accuracy': self.random_forest_model.score(X_test, y_test),
                'Classification Report': classification_report(y_test, rf_predictions, output_dict=True)
            }
        }
        
        return self.analysis_report, X_test, y_test

class AdvancedSideChannelAttackSimulator:
    def __init__(self, trained_analyzer):
        """
        Advanced Attack Simulation with Comprehensive Risk Assessment
        
        Args:
        - trained_analyzer: AdvancedSideChannelAnalyzer with trained models
        """
        self.analyzer = trained_analyzer
    
    def simulate_advanced_attack(self, attack_type='multi', test_data=None, test_labels=None):
        """
        Comprehensive Attack Simulation with Multi-Dimensional Risk Scoring
        
        Args:
        - attack_type: Specific attack type to simulate
        - test_data: Optional pre-existing test data
        - test_labels: Optional pre-existing test labels
        
        Returns:
        - Detailed attack risk assessment
        """
        # Generate attack-specific traces if no test data provided
        if test_data is None or test_labels is None:
            attack_traces, attack_labels, attack_characteristics = self.analyzer.generate_synthetic_data(attack_type)
        else:
            attack_traces = test_data
            attack_labels = test_labels
            attack_characteristics = self.analyzer.attack_characteristics
        
        # Feature extraction
        scaler = StandardScaler()
        attack_traces_scaled = scaler.fit_transform(attack_traces)
        
        def extract_advanced_features(traces):
            features = []
            for trace in traces:
                trace_features = [
                    np.mean(trace),
                    np.std(trace),
                    np.max(trace),
                    np.min(trace),
                    np.percentile(trace, 25),
                    np.percentile(trace, 75),
                    np.mean(np.abs(np.fft.fft(trace)[:len(trace)//2])),
                    np.max(np.abs(np.fft.fft(trace)[:len(trace)//2])),
                    np.mean(np.diff(trace)**2),
                    scipy.stats.skew(trace),
                    scipy.stats.kurtosis(trace)
                ]
                features.append(trace_features)
            
            return np.array(features)
        
        attack_features = extract_advanced_features(attack_traces_scaled)
        
        # Multi-model predictions
        nn_probs = self.analyzer.neural_network_model.predict(attack_features)
        nn_predictions = np.argmax(nn_probs, axis=1)
        rf_predictions = self.analyzer.random_forest_model.predict(attack_features)
        
        # Advanced Risk Scoring
        risk_scoring = self.compute_comprehensive_risk(
            attack_type, 
            attack_characteristics, 
            nn_predictions, 
            rf_predictions
        )
        
        # Comprehensive Attack Report
        attack_report = {
            'Attack Type': attack_type.capitalize(),
            'Total Traces': len(attack_traces),
            'Neural Network': {
                'Predictions': nn_predictions,
                'Probabilities': nn_probs,
                'Ground Truth Labels': attack_labels,
                'Classification Report': classification_report(attack_labels, nn_predictions, output_dict=True)
            },
            'Random Forest': {
                'Predictions': rf_predictions,
                'Classification Report': classification_report(attack_labels, rf_predictions, output_dict=True)
            },
            'Risk Assessment': risk_scoring,
            'Attack Characteristics': attack_characteristics
        }
        
        return attack_report
    
    def compute_comprehensive_risk(self, attack_type, attack_characteristics, nn_predictions, rf_predictions):
        """
        Multi-dimensional risk computation
        """
        # Attack-type specific vulnerability matrix
        attack_config = self.analyzer.attack_vulnerability_matrix.get(attack_type, {})
        
        # Risk computation variables
        risk_levels = {0: 'Low', 1: 'Moderate', 2: 'High'}
        risk_breakdown = {level: 0 for level in risk_levels.keys()}
        risk_scores = []
        
        for i, (char, nn_pred, rf_pred) in enumerate(zip(attack_characteristics, nn_predictions, rf_predictions)):
            # Base risk computation
            base_risk = max(nn_pred, rf_pred)
            
            # Enhanced risk scoring
            risk_score = base_risk * attack_config.get('base_risk_multiplier', 1.0)
            
            # Incorporate trace characteristics
            risk_score *= (1 + char['signal_complexity'] * 0.5)
            risk_score *= (1 + char['leakage_potential'] * 0.3)
            risk_score *= (1 + char['anomaly_count'] * 0.2)
            
            risk_scores.append(risk_score)
            risk_breakdown[base_risk] += 1
        
        # Overall risk metrics
        risk_assessment = {
            'Risk Levels': risk_breakdown,
            'Average Risk Score': np.mean(risk_scores),
            'Max Risk Score': np.max(risk_scores),
            'Risk Score Distribution': {
                level: f"{(count/len(attack_characteristics))*100:.2f}%" 
                for level, count in risk_breakdown.items()
            }
        }
        
        return risk_assessment

def enhance_side_channel_visualizations(analysis_report, attack_report):
    """
    Generate comprehensive visualizations for side-channel vulnerability analysis
    
    Args:
    - analysis_report: Model training analysis report
    - attack_report: Attack simulation report
    """
    st.header("🎨 Advanced Vulnerability Visualization")
    
    # Model Performance Visualization Tabs
    tab1, tab2, tab3 = st.tabs([
        "Model Prediction Alignment", 
        "Vulnerability Heatmap", 
        "Risk Probability Distribution"
    ])
    
    with tab1:
        # Sankey Diagram for Prediction Alignment
        sankey_fig = AdvancedVisualizationTechniques.risk_level_sankey_diagram(
            attack_report['Neural Network']['Predictions'],
            attack_report['Random Forest']['Predictions']
        )
        st.plotly_chart(sankey_fig, use_container_width=True)
    
    with tab2:
        # Multi-dimensional Vulnerability Heatmap
        heatmap_fig = AdvancedVisualizationTechniques.multi_dimensional_vulnerability_heatmap(
            attack_report.get('Attack Characteristics', [])
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with tab3:
        # Probabilistic Risk Distribution
        prob_dist_fig = AdvancedVisualizationTechniques.probabilistic_risk_distribution(
            attack_report['Neural Network']['Probabilities']
        )
        st.plotly_chart(prob_dist_fig, use_container_width=True)

def main():
    st.set_page_config(page_title="🔒 GLITCHCON: Side-Channel Vulnerability Analyzer", page_icon="🔒")
    
    st.title("🔒 Advanced Side-Channel Vulnerability Analyzer")
    st.markdown("AI-Powered Cryptographic Vulnerability Detection")
    
    # Session state initialization
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    # Sidebar Configuration
    st.sidebar.header("🛠 Analysis Configuration")
    num_samples = st.sidebar.slider("Number of Synthetic Traces", 1000, 10000, 3000)
    trace_length = st.sidebar.slider("Trace Length", 100, 500, 250)
    
    # Vulnerability Risk Description
    risk_descriptions = {
        0: "✅ Low Risk: Minimal Side-Channel Leakage",
        1: "⚠ Moderate Risk: Potential Information Exposure", 
        2: "🚨 High Critical Risk: Significant Vulnerability"
    }
    
    # Model Training Section
    st.header("🧠 Vulnerability Detection Model Training")
    
    train_clicked = st.button("🚀 Train Advanced Vulnerability Models")
    
    if train_clicked or st.session_state.analyzer is None:
        with st.spinner("Training Advanced Vulnerability Models..."):
            # Initialize and train analyzer
            st.session_state.analyzer = AdvancedSideChannelAnalyzer(
                num_samples=num_samples,
                trace_length=trace_length
            )
            
            # Train models
            analysis_report, X_test, y_test = st.session_state.analyzer.train_models()
            
            # Performance Visualization
            st.subheader("📊 Model Performance Metrics")
            for model_name, report in analysis_report.items():
                st.write(f"{model_name} Model Performance:")
                st.write(f"  Accuracy: {report['Accuracy']*100:.2f}%")
                
                # Detailed Classification Report
                df = pd.DataFrame(report['Classification Report']).transpose()
                st.dataframe(df)
    
    # Attack Simulation Section
    st.header("🚨 Advanced Attack Simulation")
    
    if st.session_state.analyzer is None:
        st.warning("❗ Train vulnerability models first!")
    else:
        attack_types = st.multiselect(
            "Select Attack Scenarios", 
            ['Multi-Vector', 'Power', 'Timing', 'Electromagnetic'], 
            default=['Multi-Vector']
        )
        
        if st.button("🔍 Simulate Attack Scenarios"):
            for attack_type in attack_types:
                st.subheader(f"🔬 {attack_type} Attack Simulation")
                
                # Initialize Advanced Attack Simulator
                attack_simulator = AdvancedSideChannelAttackSimulator(st.session_state.analyzer)
                
                try:
                    # Simulate Advanced Attack 
                    # Pass test data and labels from trained model for consistent evaluation
                    attack_report = attack_simulator.simulate_advanced_attack(
                        attack_type.lower().replace('-vector', ''),
                        st.session_state.analyzer.preprocessed_traces['X_test'],
                        st.session_state.analyzer.preprocessed_traces['y_test']
                    )
                    
                    # Comprehensive Risk Assessment
                    st.write("### 📈 Vulnerability Risk Assessment")
                    
                    # Risk Level Distribution
                    st.write("#### Risk Level Distribution:")
                    risk_levels = attack_report['Risk Assessment']['Risk Levels']
                    risk_distribution = attack_report['Risk Assessment']['Risk Score Distribution']
                    
                    for level, percentage in risk_distribution.items():
                        st.write(f"{risk_descriptions[level]}: {percentage} of traces")
                    
                    # Detailed Risk Metrics
                    st.write("\n#### Comprehensive Risk Metrics:")
                    st.write(f"Average Risk Score: {attack_report['Risk Assessment']['Average Risk Score']:.2f}")
                    st.write(f"Maximum Risk Score: {attack_report['Risk Assessment']['Max Risk Score']:.2f}")
                    
                    with open("C:\\Users\\Akshay Prakash\\Documents\\Semester 4\\Hackathon\\tmp\\result.txt", 'w') as f:
                        f.write(f"{attack_report['Risk Assessment']['Average Risk Score']:.2f}")
                        f.write(" ")
                        f.write(", ".join(attack_types))
                    # Enhanced Visualizations
                    enhance_side_channel_visualizations(
                        st.session_state.analyzer.analysis_report, 
                        attack_report
                    )
                
                except Exception as e:
                    st.error(f"❌ Attack Simulation Error: {e}")

if __name__ == "__main__":
    main()