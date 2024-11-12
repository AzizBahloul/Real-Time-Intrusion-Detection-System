# Real-Time-Intrusion-Detection-System
AI-powered intrusion detection system designed to safeguard Wi-Fi networks by identifying and responding to malicious activities with high accuracy and speed.
Certainly! Here are two structured outlines for your Wi-Fi Intrusion Detection project: the first focuses on data preparation and model training, while the second is for the system architecture to deploy the trained model within a security system.

---
Certainly! Here's a full description for each part of the Wi-Fi Intrusion Detection Model Training Pipeline, including details on each class, functionality, and important considerations.

---

# Wi-Fi Intrusion Detection Model Training Pipeline

The following pipeline is structured to guide the creation, training, evaluation, and deployment of an AI-based Wi-Fi Intrusion Detection System (IDS) using Python. This pipeline includes stages for data preparation, feature engineering, model training, evaluation, deployment, and monitoring.

---

## 1. Data Preparation Pipeline

### 1.1 Data Collection Setup

The `DataCollector` class is designed to handle different types of network traffic data collection, including:
- **Normal Traffic**: Data captured from regular network activity.
- **Attack Traffic**: Data from simulated attack scenarios (e.g., DoS, MitM).
- **Synthetic Data**: Data generated to balance classes or enhance training data volume.

```python
class DataCollector:
    def __init__(self, save_path: str):
        """
        Initialize the DataCollector with a path to save collected data.
        
        :param save_path: Directory to save collected data files.
        """
        self.save_path = save_path
        self.collectors = {
            'normal': self._collect_normal_traffic,
            'attack': self._collect_attack_traffic,
            'synthetic': self._generate_synthetic_data
        }
        
    async def collect_data(self, collection_type: str, duration: int):
        """
        Collect data of a specific type for a given duration.
        
        :param collection_type: Type of data to collect ('normal', 'attack', or 'synthetic').
        :param duration: Duration in seconds for which to collect the data.
        """
        collector = self.collectors.get(collection_type)
        data = await collector(duration)
        self._save_data(data, collection_type)
    
    def _save_data(self, data: pd.DataFrame, collection_type: str):
        """
        Save collected data to the specified directory with a unique filename.
        
        :param data: DataFrame of collected network traffic.
        :param collection_type: Type of collected data (for file naming).
        """
        data.to_csv(f"{self.save_path}/{collection_type}_data.csv", index=False)
```

- **Logging**: Consider adding logging for start, end, and duration of each data collection phase.
- **Saving Format**: Specify the format for saving data (e.g., CSV) to allow compatibility with feature engineering.

### 1.2 Feature Engineering Pipeline

The `FeatureEngineeringPipeline` class is responsible for transforming raw network data into structured features suitable for model input. The pipeline processes features into three main categories:

1. **Time-based Features**: Metrics such as packet rate, burst rate, and inter-arrival time.
2. **Statistical Features**: Descriptive statistics like mean, standard deviation, and quartiles for packet sizes or protocols.
3. **Protocol-specific Features**: Features specific to network protocols that are relevant for identifying attack signatures.

```python
class FeatureEngineeringPipeline:
    def __init__(self):
        """
        Initialize the FeatureEngineeringPipeline with a set of transformers.
        """
        self.transformers = [
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('feature_selector', SelectKBest(k=20))
        ]
        
    def create_features(self, raw_data: pd.DataFrame):
        """
        Transform raw data into features for model training.
        
        :param raw_data: Raw network data as a DataFrame.
        :return: DataFrame of engineered features.
        """
        features = {}
        
        # Time-based features
        features.update(self._create_time_features(raw_data))
        
        # Statistical features
        features.update(self._create_statistical_features(raw_data))
        
        # Protocol-specific features
        features.update(self._create_protocol_features(raw_data))
        
        return pd.DataFrame(features)
        
    def _create_time_features(self, data: pd.DataFrame):
        """
        Generate features related to packet timing, like packet rate.
        
        :param data: Raw network data.
        :return: Dictionary of time-based features.
        """
        return {
            'packet_rate': self._calculate_packet_rate(data),
            'burst_rate': self._calculate_burst_rate(data),
            'inter_arrival_time': self._calculate_iat(data)
        }
```

- **Categorical Feature Handling**: Include methods for encoding protocol or device types.
- **Dimensionality Reduction**: PCA and SelectKBest are applied to reduce the feature set for more efficient model training.

---

## 2. Model Training Structure

### 2.1 Base Model Structure

The `BaseModel` class serves as a foundation for different types of machine learning models, ensuring consistency in building, training, and saving models. This abstract base class enforces a structure that all models must follow.

```python
class BaseModel(ABC):
    def __init__(self, config: dict):
        """
        Initialize a base model with a configuration dictionary.
        
        :param config: Dictionary containing model configuration settings.
        """
        self.config = config
        self.model = None
        self.feature_pipeline = None
        
    @abstractmethod
    def build(self):
        """Abstract method to build the model."""
        pass
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """Abstract method to train the model."""
        pass
        
    def save(self, path: str):
        """
        Save the model, pipeline, and configuration to a specified path.
        
        :param path: Path to save the model file.
        """
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model,
                        'pipeline': self.feature_pipeline,
                        'config': self.config}, f)
```

### 2.2 XGBoost Implementation

The `XGBoostDetector` subclass uses XGBoost, an efficient and scalable model well-suited for tabular data and classification tasks.

```python
class XGBoostDetector(BaseModel):
    def build(self):
        """Initialize and configure the XGBoost model with specified parameters."""
        self.model = xgb.XGBClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            learning_rate=self.config['learning_rate'],
            objective='multi:softprob',
            eval_metric=['merror', 'mlogloss'],
            colsample_bytree=self.config.get('colsample_bytree', 0.8),
            subsample=self.config.get('subsample', 0.8)
        )
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model on training data with early stopping based on validation performance."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=True
        )
```

### 2.3 Deep Learning Implementation (LSTM)

The `LSTMDetector` is a deep learning model suitable for sequence-based data, such as time-series network packets.

```python
class LSTMDetector(BaseModel):
    def build(self):
        """Construct an LSTM model architecture for time-series data."""
        self.model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.config['n_classes'], activation='softmax')
        ])
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the LSTM model with specified hyperparameters."""
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.fit(
            X, y,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5),
                keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    save_best_only=True
                )
            ]
        )
```

---

## 3. Model Evaluation and Selection

### 3.1 Model Evaluation Pipeline

The `ModelEvaluator` evaluates and compares model performance using metrics like accuracy, precision, recall, F1 score, and AUC.

```python
class ModelEvaluator:
    def __init__(self, metrics: List[str]):
        """Initialize the evaluator with a list of metrics to compute."""
        self.metrics = metrics
        self.results = {}
        
    def evaluate_model(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance and return metric scores."""
        predictions = model.predict(X_test)
        
        results = {}
        for metric in self.metrics:
            results[metric] = self._calculate_metric(
                metric, y_test, predictions
            )
            
        self.results[model.__class__.__name__] = results
        return results
        
    def compare_models(self):
        """Return a DataFrame comparison of each model's performance metrics."""
        return pd.DataFrame(self.results).T
```

---

## 4. Model Deployment and Monitoring

### 4.1 Model Deployment Pipeline

The `ModelDeployment` class enables model deployment, including validation and monitoring.

```python
class ModelDeployment:
    def __init__(self, model_path: str):
        """Initialize deployment with model path and monitoring setup."""
        self.model = self.load_model(model

_path)
        self.performance_monitor = PerformanceMonitor()
        
    def deploy(self):
        """Deploy the model, validate, and initiate monitoring."""
        self.validate_model()
        self.backup_current_model()
        self.update_production_model()
        self.start_monitoring()
```

### 4.2 Performance Monitoring

`PerformanceMonitor` tracks deployed model metrics and identifies significant performance drift.

```python
class PerformanceMonitor:
    def __init__(self):
        """Initialize monitoring with historical metrics tracking."""
        self.metrics_history = defaultdict(list)
        
    def track_metrics(self, predictions: np.ndarray, actual: np.ndarray):
        """Track performance metrics over time for deployed model."""
        metrics = self.calculate_metrics(predictions, actual)
        self.update_history(metrics)
        
        if self.should_retrain(metrics):
            self.trigger_retraining()
    
    def should_retrain(self, metrics: dict) -> bool:
        """Evaluate if model retraining is necessary based on performance decay."""
        # Logic for retraining based on threshold drift in metrics
        pass
```

---

Each component ensures that the IDS model can adapt to changes in network traffic patterns and maintain high accuracy in identifying potential intrusions. This pipeline balances flexibility and scalability, allowing it to meet the dynamic needs of Wi-Fi network security.

## 1. Structure for Data Preparation and Model Training Pipeline

### 1.1 Objective
To prepare, train, and validate machine learning and deep learning models that detect potential intrusions in Wi-Fi networks. This structure will emphasize data collection, feature engineering, model training, evaluation, and selection of the best-performing model.

### 1.2 Recommended Kaggle Datasets
For Wi-Fi intrusion detection, here are popular datasets that contain labeled network traffic data (both normal and attack types):
- **NSL-KDD Dataset**: An improved version of the KDD Cup 1999 dataset, commonly used for network intrusion detection research. It includes various attack types like DoS, probing, and remote-to-local.
  - [NSL-KDD on Kaggle](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data)
- **UNSW-NB15 Dataset**: Captures modern network traffic data with realistic attack scenarios. This dataset includes a wider variety of attack types, making it ideal for model generalization.
  - [UNSW-NB15 on Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
- **CICIDS2017 Dataset**: Contains network traffic data with a balance of attack types and normal activity. Includes DoS, brute force, and infiltration attacks.
  - [CICIDS2017 on Kaggle](https://www.kaggle.com/datasets/cicdataset/cicids2017)

### 1.3 Model Training Workflow

#### **Step 1: Data Collection and Preprocessing**
- **Class**: `DataCollector`
  - Use data from Kaggle (e.g., NSL-KDD, UNSW-NB15, or CICIDS2017) and create a labeled dataset with both normal and attack traffic.
  - Clean and preprocess data (e.g., handling missing values, encoding categorical variables).
- **Output**: Preprocessed data ready for feature engineering.

#### **Step 2: Feature Engineering**
- **Class**: `FeatureEngineeringPipeline`
  - Extract time-based, statistical, and protocol-specific features using data transformation techniques.
  - Apply dimensionality reduction (PCA or SelectKBest) to reduce feature space and improve model performance.
- **Output**: Engineered feature set for model input.

#### **Step 3: Model Training**
- **Class**: `BaseModel` and derived classes (`XGBoostDetector`, `LSTMDetector`)
  - Train a range of models, starting with traditional ML algorithms (e.g., XGBoost) and sequence models like LSTM for time-series data.
  - Implement cross-validation and early stopping to optimize models.
- **Output**: Trained models saved with their configurations.

#### **Step 4: Model Evaluation and Selection**
- **Class**: `ModelEvaluator`
  - Evaluate models based on metrics (e.g., accuracy, F1-score, AUC) and compare results.
  - Select the best-performing model for deployment.
- **Output**: Best model saved in a file (e.g., `model.pkl`), along with scalers and transformers (e.g., `scaler.pkl`, `label_encoders.pkl`).

---

## 2. Structure of the Security System for Real-Time Intrusion Detection

### 2.1 Objective
To deploy a real-time intrusion detection system (IDS) using the trained model, continuously monitor its performance, and trigger alerts for any detected intrusions.

### 2.2 System Architecture Components

#### **Component 1: Data Stream Collector**
- **Purpose**: Continuously capture live Wi-Fi traffic data from the network.
- **Technology**: Use packet capture tools such as Scapy or Wireshark’s Python API.
- **Workflow**:
  - Capture network traffic packets.
  - Aggregate data into small time windows (e.g., every 5 seconds) for analysis.
- **Output**: Stream of captured network traffic data for processing.

#### **Component 2: Preprocessing and Feature Engineering**
- **Purpose**: Real-time transformation of raw network data into model-compatible features.
- **Class**: `FeatureEngineeringPipeline` (from training structure)
- **Workflow**:
  - Apply the same transformations used during training (e.g., time-based, statistical features).
  - Use saved scalers and encoders (`scaler.pkl`, `label_encoders.pkl`) to transform data consistently.
- **Output**: Processed data ready for model prediction.

#### **Component 3: Intrusion Detection Model**
- **Purpose**: Run the pre-trained model to detect intrusions in real-time.
- **Model**: Best model from training pipeline (e.g., XGBoost, LSTM)
- **Workflow**:
  - Load model and feature pipeline (from `model.pkl` and other relevant files).
  - Apply the model to predict whether traffic is normal or an attack.
- **Output**: Prediction result (normal or specific attack type).

#### **Component 4: Alert System**
- **Purpose**: Notify security personnel or trigger automated responses in case of detected intrusions.
- **Technology**: Notification services like Twilio (for SMS) or Slack API for real-time alerts.
- **Workflow**:
  - Upon detection of an attack, generate a message including details like time, type of attack, and network IP addresses involved.
  - Send alert to a specified notification channel (e.g., email, SMS).
- **Output**: Alert messages in real-time.

#### **Component 5: Performance Monitoring and Retraining**
- **Class**: `PerformanceMonitor`
- **Purpose**: Track model performance in real-time, log metrics, and initiate retraining if performance drops.
- **Workflow**:
  - Monitor performance metrics (e.g., accuracy, F1-score).
  - If significant performance drift is detected, trigger data collection and retraining of the model.
- **Output**: Continuous performance logs and retrained model if needed.

---

### Summary of Outputs and Files

1. **Data Preparation and Training Pipeline Outputs**:
   - **Feature Set**: `engineered_features.csv`
   - **Trained Model**: `model.pkl`
   - **Pipeline Components**: `scaler.pkl`, `label_encoders.pkl`

2. **Security System Outputs**:
   - **Real-time Predictions**: Stream of predictions (normal or attack).
   - **Alert Notifications**: Alert messages via SMS, email, or other platforms.
   - **Performance Logs**: Continuous monitoring metrics, retraining events.

---



### 1. Data Preparation and Model Training Pipeline Structure

```bash
project_root/
├── data/
│   ├── raw_data/
│   │   ├── nsl_kdd.csv                # NSL-KDD raw data
│   │   ├── unsw_nb15.csv              # UNSW-NB15 raw data
│   │   └── cicids2017.csv             # CICIDS2017 raw data
│   ├── processed_data/
│   │   ├── engineered_features.csv    # Feature-engineered dataset
│   │   └── labels.csv                 # Corresponding labels
│   └── scalers_encoders/
│       ├── scaler.pkl                 # StandardScaler object
│       └── label_encoders.pkl         # Encoders for categorical features
│
├── models/
│   ├── base_model.py                  # Abstract base model class
│   ├── xgboost_detector.py            # XGBoost-based model
│   ├── lstm_detector.py               # LSTM-based deep learning model
│   ├── model_evaluator.py             # Model evaluation and comparison
│   └── trained_models/
│       └── best_model.pkl             # Best trained model (final)
│
├── notebooks/
│   └── data_analysis.ipynb            # Jupyter Notebook for data exploration
│
├── src/
│   ├── data_collection.py             # DataCollector class for data acquisition
│   ├── feature_engineering.py         # FeatureEngineeringPipeline class
│   ├── training_pipeline.py           # Training and evaluation script
│   └── hyperparameter_optimization.py # Hyperparameter search for model
│
└── config/
    ├── model_config.yaml              # Configurations for model hyperparameters
    └── feature_config.yaml            # Feature engineering configurations
```

---

### 2. Security System Deployment Structure

```bash
deployment_root/
├── live_data/
│   ├── incoming_traffic/              # Directory for live network traffic captures
│   └── processed_live_features.csv    # Real-time feature-engineered data for prediction
│
├── src/
│   ├── intrusion_detection.py         # Main script for intrusion detection
│   ├── alert_system.py                # Notification and alert functionality
│   ├── feature_engineering.py         # Real-time feature engineering
│   ├── model_loader.py                # Model loading and inference setup
│   ├── performance_monitor.py         # Performance tracking and monitoring
│   ├── packet_capture.py              # Real-time packet capture module
│   └── retraining_trigger.py          # Script to handle retraining upon drift
│
├── logs/
│   ├── performance_logs.txt           # Log file for performance metrics
│   ├── alerts.log                     # Log file for alerts triggered by intrusions
│   └── retraining_history.log         # Log of retraining events and results
│
├── models/
│   ├── best_model.pkl                 # Trained and validated model
│   ├── scaler.pkl                     # StandardScaler object for real-time data
│   └── label_encoders.pkl             # Encoders for real-time inference
│
└── config/
    ├── deployment_config.yaml         # Deployment-specific configurations
    └── alert_config.yaml              # Alert settings for notifications
```

---

These structures help keep the project organized and modular, separating each function (data processing, model training, and security deployment) for ease of maintenance and scaling. Each directory or file represents a component crucial to either the data preparation/model training pipeline or the real-time security system.




