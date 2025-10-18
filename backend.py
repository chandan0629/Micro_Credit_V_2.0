from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json
from datetime import datetime
from typing import Dict, Any, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class MultiModelCreditAssessment:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_performances = {}
        self.feature_names = ['total_transactions', 'avg_transaction_amount', 
                            'payment_consistency_score', 'business_age_months', 
                            'digital_footprint_score']
        self.trained = False
        
    def load_and_prepare_data(self, csv_path='credit_risk_data.csv'):
        """Load and prepare the credit risk dataset"""
        try:
            # Load the dataset
            data = pd.read_csv(csv_path)
            logger.info(f"Loaded dataset with {len(data)} records")
            
            # Prepare features
            X = data[self.feature_names].values
            
            # Create realistic labels based on multiple criteria
            # Lower risk for: high payment score, established business, good digital presence
            risk_labels = []
            for _, row in data.iterrows():
                risk_score = 0
                
                # Payment consistency factor (40% weight)
                if row['payment_consistency_score'] < 70:
                    risk_score += 40
                elif row['payment_consistency_score'] < 85:
                    risk_score += 20
                
                # Business age factor (25% weight)
                if row['business_age_months'] < 12:
                    risk_score += 25
                elif row['business_age_months'] < 36:
                    risk_score += 15
                
                # Transaction volume factor (20% weight)
                if row['total_transactions'] < 5000:
                    risk_score += 20
                elif row['total_transactions'] < 15000:
                    risk_score += 10
                
                # Digital footprint factor (15% weight)
                if row['digital_footprint_score'] < 60:
                    risk_score += 15
                elif row['digital_footprint_score'] < 80:
                    risk_score += 8
                
                # 1 = High Risk, 0 = Low Risk
                risk_labels.append(1 if risk_score > 50 else 0)
            
            y = np.array(risk_labels)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
            logger.info(f"Risk distribution - High Risk: {sum(y)}, Low Risk: {len(y) - sum(y)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def build_neural_network(self, input_shape):
        """Build the ensemble neural network"""
        input_layer = tf.keras.layers.Input(shape=(input_shape,))
        
        # First neural network branch
        x1 = tf.keras.layers.Dense(64, activation='relu')(input_layer)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.Dropout(0.3)(x1)
        x1 = tf.keras.layers.Dense(32, activation='relu')(x1)
        output1 = tf.keras.layers.Dense(1, activation='sigmoid')(x1)
        
        # Second neural network branch
        x2 = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x2 = tf.keras.layers.Dropout(0.4)(x2)
        x2 = tf.keras.layers.Dense(64, activation='relu')(x2)
        x2 = tf.keras.layers.Dropout(0.3)(x2)
        output2 = tf.keras.layers.Dense(1, activation='sigmoid')(x2)
        
        # Ensemble output
        ensemble_output = tf.keras.layers.Average()([output1, output2])
        model = tf.keras.Model(inputs=input_layer, outputs=ensemble_output)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train_all_models(self, csv_path='credit_risk_data.csv'):
        """Train all models and compare their performance"""
        try:
            # Load data
            X_train, X_test, y_train, y_test = self.load_and_prepare_data(csv_path)
            
            # Initialize scalers for each model
            self.scalers = {
                'neural_network': StandardScaler(),
                'random_forest': StandardScaler(),
                'gradient_boost': StandardScaler(),
                'logistic_regression': StandardScaler(),
                'svm': StandardScaler()
            }
            
            # Scale features
            X_train_scaled = {}
            X_test_scaled = {}
            
            for model_name, scaler in self.scalers.items():
                X_train_scaled[model_name] = scaler.fit_transform(X_train)
                X_test_scaled[model_name] = scaler.transform(X_test)
            
            # 1. Neural Network (Ensemble)
            logger.info("Training Neural Network...")
            nn_model = self.build_neural_network(X_train.shape[1])
            
            # Add early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = nn_model.fit(
                X_train_scaled['neural_network'], y_train,
                validation_data=(X_test_scaled['neural_network'], y_test),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.models['neural_network'] = nn_model
            nn_pred = (nn_model.predict(X_test_scaled['neural_network']) > 0.5).astype(int)
            self.model_performances['neural_network'] = {
                'accuracy': accuracy_score(y_test, nn_pred),
                'model_type': 'Deep Learning Ensemble'
            }
            
            # 2. Random Forest
            logger.info("Training Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(X_train_scaled['random_forest'], y_train)
            self.models['random_forest'] = rf_model
            
            rf_pred = rf_model.predict(X_test_scaled['random_forest'])
            self.model_performances['random_forest'] = {
                'accuracy': accuracy_score(y_test, rf_pred),
                'model_type': 'Ensemble Tree-based'
            }
            
            # 3. Gradient Boosting
            logger.info("Training Gradient Boosting...")
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            gb_model.fit(X_train_scaled['gradient_boost'], y_train)
            self.models['gradient_boost'] = gb_model
            
            gb_pred = gb_model.predict(X_test_scaled['gradient_boost'])
            self.model_performances['gradient_boost'] = {
                'accuracy': accuracy_score(y_test, gb_pred),
                'model_type': 'Gradient Boosting'
            }
            
            # 4. Logistic Regression
            logger.info("Training Logistic Regression...")
            lr_model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
            lr_model.fit(X_train_scaled['logistic_regression'], y_train)
            self.models['logistic_regression'] = lr_model
            
            lr_pred = lr_model.predict(X_test_scaled['logistic_regression'])
            self.model_performances['logistic_regression'] = {
                'accuracy': accuracy_score(y_test, lr_pred),
                'model_type': 'Linear Classification'
            }
            
            # 5. Support Vector Machine
            logger.info("Training SVM...")
            svm_model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
            svm_model.fit(X_train_scaled['svm'], y_train)
            self.models['svm'] = svm_model
            
            svm_pred = svm_model.predict(X_test_scaled['svm'])
            self.model_performances['svm'] = {
                'accuracy': accuracy_score(y_test, svm_pred),
                'model_type': 'Support Vector Machine'
            }
            
            self.trained = True
            logger.info("All models trained successfully!")
            
            # Log performance summary
            for model_name, performance in self.model_performances.items():
                logger.info(f"{model_name}: {performance['accuracy']:.4f}")
                
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def predict_with_all_models(self, financial_data: Dict[str, Any]):
        """Make predictions using all trained models"""
        if not self.trained:
            raise ValueError("Models not trained yet!")
        
        # Prepare features
        features = np.array([[
            financial_data.get('total_transactions', 0),
            financial_data.get('avg_transaction_amount', 0),
            financial_data.get('payment_consistency_score', 0),
            financial_data.get('business_age_months', 0),
            financial_data.get('digital_footprint_score', 0)
        ]])
        
        predictions = {}
        
        for model_name, model in self.models.items():
            # Scale features for this model
            features_scaled = self.scalers[model_name].transform(features)
            
            if model_name == 'neural_network':
                # Neural network returns probability
                risk_prob = model.predict(features_scaled)[0][0]
            else:
                # Other models - get probability
                risk_prob = model.predict_proba(features_scaled)[0][1]
            
            # Generate recommendation
            recommendation = self.generate_loan_recommendation(risk_prob)
            
            predictions[model_name] = {
                'risk_score': float(risk_prob),
                'risk_percentage': round(risk_prob * 100, 2),
                'recommendation': recommendation,
                'model_info': self.model_performances[model_name]
            }
        
        return predictions
    
    def generate_loan_recommendation(self, risk_score: float) -> Dict[str, Any]:
        """Generate loan recommendation based on risk score"""
        if risk_score < 0.3:
            return {
                'max_loan_amount': 50000,
                'interest_rate': 5.5,
                'risk_category': 'Low Risk',
                'approval_probability': 95,
                'recommended_term_months': 24
            }
        elif risk_score < 0.6:
            return {
                'max_loan_amount': 25000,
                'interest_rate': 8.5,
                'risk_category': 'Medium Risk',
                'approval_probability': 75,
                'recommended_term_months': 18
            }
        else:
            return {
                'max_loan_amount': 10000,
                'interest_rate': 12.5,
                'risk_category': 'High Risk',
                'approval_probability': 45,
                'recommended_term_months': 12
            }
    
    def get_ensemble_prediction(self, predictions: Dict):
        """Get ensemble prediction by averaging all models"""
        risk_scores = [pred['risk_score'] for pred in predictions.values()]
        ensemble_risk_score = np.mean(risk_scores)
        
        return {
            'risk_score': float(ensemble_risk_score),
            'risk_percentage': round(ensemble_risk_score * 100, 2),
            'recommendation': self.generate_loan_recommendation(ensemble_risk_score),
            'model_info': {
                'accuracy': np.mean([pred['model_info']['accuracy'] for pred in predictions.values()]),
                'model_type': 'Ensemble of All Models'
            }
        }

# Initialize the global model instance
credit_model = MultiModelCreditAssessment()

# Global variable to store assessment history
assessment_history = []

@app.route('/api/train', methods=['POST'])
def train_models():
    """Endpoint to train all models"""
    try:
        # Check if CSV file exists
        csv_path = 'credit_risk_data.csv'
        if not os.path.exists(csv_path):
            return jsonify({
                'success': False,
                'error': 'Dataset file not found. Please upload credit_risk_data.csv'
            }), 400
        
        credit_model.train_all_models(csv_path)
        
        return jsonify({
            'success': True,
            'message': 'All models trained successfully',
            'model_performances': credit_model.model_performances
        })
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/assess', methods=['POST'])
def assess_credit_risk():
    """Main endpoint for credit risk assessment"""
    try:
        if not credit_model.trained:
            return jsonify({
                'success': False,
                'error': 'Models not trained. Please train models first.'
            }), 400
        
        # Get financial data from request
        financial_data = request.get_json()
        
        # Validate required fields
        required_fields = ['total_transactions', 'avg_transaction_amount', 
                         'payment_consistency_score', 'business_age_months', 
                         'digital_footprint_score']
        
        for field in required_fields:
            if field not in financial_data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Get predictions from all models
        predictions = credit_model.predict_with_all_models(financial_data)
        
        # Get ensemble prediction
        ensemble_prediction = credit_model.get_ensemble_prediction(predictions)
        
        # Add ensemble to predictions
        predictions['ensemble'] = ensemble_prediction
        
        # Store in history
        assessment_record = {
            'timestamp': datetime.now().isoformat(),
            'input_data': financial_data,
            'predictions': predictions
        }
        assessment_history.append(assessment_record)
        
        # Keep only last 100 records
        if len(assessment_history) > 100:
            assessment_history.pop(0)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'ensemble': ensemble_prediction,
            'model_count': len(predictions) - 1,  # Exclude ensemble from count
            'assessment_id': len(assessment_history)
        })
        
    except Exception as e:
        logger.error(f"Assessment error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/performance', methods=['GET'])
def get_model_performance():
    """Get performance metrics for all models"""
    try:
        if not credit_model.trained:
            return jsonify({
                'success': False,
                'error': 'Models not trained yet'
            }), 400
        
        return jsonify({
            'success': True,
            'performances': credit_model.model_performances
        })
        
    except Exception as e:
        logger.error(f"Performance fetch error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history', methods=['GET'])
def get_assessment_history():
    """Get assessment history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        
        # Return last 'limit' assessments
        recent_history = assessment_history[-limit:] if assessment_history else []
        
        return jsonify({
            'success': True,
            'history': recent_history,
            'total_assessments': len(assessment_history)
        })
        
    except Exception as e:
        logger.error(f"History fetch error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export/csv', methods=['GET'])
def export_assessments_csv():
    """Export assessment history as CSV"""
    try:
        if not assessment_history:
            return jsonify({
                'success': False,
                'error': 'No assessment history to export'
            }), 400
        
        # Prepare data for CSV export
        export_data = []
        for record in assessment_history:
            row = {
                'timestamp': record['timestamp'],
                **record['input_data']
            }
            
            # Add predictions from each model
            for model_name, prediction in record['predictions'].items():
                row[f'{model_name}_risk_score'] = prediction['risk_score']
                row[f'{model_name}_max_loan'] = prediction['recommendation']['max_loan_amount']
                row[f'{model_name}_interest_rate'] = prediction['recommendation']['interest_rate']
                row[f'{model_name}_risk_category'] = prediction['recommendation']['risk_category']
            
            export_data.append(row)
        
        # Create DataFrame and convert to CSV
        df = pd.DataFrame(export_data)
        csv_data = df.to_csv(index=False)
        
        return jsonify({
            'success': True,
            'csv_data': csv_data,
            'filename': f'credit_assessments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        })
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    return jsonify({
        'success': True,
        'status': 'running',
        'models_trained': credit_model.trained,
        'total_assessments': len(assessment_history),
        'available_models': list(credit_model.models.keys()) if credit_model.trained else [],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting Credit Risk Assessment API Server...")
    logger.info("Available endpoints:")
    logger.info("POST /api/train - Train all models")
    logger.info("POST /api/assess - Assess credit risk")
    logger.info("GET /api/models/performance - Get model performance")
    logger.info("GET /api/history - Get assessment history")
    logger.info("GET /api/export/csv - Export assessments as CSV")
    logger.info("GET /api/status - Get system status")
    
    # Auto-train models if CSV exists
    if os.path.exists('credit_risk_data.csv'):
        logger.info("Found credit_risk_data.csv, auto-training models...")
        try:
            credit_model.train_all_models()
            logger.info("Auto-training completed successfully!")
        except Exception as e:
            logger.error(f"Auto-training failed: {str(e)}")
    
    # Allow configuring port and debug mode via environment variables so
    # the server can be started on a different port (for example 5001)
    port = int(os.environ.get('PORT', '5000'))
    debug_env = os.environ.get('FLASK_DEBUG', 'True').lower()
    debug = debug_env in ('1', 'true', 'yes')

    app.run(debug=debug, host='0.0.0.0', port=port)
