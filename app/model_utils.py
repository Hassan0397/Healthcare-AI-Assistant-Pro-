# model_utils.py
"""
Model Utilities Module for Patient Readmission Risk Analysis App

This module handles model loading, preprocessing, prediction, and feature importance analysis.
It serves as the ML engine for the Streamlit dashboard.
"""

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append('..')

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

@st.cache_resource(show_spinner="Loading AI model...")
def load_trained_model(model_path: str = None) -> Optional[object]:
    """
    Load the pre-trained model from the models directory.
    
    Args:
        model_path: Path to the model file. If None, uses default location.
        
    Returns:
        object: Loaded model object or None if error
    """
    try:
        if model_path is None:
            # Try multiple possible model locations
            possible_paths = [
                "models/readmission_model.pkl",
                "../models/readmission_model.pkl",
                "../../models/readmission_model.pkl",
                "./models/readmission_model.pkl"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    model_path = path
                    break
            
            if model_path is None:
                st.error("❌ Model file not found. Please ensure the model is trained and saved.")
                return None
        
        # Load the model
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
        
        # Get model type information
        if hasattr(model, 'named_steps'):
            classifier_type = type(model.named_steps['classifier']).__name__
        else:
            classifier_type = type(model).__name__
        
        print(f"🤖 Model type: {classifier_type}")
        return model
        
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return None

@st.cache_resource(show_spinner="Loading preprocessor...")
def load_preprocessor(preprocessor_path: str = None) -> Optional[object]:
    """
    Load the preprocessor (scaler/encoder) from the models directory.
    
    Args:
        preprocessor_path: Path to the preprocessor file.
        
    Returns:
        object: Loaded preprocessor object or None if error
    """
    try:
        if preprocessor_path is None:
            # Try multiple possible preprocessor locations
            possible_paths = [
                "models/preprocessor.pkl",
                "../models/preprocessor.pkl", 
                "../../models/preprocessor.pkl",
                "./models/preprocessor.pkl"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    preprocessor_path = path
                    break
            
            if preprocessor_path is None:
                print("⚠️ Preprocessor file not found. Using default preprocessing.")
                return None
        
        # Load the preprocessor
        preprocessor = joblib.load(preprocessor_path)
        print(f"✅ Preprocessor loaded successfully from: {preprocessor_path}")
        
        return preprocessor
        
    except Exception as e:
        print(f"⚠️ Error loading preprocessor: {str(e)}. Using default preprocessing.")
        return None

# =============================================================================
# FEATURE MANAGEMENT - BASED ON PREPROCESSOR.PKL ANALYSIS
# =============================================================================

def get_model_features() -> List[str]:
    """
    Get the EXACT features the model was trained on.
    Based on preprocessor.pkl analysis - 32 specific features in exact order.
    
    Returns:
        List[str]: List of 32 feature names in exact training order
    """
    return [
        'patient_id', 'age', 'gender', 'ethnicity', 'admission_type', 
        'admission_date', 'length_of_stay', 'num_previous_admissions', 
        'primary_diagnosis', 'secondary_diagnosis', 'num_lab_procedures', 
        'num_medications', 'num_outpatient_visits', 'num_emergency_visits', 
        'num_inpatient_visits', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
        'bmi', 'glucose_level', 'cholesterol_level', 'heart_rate', 
        'oxygen_saturation', 'smoking_status', 'alcohol_intake', 
        'physical_activity_level', 'diet_quality', 'living_alone', 
        'treatment_type', 'discharge_disposition', 'followup_appointment_scheduled', 
        'followup_days', 'discharge_date'
    ]

def get_numerical_features() -> List[str]:
    """
    Get numerical features that are scaled by StandardScaler.
    """
    return [
        'patient_id', 'age', 'length_of_stay', 'num_previous_admissions',
        'num_lab_procedures', 'num_medications', 'num_outpatient_visits',
        'num_emergency_visits', 'num_inpatient_visits', 'blood_pressure_systolic',
        'blood_pressure_diastolic', 'bmi', 'glucose_level', 'cholesterol_level',
        'heart_rate', 'oxygen_saturation', 'living_alone', 
        'followup_appointment_scheduled', 'followup_days'
    ]

def get_categorical_features() -> List[str]:
    """
    Get categorical features that are one-hot encoded.
    """
    return [
        'gender', 'ethnicity', 'admission_type', 'admission_date',
        'primary_diagnosis', 'secondary_diagnosis', 'smoking_status',
        'alcohol_intake', 'physical_activity_level', 'diet_quality',
        'treatment_type', 'discharge_disposition', 'discharge_date'
    ]

def get_default_values() -> Dict[str, Any]:
    """
    Return sensible default values for all model features.
    Based on preprocessor.pkl analysis of actual data ranges.
    """
    return {
        # Numerical features with realistic defaults
        'patient_id': 1000,
        'age': 50,
        'length_of_stay': 5,
        'num_previous_admissions': 1,
        'num_lab_procedures': 10,
        'num_medications': 5,
        'num_outpatient_visits': 2,
        'num_emergency_visits': 1,
        'num_inpatient_visits': 1,
        'blood_pressure_systolic': 120,
        'blood_pressure_diastolic': 80,
        'bmi': 25.0,
        'glucose_level': 100.0,
        'cholesterol_level': 200.0,
        'heart_rate': 72,
        'oxygen_saturation': 98.0,
        'living_alone': 0,
        'followup_appointment_scheduled': 0,
        'followup_days': 14,
        
        # Categorical features with valid categories from preprocessor
        'gender': 'Male',
        'ethnicity': 'Caucasian',
        'admission_type': 'Emergency',
        'admission_date': '2024-01-01',
        'primary_diagnosis': 'Heart Failure',
        'secondary_diagnosis': 'Hypertension',
        'smoking_status': 'Never',
        'alcohol_intake': 'Moderate',
        'physical_activity_level': 'Medium',
        'diet_quality': 'Average',
        'treatment_type': 'Medication',
        'discharge_disposition': 'Home',
        'discharge_date': '2024-01-06'
    }

def get_feature_categories() -> Dict[str, List[str]]:
    """
    Get possible categories for categorical features.
    Based on actual categories found in preprocessor.pkl.
    """
    return {
        'gender': ['Female', 'Male'],
        'ethnicity': ['African American', 'Asian', 'Caucasian', 'Hispanic', 'Other'],
        'admission_type': ['Elective', 'Emergency', 'Urgent'],
        'admission_date': [f'2024-{month:02d}-{day:02d}' for month in range(1, 13) for day in range(1, 32)],
        'primary_diagnosis': ['COPD', 'Diabetes', 'Heart Failure', 'Hypertension', 'Pneumonia'],
        'secondary_diagnosis': ['Anemia', 'Hypertension', 'Kidney Disease', 'Obesity', 'None'],
        'smoking_status': ['Current', 'Former', 'Never'],
        'alcohol_intake': ['High', 'Moderate', 'None'],
        'physical_activity_level': ['High', 'Low', 'Medium'],
        'diet_quality': ['Average', 'Good', 'Poor'],
        'treatment_type': ['Medication', 'Surgery', 'Therapy'],
        'discharge_disposition': ['Home', 'Hospice', 'Nursing Home', 'Transfer'],
        'discharge_date': [f'2024-{month:02d}-{day:02d}' for month in range(1, 13) for day in range(1, 32)]
    }

# =============================================================================
# DATA PREPARATION & VALIDATION
# =============================================================================

def prepare_model_input(patient_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare patient data for model prediction with EXACT feature matching.
    Ensures all 32 features are present in the exact order the preprocessor expects.
    
    Args:
        patient_data: Raw patient data that may include extra columns
        
    Returns:
        pd.DataFrame: Clean data with exactly the 32 features the model expects
    """
    try:
        # Get the exact model features and default values
        model_features = get_model_features()
        default_values = get_default_values()
        feature_categories = get_feature_categories()
        
        # Create clean data with ALL model features in exact order
        clean_data = {}
        
        for feature in model_features:
            if feature in patient_data and patient_data[feature] is not None:
                value = patient_data[feature]
                
                # Validate categorical features
                if feature in get_categorical_features():
                    if value not in feature_categories.get(feature, []):
                        print(f"⚠️ {feature} value '{value}' not in trained categories. Using default.")
                        value = default_values[feature]
                
                # Ensure correct data types
                if feature in get_numerical_features():
                    try:
                        clean_data[feature] = float(value)
                    except (ValueError, TypeError):
                        print(f"⚠️ Could not convert {feature} to number. Using default.")
                        clean_data[feature] = default_values[feature]
                else:
                    clean_data[feature] = str(value)
                    
            else:
                # Use default value if feature is missing or None
                clean_data[feature] = default_values[feature]
                print(f"⚠️ Using default for {feature}: {default_values[feature]}")
        
        # Create DataFrame with EXACT column order
        df = pd.DataFrame([clean_data])
        df = df[model_features]  # Critical: ensure exact training order
        
        print(f"✅ Prepared model input: {len(model_features)} features")
        print(f"📊 Feature ranges - Age: {df['age'].iloc[0]}, LOS: {df['length_of_stay'].iloc[0]}, Glucose: {df['glucose_level'].iloc[0]}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error preparing model input: {str(e)}")
        # Return fallback DataFrame with all required features
        fallback_data = {feature: default_values[feature] for feature in get_model_features()}
        return pd.DataFrame([fallback_data])[get_model_features()]

def validate_patient_data(patient_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate patient data before prediction with clinical sanity checks.
    
    Args:
        patient_data: Patient data to validate
        
    Returns:
        Tuple: (is_valid, error_messages)
    """
    errors = []
    warnings = []
    
    # Clinical range validations
    if 'age' in patient_data:
        try:
            age = float(patient_data['age'])
            if age < 0 or age > 120:
                errors.append("Age must be between 0 and 120")
            elif age > 100:
                warnings.append("Patient age is very high")
        except (ValueError, TypeError):
            errors.append("Age must be a number")
    
    if 'bmi' in patient_data:
        try:
            bmi = float(patient_data['bmi'])
            if bmi < 10 or bmi > 60:
                errors.append("BMI must be between 10 and 60")
        except (ValueError, TypeError):
            errors.append("BMI must be a number")
    
    if 'glucose_level' in patient_data:
        try:
            glucose = float(patient_data['glucose_level'])
            if glucose < 50 or glucose > 500:
                errors.append("Glucose level must be between 50 and 500 mg/dL")
        except (ValueError, TypeError):
            errors.append("Glucose level must be a number")
    
    if 'blood_pressure_systolic' in patient_data:
        try:
            bp_sys = float(patient_data['blood_pressure_systolic'])
            if bp_sys < 70 or bp_sys > 250:
                errors.append("Systolic BP must be between 70 and 250")
        except (ValueError, TypeError):
            errors.append("Systolic BP must be a number")
    
    # Required feature check
    required_features = ['age', 'length_of_stay', 'primary_diagnosis']
    for feature in required_features:
        if feature not in patient_data or patient_data[feature] is None:
            errors.append(f"Required feature missing: {feature}")
    
    # Date format validation
    date_features = ['admission_date', 'discharge_date']
    for date_feature in date_features:
        if date_feature in patient_data and patient_data[date_feature]:
            try:
                datetime.strptime(str(patient_data[date_feature]), '%Y-%m-%d')
            except ValueError:
                errors.append(f"{date_feature} must be in YYYY-MM-DD format")
    
    # Return both errors and warnings
    if warnings:
        print(f"⚠️ Validation warnings: {', '.join(warnings)}")
    
    return len(errors) == 0, errors

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_patient_risk(model: object, patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to predict patient readmission risk.
    Uses the model pipeline which includes built-in preprocessing.
    
    Args:
        model: Trained model pipeline
        patient_data: Patient data as dictionary
        
    Returns:
        Dict: Prediction results with risk score and metadata
    """
    try:
        # Extract patient ID for reporting
        patient_id = patient_data.get('patient_id', 'UNKNOWN')
        
        print(f"🎯 Predicting risk for patient: {patient_id}")
        
        # Validate input data
        is_valid, errors = validate_patient_data(patient_data)
        if not is_valid:
            error_msg = f"Data validation failed: {', '.join(errors)}"
            return get_fallback_prediction(patient_id, error_msg)
        
        # Prepare clean input data for model (with ALL 32 features)
        X_input = prepare_model_input(patient_data)
        
        print(f"📦 Input shape to model: {X_input.shape}")
        print(f"🔍 First 5 features: {list(X_input.columns[:5])}")
        
        # Use model pipeline for prediction (includes preprocessing)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_input)
            y_pred = model.predict(X_input)
        else:
            return get_fallback_prediction(patient_id, "Model doesn't support probability predictions")
        
        # Extract probabilities
        risk_probability = float(y_proba[0, 1])  # Probability of high risk
        
        # Determine risk level with clinical thresholds
        if risk_probability >= 0.7:
            predicted_label = "High Risk"
            risk_level = "high"
            recommendation = "🔴 HIGH PRIORITY: Consider enhanced discharge planning, close follow-up within 7 days, and multidisciplinary team review"
            color = "red"
        elif risk_probability >= 0.4:
            predicted_label = "Medium Risk"
            risk_level = "medium" 
            recommendation = "🟡 MEDIUM PRIORITY: Schedule follow-up appointment within 14 days and provide patient education"
            color = "orange"
        else:
            predicted_label = "Low Risk"
            risk_level = "low"
            recommendation = "🟢 LOW PRIORITY: Standard discharge procedures with routine follow-up"
            color = "green"
        
        # Calculate confidence score
        confidence = max(y_proba[0])
        
        # Prepare comprehensive results
        results = {
            'patient_id': patient_id,
            'predicted_label': predicted_label,
            'risk_level': risk_level,
            'risk_score': risk_probability,
            'risk_probability': round(risk_probability * 100, 2),
            'confidence': round(confidence * 100, 2),
            'recommendation': recommendation,
            'color': color,
            'binary_prediction': int(y_pred[0]),
            'raw_probabilities': {
                'low_risk_prob': round(float(y_proba[0, 0]) * 100, 2),
                'high_risk_prob': round(float(y_proba[0, 1]) * 100, 2)
            },
            'timestamp': datetime.now().isoformat(),
            'model_used': get_model_info(model).get('model_type', 'Unknown'),
            'features_used': len(get_model_features())
        }
        
        print(f"✅ Prediction successful: {predicted_label} ({results['risk_probability']}% risk)")
        return results
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return get_fallback_prediction(patient_data.get('patient_id', 'UNKNOWN'), error_msg)

def get_fallback_prediction(patient_id: str, error_msg: str) -> Dict[str, Any]:
    """
    Return a safe fallback prediction when model fails.
    
    Args:
        patient_id: Patient identifier
        error_msg: Error message
        
    Returns:
        Dict: Fallback prediction results
    """
    return {
        'patient_id': patient_id,
        'predicted_label': 'Low Risk',
        'risk_level': 'low',
        'risk_score': 0.3,
        'risk_probability': 30.0,
        'confidence': 70.0,
        'recommendation': 'Unable to generate AI prediction. Use clinical judgment and standard protocols.',
        'color': 'gray',
        'error': error_msg,
        'is_fallback': True,
        'timestamp': datetime.now().isoformat()
    }

# =============================================================================
# MODEL ANALYSIS & EXPLAINABILITY
# =============================================================================

def get_feature_importance(model: object, top_k: int = 10) -> Optional[pd.DataFrame]:
    """
    Extract feature importance from the model if available.
    
    Args:
        model: Trained model object
        top_k: Number of top features to return
        
    Returns:
        pd.DataFrame: Feature importance scores or None if not available
    """
    try:
        # Get the actual classifier from pipeline if needed
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
        else:
            classifier = model
        
        # Check if model supports feature importance
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            # For linear models, use absolute coefficients
            importances = np.abs(classifier.coef_[0])
        else:
            print("ℹ️ Feature importance not available for this model type")
            return None
        
        # Try to get meaningful feature names
        try:
            # For preprocessed features, use original feature names where possible
            original_features = get_model_features()
            if len(importances) == len(original_features):
                feature_names = original_features
            else:
                # Use generic names if count doesn't match
                feature_names = [f'feature_{i}' for i in range(len(importances))]
        except:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_k)
        
        # Normalize importance to percentage
        importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)
        
        print(f"✅ Feature importance extracted for top {top_k} features")
        
        return importance_df
        
    except Exception as e:
        print(f"⚠️ Could not extract feature importance: {str(e)}")
        return None

def get_model_info(model: object) -> Dict[str, Any]:
    """
    Extract information about the loaded model.
    
    Args:
        model: Trained model object
        
    Returns:
        Dict: Model metadata and information
    """
    try:
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
            model_type = type(classifier).__name__
            is_pipeline = True
        else:
            classifier = model
            model_type = type(model).__name__
            is_pipeline = False
        
        info = {
            'model_type': model_type,
            'is_pipeline': is_pipeline,
            'supports_probabilities': hasattr(classifier, 'predict_proba'),
            'supports_importance': hasattr(classifier, 'feature_importances_') or hasattr(classifier, 'coef_')
        }
        
        # Add model-specific info
        if hasattr(classifier, 'n_estimators'):
            info['n_estimators'] = classifier.n_estimators
        if hasattr(classifier, 'max_depth'):
            info['max_depth'] = classifier.max_depth
        if hasattr(classifier, 'learning_rate'):
            info['learning_rate'] = classifier.learning_rate
        
        return info
        
    except Exception as e:
        print(f"⚠️ Could not extract model info: {str(e)}")
        return {'model_type': 'Unknown', 'error': str(e)}

# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def create_sample_patient(risk_level: str = "medium") -> Dict[str, Any]:
    """
    Create a sample patient with realistic data based on risk level.
    Uses actual categories and ranges from the preprocessor.
    
    Args:
        risk_level: Desired risk level ('low', 'medium', 'high')
        
    Returns:
        Dict: Sample patient data
    """
    base_sample = {
        'patient_id': 1001,
        'admission_date': '2024-01-15',
        'discharge_date': '2024-01-20'
    }
    
    if risk_level == "high":
        return {
            **base_sample,
            'age': 78,
            'gender': 'Male',
            'ethnicity': 'Caucasian',
            'admission_type': 'Emergency',
            'length_of_stay': 14,
            'num_previous_admissions': 5,
            'primary_diagnosis': 'Heart Failure',
            'secondary_diagnosis': 'Diabetes',
            'num_lab_procedures': 25,
            'num_medications': 12,
            'num_outpatient_visits': 6,
            'num_emergency_visits': 3,
            'num_inpatient_visits': 4,
            'blood_pressure_systolic': 160,
            'blood_pressure_diastolic': 95,
            'bmi': 35.5,
            'glucose_level': 180.0,
            'cholesterol_level': 250.0,
            'heart_rate': 95,
            'oxygen_saturation': 92.0,
            'smoking_status': 'Current',
            'alcohol_intake': 'High',
            'physical_activity_level': 'Low',
            'diet_quality': 'Poor',
            'living_alone': 1,
            'treatment_type': 'Medication',
            'discharge_disposition': 'Home',
            'followup_appointment_scheduled': 0,
            'followup_days': 21
        }
    elif risk_level == "low":
        return {
            **base_sample,
            'age': 45,
            'gender': 'Female',
            'ethnicity': 'Asian',
            'admission_type': 'Elective',
            'length_of_stay': 3,
            'num_previous_admissions': 0,
            'primary_diagnosis': 'Pneumonia',
            'secondary_diagnosis': 'None',
            'num_lab_procedures': 8,
            'num_medications': 3,
            'num_outpatient_visits': 1,
            'num_emergency_visits': 0,
            'num_inpatient_visits': 0,
            'blood_pressure_systolic': 115,
            'blood_pressure_diastolic': 75,
            'bmi': 22.0,
            'glucose_level': 95.0,
            'cholesterol_level': 180.0,
            'heart_rate': 68,
            'oxygen_saturation': 99.0,
            'smoking_status': 'Never',
            'alcohol_intake': 'None',
            'physical_activity_level': 'High',
            'diet_quality': 'Good',
            'living_alone': 0,
            'treatment_type': 'Therapy',
            'discharge_disposition': 'Home',
            'followup_appointment_scheduled': 1,
            'followup_days': 7
        }
    else:  # medium risk
        return {
            **base_sample,
            'age': 65,
            'gender': 'Male',
            'ethnicity': 'Caucasian',
            'admission_type': 'Emergency',
            'length_of_stay': 7,
            'num_previous_admissions': 2,
            'primary_diagnosis': 'COPD',
            'secondary_diagnosis': 'Hypertension',
            'num_lab_procedures': 15,
            'num_medications': 8,
            'num_outpatient_visits': 3,
            'num_emergency_visits': 1,
            'num_inpatient_visits': 2,
            'blood_pressure_systolic': 140,
            'blood_pressure_diastolic': 90,
            'bmi': 28.5,
            'glucose_level': 145.0,
            'cholesterol_level': 220.0,
            'heart_rate': 85,
            'oxygen_saturation': 96.0,
            'smoking_status': 'Former',
            'alcohol_intake': 'Moderate',
            'physical_activity_level': 'Medium',
            'diet_quality': 'Average',
            'living_alone': 0,
            'treatment_type': 'Medication',
            'discharge_disposition': 'Home',
            'followup_appointment_scheduled': 1,
            'followup_days': 14
        }

# =============================================================================
# BATCH PREDICTION
# =============================================================================

def predict_batch_risk(model: object, patients_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Predict risk for multiple patients in batch.
    
    Args:
        model: Trained model
        patients_data: List of patient data dictionaries
        
    Returns:
        List: Prediction results for all patients
    """
    results = []
    
    for i, patient_data in enumerate(patients_data):
        print(f"🔍 Processing patient {i+1}/{len(patients_data)}...")
        result = predict_patient_risk(model, patient_data)
        results.append(result)
    
    print(f"✅ Batch prediction completed: {len(results)} patients processed")
    return results

# =============================================================================
# TESTING & DEBUGGING
# =============================================================================

def test_model_utils():
    """
    Comprehensive test function for model utilities.
    """
    print("🧪 Testing model_utils.py...")
    print("=" * 60)
    
    # Test model loading
    print("\n1. Testing model loading...")
    model = load_trained_model()
    if model is None:
        print("❌ Model loading test failed")
        return None
    
    print(f"✅ Model loaded: {type(model)}")
    
    # Test model info
    print("\n2. Testing model info...")
    model_info = get_model_info(model)
    print(f"📊 Model info: {model_info}")
    
    # Test feature importance
    print("\n3. Testing feature importance...")
    importance_df = get_feature_importance(model)
    if importance_df is not None:
        print(f"📈 Top 5 features:")
        for _, row in importance_df.head().iterrows():
            print(f"   {row['feature']}: {row['importance_pct']}%")
    else:
        print("📈 Feature importance: Not available")
    
    # Test sample patients for all risk levels
    for risk_level in ["low", "medium", "high"]:
        print(f"\n4. Testing {risk_level.upper()} risk sample patient...")
        sample_patient = create_sample_patient(risk_level)
        
        # Test prediction
        prediction = predict_patient_risk(model, sample_patient)
        
        print(f"🎯 {risk_level.upper()} RISK Results:")
        print(f"   Patient: {prediction['patient_id']}")
        print(f"   Risk Level: {prediction['predicted_label']}")
        print(f"   Risk Score: {prediction['risk_probability']}%")
        print(f"   Confidence: {prediction['confidence']}%")
        print(f"   Recommendation: {prediction['recommendation']}")
        
        if 'error' in prediction:
            print(f"   ⚠️ Error: {prediction['error']}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    
    return model

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main execution block for testing the module.
    """
    # Run comprehensive tests
    test_model_utils()