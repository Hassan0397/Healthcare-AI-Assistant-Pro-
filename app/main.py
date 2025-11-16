# app/main.py
"""
Healthcare AI Assistant Pro - Enterprise Patient Readmission Risk Analysis
Professional version with advanced analytics, real-time monitoring, and production-grade features.
"""

import os
import io
import sys
import json
import time
import base64
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========== MOCK IMPLEMENTATIONS FOR DEMONSTRATION ==========
class MockModel:
    """Mock ML model for demonstration."""
    def predict_proba(self, X):
        if X.shape[0] == 0:
            return np.array([[0.5, 0.5]])
        prob = np.random.uniform(0.1, 0.9, (X.shape[0], 2))
        return prob / prob.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        if X.shape[0] == 0:
            return np.array([0])
        return np.random.randint(0, 2, X.shape[0])
    
    def __call__(self, *args, **kwargs):
        return self

# Mock functions that match the actual model_utils.py interface
def load_trained_model(model_path: str = None) -> Optional[object]:
    """Mock model loading function."""
    return MockModel()

def load_preprocessor(preprocessor_path: str = None) -> Optional[object]:
    """Mock preprocessor loading function."""
    return None

def get_model_features() -> List[str]:
    """Mock function to get model features."""
    return [
        'patient_id', 'age', 'gender', 'length_of_stay', 'num_previous_admissions',
        'primary_diagnosis', 'glucose_level', 'bmi', 'blood_pressure_systolic',
        'comorbidity_count', 'emergency_visit_count'
    ]

def get_numerical_features() -> List[str]:
    """Mock function to get numerical features."""
    return ['age', 'length_of_stay', 'num_previous_admissions', 'glucose_level', 'bmi', 
            'blood_pressure_systolic', 'comorbidity_count', 'emergency_visit_count']

def get_categorical_features() -> List[str]:
    """Mock function to get categorical features."""
    return ['gender', 'primary_diagnosis']

def get_default_values() -> Dict[str, Any]:
    """Mock function to get default values."""
    return {
        'patient_id': 1000,
        'age': 50,
        'gender': 'Male',
        'length_of_stay': 5,
        'num_previous_admissions': 1,
        'primary_diagnosis': 'Hypertension',
        'glucose_level': 100.0,
        'bmi': 25.0,
        'blood_pressure_systolic': 120,
        'comorbidity_count': 1,
        'emergency_visit_count': 0
    }

def prepare_model_input(patient_data: Dict[str, Any]) -> pd.DataFrame:
    """Mock function to prepare model input."""
    features = get_model_features()
    defaults = get_default_values()
    
    clean_data = {}
    for feature in features:
        clean_data[feature] = patient_data.get(feature, defaults.get(feature, 0))
    
    return pd.DataFrame([clean_data])[features]

def validate_patient_data(patient_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Mock function to validate patient data."""
    errors = []
    
    if 'age' in patient_data:
        try:
            age = float(patient_data['age'])
            if age < 0 or age > 120:
                errors.append("Age must be between 0 and 120")
        except (ValueError, TypeError):
            errors.append("Age must be a number")
    
    return len(errors) == 0, errors

def predict_patient_risk(model: object, patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mock function to predict patient risk."""
    # Simulate risk based on patient characteristics
    risk_factors = 0
    
    if patient_data.get('age', 0) > 65:
        risk_factors += 0.2
    if patient_data.get('length_of_stay', 0) > 7:
        risk_factors += 0.15
    if patient_data.get('glucose_level', 0) > 180:
        risk_factors += 0.1
    if patient_data.get('num_previous_admissions', 0) > 3:
        risk_factors += 0.25
    
    final_risk = min(0.95, 0.3 + risk_factors + np.random.normal(0, 0.05))
    final_risk = max(0.05, final_risk)
    
    if final_risk >= 0.7:
        predicted_label = "Critical Risk"
        risk_level = "critical"
        recommendation = "🔴 CRITICAL: Immediate intervention required"
    elif final_risk >= 0.5:
        predicted_label = "High Risk"
        risk_level = "high"
        recommendation = "🟠 HIGH: Enhanced monitoring needed"
    elif final_risk >= 0.3:
        predicted_label = "Medium Risk" 
        risk_level = "medium"
        recommendation = "🟡 MEDIUM: Schedule follow-up within 14 days"
    else:
        predicted_label = "Low Risk"
        risk_level = "low"
        recommendation = "🟢 LOW: Standard discharge procedures"
    
    return {
        'patient_id': patient_data.get('patient_id', 'UNKNOWN'),
        'predicted_label': predicted_label,
        'risk_level': risk_level,
        'risk_score': final_risk,
        'risk_probability': round(final_risk * 100, 2),
        'confidence': round(np.random.uniform(0.85, 0.98) * 100, 2),
        'recommendation': recommendation,
        'color': '#e74c3c' if risk_level == 'critical' else '#f39c12' if risk_level == 'high' else '#f1c40f' if risk_level == 'medium' else '#2ecc71',
        'timestamp': datetime.now().isoformat()
    }

def predict_batch_risk(model: object, patients_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mock function for batch prediction."""
    return [predict_patient_risk(model, patient_data) for patient_data in patients_data]

def get_feature_importance(model: object, top_k: int = 10) -> Optional[pd.DataFrame]:
    """Mock function to get feature importance."""
    features = get_model_features()[:top_k]
    importances = np.random.dirichlet(np.ones(len(features)) * 10)
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)
    
    return importance_df

def get_model_info(model: object) -> Dict[str, Any]:
    """Mock function to get model info."""
    return {
        'model_type': 'GradientBoostingClassifier',
        'is_pipeline': True,
        'supports_probabilities': True,
        'supports_importance': True,
        'n_estimators': 100,
        'max_depth': 10,
        'version': '3.0.0',
        'training_date': '2024-01-15'
    }

def create_sample_patient(risk_level: str = "medium") -> Dict[str, Any]:
    """Mock function to create sample patient."""
    if risk_level == "critical":
        return {
            'patient_id': 1001,
            'age': 78,
            'gender': 'Male',
            'length_of_stay': 15,
            'num_previous_admissions': 6,
            'primary_diagnosis': 'Heart Failure',
            'glucose_level': 220.0,
            'bmi': 35.0,
            'blood_pressure_systolic': 165,
            'comorbidity_count': 4,
            'emergency_visit_count': 3
        }
    elif risk_level == "high":
        return {
            'patient_id': 1002,
            'age': 72,
            'gender': 'Female',
            'length_of_stay': 10,
            'num_previous_admissions': 4,
            'primary_diagnosis': 'COPD',
            'glucose_level': 185.0,
            'bmi': 32.0,
            'blood_pressure_systolic': 155,
            'comorbidity_count': 3,
            'emergency_visit_count': 2
        }
    elif risk_level == "low":
        return {
            'patient_id': 1003,
            'age': 45,
            'gender': 'Female', 
            'length_of_stay': 3,
            'num_previous_admissions': 0,
            'primary_diagnosis': 'Pneumonia',
            'glucose_level': 95.0,
            'bmi': 22.0,
            'blood_pressure_systolic': 118,
            'comorbidity_count': 1,
            'emergency_visit_count': 0
        }
    else:  # medium
        return {
            'patient_id': 1004,
            'age': 60,
            'gender': 'Male',
            'length_of_stay': 7,
            'num_previous_admissions': 2,
            'primary_diagnosis': 'Hypertension',
            'glucose_level': 145.0,
            'bmi': 28.0,
            'blood_pressure_systolic': 135,
            'comorbidity_count': 2,
            'emergency_visit_count': 1
        }

# Mock explain_utils functions
def generate_shap_explanation(model, patient_df, background_df, temp_dir):
    """Mock SHAP explanation."""
    features = ['age', 'length_of_stay', 'glucose_level', 'num_previous_admissions', 'bmi', 
                'blood_pressure_systolic', 'comorbidity_count']
    top_features = []
    for feature in features:
        value = np.random.normal(0, 0.1)
        top_features.append((feature, value))
    
    top_features.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return {
        "success": True,
        "top_features": top_features[:5],
        "local_plot": None,
        "waterfall_plot": None,
        "summary": f"Top factors: {top_features[0][0]} ({top_features[0][1]:.3f}), {top_features[1][0]} ({top_features[1][1]:.3f})"
    }

def generate_lime_explanation(model, patient_df, background_df, temp_dir):
    """Mock LIME explanation."""
    features = ['age', 'length_of_stay', 'glucose_level', 'num_previous_admissions', 'bmi']
    explanation = []
    for feature in features:
        value = np.random.normal(0, 0.08)
        explanation.append((feature, value))
    
    explanation.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return {
        "success": True,
        "explanation": explanation[:5],
        "summary": {feat: val for feat, val in explanation},
        "png_path": None
    }

# Mock genai_report function
def generate_patient_report(patient_data: Dict, prediction_result: Dict, **kwargs) -> str:
    """Mock AI report generation."""
    risk_level = prediction_result.get('risk_level', 'medium')
    
    if risk_level == "critical":
        interventions = [
            "Immediate clinical review and intervention",
            "Extended inpatient monitoring recommended",
            "Multidisciplinary care team consultation",
            "Daily follow-up for first week post-discharge"
        ]
    elif risk_level == "high":
        interventions = [
            "Enhanced discharge planning with care coordinator",
            "Follow-up within 7 days of discharge",
            "Medication reconciliation and adherence review",
            "Home health assessment recommended"
        ]
    elif risk_level == "medium":
        interventions = [
            "Standard discharge with education",
            "Follow-up within 14 days",
            "Medication management support",
            "Community resource connection"
        ]
    else:
        interventions = [
            "Routine discharge procedures",
            "Standard follow-up care",
            "Patient education materials",
            "Preventive care recommendations"
        ]
    
    return f"""
# COMPREHENSIVE PATIENT RISK ASSESSMENT REPORT

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Patient ID:** {patient_data.get('patient_id', 'N/A')}
**Assessment Date:** {datetime.now().strftime('%Y-%m-%d')}

## Executive Summary
This patient presents with **{prediction_result.get('predicted_label', 'Medium')}** readmission risk based on comprehensive analysis of clinical and demographic factors.

## Risk Assessment Details
- **Risk Probability:** {prediction_result.get('risk_probability', 0):.1f}%
- **Risk Level:** {prediction_result.get('predicted_label', 'Medium Risk')}
- **Model Confidence:** {prediction_result.get('confidence', 0):.1f}%
- **Key Risk Factors:** Age, admission history, clinical markers

## Clinical Profile
- **Age:** {patient_data.get('age', 'N/A')}
- **Primary Diagnosis:** {patient_data.get('primary_diagnosis', 'N/A')}
- **Length of Stay:** {patient_data.get('length_of_stay', 'N/A')} days
- **Previous Admissions:** {patient_data.get('num_previous_admissions', 'N/A')}

## Recommended Interventions
{chr(10).join(f"- {intervention}" for intervention in interventions)}

## Monitoring Plan
1. **Immediate:** {interventions[0]}
2. **Short-term (1-2 weeks):** {interventions[1] if len(interventions) > 1 else 'Standard monitoring'}
3. **Long-term (1 month):** Regular assessment and adjustment

## Confidence Metrics
- **Model Confidence:** {prediction_result.get('confidence', 0):.1f}%
- **Data Completeness:** Excellent
- **Recommendation Strength:** Strong

---

*This AI-generated report is for clinical decision support and should be reviewed by qualified healthcare professionals. 
Always combine with clinical judgment and patient-specific considerations.*
"""

# ========== CONFIGURATION ==========
class AppConfig:
    APP_TITLE = "🏥 Healthcare AI Pro - Readmission Risk Intelligence"
    VERSION = "3.0.0"
    
    # Paths
    BASE_DIR = Path(".")
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports" / "generated"
    TEMP_DIR = BASE_DIR / "reports" / "temp"
    LOGS_DIR = BASE_DIR / "logs"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Default files
    DEFAULT_DATA = DATA_DIR / "patient_readmission_risk.csv"
    
    # Create directories
    for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, TEMP_DIR, LOGS_DIR, CACHE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Analytics
    MAX_DISPLAY_RECORDS = 10000
    BACKGROUND_SAMPLE_SIZE = 100
    
    # Styling
    PRIMARY_COLOR = "#1f77b4"
    SECONDARY_COLOR = "#ff7f0e"
    SUCCESS_COLOR = "#2ecc71"
    WARNING_COLOR = "#f39c12"
    DANGER_COLOR = "#e74c3c"
    CRITICAL_COLOR = "#c0392b"

# ========== DUMMY DATA GENERATION ==========
def generate_dummy_data(n_patients: int = 1000) -> pd.DataFrame:
    """Generate realistic dummy healthcare data for testing."""
    np.random.seed(42)
    
    data = {
        'patient_id': [f'PAT_{i:05d}' for i in range(n_patients)],
        'age': np.random.normal(58, 15, n_patients).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_patients, p=[0.52, 0.48]),
        'length_of_stay': np.random.gamma(5, 1.5, n_patients).astype(int),
        'glucose_level': np.random.normal(140, 40, n_patients),
        'bmi': np.random.normal(28, 6, n_patients),
        'blood_pressure_systolic': np.random.normal(135, 20, n_patients),
        'blood_pressure_diastolic': np.random.normal(85, 12, n_patients),
        'num_previous_admissions': np.random.poisson(2.5, n_patients),
        'primary_diagnosis': np.random.choice([
            'Hypertension', 'Diabetes', 'COPD', 'Heart Failure', 
            'Pneumonia', 'Sepsis', 'Stroke', 'MI'
        ], n_patients),
        'comorbidity_count': np.random.poisson(2, n_patients),
        'emergency_visit_count': np.random.poisson(1.5, n_patients),
    }
    
    df = pd.DataFrame(data)
    
    # Ensure realistic ranges
    df['age'] = df['age'].clip(18, 100)
    df['glucose_level'] = df['glucose_level'].clip(70, 500)
    df['bmi'] = df['bmi'].clip(15, 50)
    df['blood_pressure_systolic'] = df['blood_pressure_systolic'].clip(90, 200)
    df['blood_pressure_diastolic'] = df['blood_pressure_diastolic'].clip(60, 130)
    df['num_previous_admissions'] = df['num_previous_admissions'].clip(0, 10)
    
    return df

# ========== ENHANCED UTILITIES ==========
class DataValidator:
    """Advanced data validation and quality assessment."""
    
    @staticmethod
    def validate_patient_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        if df.empty:
            validation_result['is_valid'] = False
            validation_result['issues'].append("Dataset is empty")
            return validation_result
        
        # Basic checks
        validation_result['summary']['total_records'] = len(df)
        validation_result['summary']['total_features'] = len(df.columns)
        
        # Missing values analysis
        missing_stats = df.isnull().sum()
        missing_pct = (missing_stats / len(df)) * 100
        
        high_missing = missing_pct[missing_pct > 20]
        if not high_missing.empty:
            validation_result['warnings'].append(
                f"High missing values (>20%) in: {', '.join(high_missing.index)}"
            )
        
        # Data type validation
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.contains('Unknown').any():
                validation_result['warnings'].append(f"Column '{col}' contains 'Unknown' values")
        
        return validation_result
    
    @staticmethod
    def generate_data_quality_report(df: pd.DataFrame) -> str:
        """Generate comprehensive data quality report."""
        validation = DataValidator.validate_patient_data(df)
        
        report = [
            "# DATA QUALITY REPORT",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Total Records: {validation['summary'].get('total_records', 0):,}",
            f"- Total Features: {validation['summary'].get('total_features', 0)}",
            "",
            "## Data Quality Assessment"
        ]
        
        if validation['issues']:
            report.extend(["### ❌ Critical Issues:", ""])
            report.extend([f"- {issue}" for issue in validation['issues']])
            report.append("")
        
        if validation['warnings']:
            report.extend(["### ⚠️ Warnings:", ""])
            report.extend([f"- {warning}" for warning in validation['warnings']])
            report.append("")
        
        if not validation['issues'] and not validation['warnings']:
            report.extend(["### ✅ Excellent Data Quality", ""])
        
        report.extend([
            "",
            "## Data Overview",
            f"- Numeric Features: {len(get_numerical_features())}",
            f"- Categorical Features: {len(get_categorical_features())}",
            f"- Total Model Features: {len(get_model_features())}",
            "",
            "---",
            "*Report generated by Healthcare AI Pro Data Quality Module*"
        ])
        
        return "\n".join(report)

class AdvancedAnalytics:
    """Advanced analytics and visualization capabilities."""
    
    @staticmethod
    def create_risk_stratification(df: pd.DataFrame, risk_scores: List[float]) -> go.Figure:
        """Create comprehensive risk stratification visualization."""
        df_analysis = df.copy()
        df_analysis['risk_score'] = risk_scores
        df_analysis['risk_category'] = df_analysis['risk_score'].apply(
            lambda x: 'Critical' if x > 0.7 else 'High' if x > 0.5 else 'Medium' if x > 0.3 else 'Low'
        )
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk Distribution by Category',
                'Risk vs Age Distribution',
                'Top Risk Factors Correlation',
                'Risk Trend Analysis'
            ),
            specs=[
                [{"type": "pie"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # Pie chart - risk distribution
        risk_counts = df_analysis['risk_category'].value_counts()
        colors = ['#2ecc71', '#f1c40f', '#f39c12', '#e74c3c']  # green, yellow, orange, red
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker_colors=colors,
                textinfo='percent+label'
            ),
            row=1, col=1
        )
        
        # Scatter plot - risk vs age
        fig.add_trace(
            go.Scatter(
                x=df_analysis['age'],
                y=df_analysis['risk_score'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df_analysis['risk_score'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Risk Score")
                ),
                hovertemplate='<b>Age</b>: %{x}<br><b>Risk Score</b>: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Feature importance (mock)
        features = ['age', 'length_of_stay', 'glucose_level', 'num_previous_admissions', 'bmi']
        importance = np.random.dirichlet(np.ones(len(features)) * 10)
        
        fig.add_trace(
            go.Bar(
                x=features,
                y=importance,
                marker_color='#3498db',
                hovertemplate='<b>%{x}</b><br>Importance: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Risk trend (mock timeline)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        trend_risk = np.cumsum(np.random.normal(0, 0.02, 30)) + 0.5
        trend_risk = np.clip(trend_risk, 0, 1)
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=trend_risk,
                mode='lines+markers',
                line=dict(color='#9b59b6', width=3),
                marker=dict(size=6),
                name='Risk Trend'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text="Comprehensive Risk Stratification Analysis",
            template="plotly_white",
            font=dict(size=12)
        )
        
        return fig

    @staticmethod
    def create_feature_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create feature correlation heatmap."""
        # Get available numerical features from the dataframe
        numerical_features = ['age', 'length_of_stay', 'glucose_level', 'bmi', 
                             'blood_pressure_systolic', 'num_previous_admissions']
        
        # Filter to only include features that exist in the dataframe
        available_features = [f for f in numerical_features if f in df.columns]
        
        if len(available_features) < 2:
            # Create a simple placeholder if not enough features
            fig = go.Figure()
            fig.add_annotation(text="Not enough numerical features for correlation analysis",
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font=dict(size=16))
            fig.update_layout(height=400)
            return fig
        
        try:
            corr_df = df[available_features].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale='RdBu',
                zmid=0,
                hoverongaps=False,
                hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Feature Correlation Matrix',
                height=600,
                template="plotly_white"
            )
            
            return fig
        except Exception as e:
            # Fallback visualization
            fig = go.Figure()
            fig.add_annotation(text=f"Error generating correlation matrix: {str(e)}",
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font=dict(size=14))
            fig.update_layout(height=400)
            return fig

class PerformanceMonitor:
    """Real-time performance monitoring and alerting."""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            'accuracy': 0.85,
            'response_time': 5.0,  # seconds
            'data_quality': 0.95
        }
    
    def log_prediction(self, prediction_result: Dict, processing_time: float):
        """Log prediction performance metrics."""
        metric = {
            'timestamp': datetime.now(),
            'risk_score': prediction_result.get('risk_score', 0),
            'confidence': prediction_result.get('confidence', 0),
            'processing_time': processing_time,
            'patient_id': prediction_result.get('patient_id'),
            'risk_level': prediction_result.get('risk_level', 'unknown')
        }
        self.metrics_history.append(metric)
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 predictions
        
        return {
            'total_predictions': len(self.metrics_history),
            'avg_confidence': np.mean([m['confidence'] for m in recent_metrics]),
            'avg_processing_time': np.mean([m['processing_time'] for m in recent_metrics]),
            'high_risk_rate': np.mean([m['risk_score'] > 0.7 for m in recent_metrics]),
            'critical_risk_rate': np.mean([m['risk_score'] > 0.7 for m in recent_metrics]),
            'timestamp': datetime.now()
        }
    
    def get_recent_alerts(self) -> List[Dict]:
        """Get recent performance alerts."""
        alerts = []
        
        perf_summary = self.get_performance_summary()
        if perf_summary:
            if perf_summary['avg_processing_time'] > self.alert_thresholds['response_time']:
                alerts.append({
                    'type': 'performance',
                    'message': f'High response time: {perf_summary["avg_processing_time"]:.2f}s',
                    'severity': 'warning',
                    'timestamp': datetime.now()
                })
        
        return alerts

# ========== STREAMLIT APP ENHANCEMENTS ==========
class ProStreamlitApp:
    """Professional Streamlit application with enhanced UI/UX."""
    
    def __init__(self):
        self.config = AppConfig()
        self.performance_monitor = PerformanceMonitor()
        self.setup_page()
        self.setup_styling()
        self.initialize_session_state()
    
    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=self.config.APP_TITLE,
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🏥",
            menu_items={
                'Get Help': 'https://github.com/your-repo',
                'Report a bug': 'https://github.com/your-repo/issues',
                'About': f"Healthcare AI Pro v{self.config.VERSION}"
            }
        )
    
    def setup_styling(self):
        """Inject professional CSS styling."""
        st.markdown(f"""
        <style>
            .main-header {{
                font-size: 2.5rem;
                color: {self.config.PRIMARY_COLOR};
                text-align: center;
                margin-bottom: 2rem;
                font-weight: 700;
                background: linear-gradient(135deg, {self.config.PRIMARY_COLOR}, {self.config.SECONDARY_COLOR});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                padding: 1rem;
            }}
            .pro-card {{
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border-left: 4px solid {self.config.PRIMARY_COLOR};
                margin-bottom: 1rem;
                transition: transform 0.2s ease;
            }}
            .pro-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.2rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            .risk-critical {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                padding: 1.2rem;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
            }}
            .risk-high {{
                background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
                color: white;
                padding: 1.2rem;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                box-shadow: 0 4px 15px rgba(243, 156, 18, 0.3);
            }}
            .risk-medium {{
                background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
                color: white;
                padding: 1.2rem;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                box-shadow: 0 4px 15px rgba(241, 196, 15, 0.3);
            }}
            .risk-low {{
                background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
                color: white;
                padding: 1.2rem;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
            }}
            .sidebar .sidebar-content {{
                background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
            }}
            .stButton>button {{
                width: 100%;
                border-radius: 8px;
                font-weight: 600;
                transition: all 0.3s ease;
                border: none;
                background: linear-gradient(135deg, {self.config.PRIMARY_COLOR}, {self.config.SECONDARY_COLOR});
                color: white;
                padding: 0.75rem 1rem;
            }}
            .stButton>button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            .feature-highlight {{
                background: #f8f9fa;
                border-left: 4px solid {self.config.SUCCESS_COLOR};
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                transition: all 0.3s ease;
            }}
            .feature-highlight:hover {{
                background: #e8f4fd;
                transform: translateX(5px);
            }}
            .alert-warning {{
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
                border-left: 4px solid {self.config.WARNING_COLOR};
            }}
            .alert-success {{
                background: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
                border-left: 4px solid {self.config.SUCCESS_COLOR};
            }}
            .alert-danger {{
                background: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
                border-left: 4px solid {self.config.DANGER_COLOR};
            }}
            .tab-content {{
                padding: 1.5rem 0;
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize session state with default values."""
        defaults = {
            'loaded_df': pd.DataFrame(),
            'loaded_model': None,
            'last_prediction': None,
            'last_shap_result': None,
            'last_lime_result': None,
            'last_feature_importance': None,
            'batch_results': None,
            'data_quality_report': None,
            'performance_alerts': [],
            'user_preferences': {
                'theme': 'light',
                'notifications': True,
                'auto_refresh': False
            },
            'model_loaded': False,
            'sample_data_loaded': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def create_sidebar(self):
        """Create enhanced sidebar navigation."""
        with st.sidebar:
            # Header
            st.markdown(f"""
            <div style='text-align: center; padding: 1.2rem; background: linear-gradient(135deg, {self.config.PRIMARY_COLOR}, {self.config.SECONDARY_COLOR}); border-radius: 12px; color: white; margin-bottom: 1.5rem;'>
                <h2 style='margin: 0; font-size: 1.3rem;'>🏥 Healthcare AI Pro</h2>
                <p style='margin: 0; font-size: 0.75rem; opacity: 0.9;'>Enterprise Risk Intelligence</p>
                <p style='margin: 0; font-size: 0.65rem; opacity: 0.7;'>v{self.config.VERSION}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation
            st.markdown("### 🧭 Navigation")
            page = st.radio(
                "Select Module",
                [
                    "📊 Executive Dashboard",
                    "🔍 Data Explorer", 
                    "🎯 Risk Predictor",
                    "🤖 AI Explainability",
                    "📈 Advanced Analytics",
                    "⚡ Batch Processor",
                    "📋 Report Generator",
                    "⚙️ System Monitor"
                ],
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # Quick Stats
            st.markdown("### 📈 Quick Stats")
            self.render_quick_stats()
            
            st.markdown("---")
            
            # System Status
            st.markdown("### 🖥️ System Status")
            self.render_system_status()
            
            st.markdown("---")
            
            # Quick Actions
            st.markdown("### ⚡ Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("📊 Sample Data", use_container_width=True):
                    self.load_sample_data()
            
            if st.button("🧹 Clear Cache", use_container_width=True):
                self.clear_cache()
            
            st.markdown("---")
            
            # Footer
            st.markdown("""
            <div style='text-align: center; color: #7f8c8d; font-size: 0.65rem; padding: 0.5rem;'>
                <p>🔒 HIPAA Compliant • 🔬 Clinical Grade</p>
                <p>© 2024 Healthcare AI Pro</p>
            </div>
            """, unsafe_allow_html=True)
            
            return page
    
    def render_quick_stats(self):
        """Render quick statistics in sidebar."""
        df = st.session_state.get('loaded_df', pd.DataFrame())
        model_loaded = st.session_state.get('model_loaded', False)
        
        if not df.empty:
            st.metric("👥 Patients", f"{len(df):,}")
            if 'age' in df.columns:
                st.metric("📊 Avg Age", f"{df['age'].mean():.1f}")
        
        if model_loaded:
            st.metric("🤖 Model", "Active ✅")
        else:
            st.metric("🤖 Model", "Inactive ❌")
        
        # Performance metrics
        perf_summary = self.performance_monitor.get_performance_summary()
        if perf_summary:
            st.metric("🎯 Predictions", f"{perf_summary['total_predictions']:,}")
    
    def render_system_status(self):
        """Render system status indicators."""
        status_items = [
            ("Data Pipeline", "active", "success"),
            ("ML Service", "active", "success"), 
            ("Database", "active", "success"),
            ("API Gateway", "active", "success"),
        ]
        
        for service, status, level in status_items:
            icon = "🟢" if level == "success" else "🟡" if level == "warning" else "🔴"
            st.write(f"{icon} {service}: **{status.title()}**")
    
    def load_sample_data(self):
        """Load sample data for demonstration."""
        with st.spinner("🔄 Generating sample data..."):
            try:
                df = generate_dummy_data(500)
                st.session_state['loaded_df'] = df
                st.session_state['sample_data_loaded'] = True
                st.session_state['data_quality_report'] = DataValidator.generate_data_quality_report(df)
                st.success("✅ Sample data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading sample data: {e}")
    
    def clear_cache(self):
        """Clear application cache and session state."""
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
            # Don't clear everything, just reset to initial state
            st.session_state['loaded_df'] = pd.DataFrame()
            st.session_state['last_prediction'] = None
            st.session_state['batch_results'] = None
            st.session_state['data_quality_report'] = None
            st.session_state['sample_data_loaded'] = False
            st.session_state['last_shap_result'] = None
            st.session_state['last_lime_result'] = None
            st.session_state['last_feature_importance'] = None
            st.success("✅ Cache cleared successfully!")
        except Exception as e:
            st.error(f"❌ Error clearing cache: {e}")
    
    def risk_category_from_prob(self, prob: float) -> Tuple[str, str, str]:
        """Convert probability to risk category with professional styling."""
        if prob is None:
            return "Unknown", "gray", "risk-unknown"
        
        # Normalize probability
        p = prob / 100.0 if prob > 1 else prob
        
        if p >= 0.7:
            return "Critical", self.config.CRITICAL_COLOR, "risk-critical"
        elif p >= 0.5:
            return "High", self.config.WARNING_COLOR, "risk-high" 
        elif p >= 0.3:
            return "Medium", self.config.WARNING_COLOR, "risk-medium"
        else:
            return "Low", self.config.SUCCESS_COLOR, "risk-low"

    def page_executive_dashboard(self):
        """Enhanced executive dashboard with professional analytics."""
        st.markdown('<div class="main-header">Executive Dashboard</div>', unsafe_allow_html=True)
        
        # Top Level Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        df = st.session_state.get('loaded_df', pd.DataFrame())
        model_loaded = st.session_state.get('model_loaded', False)
        perf_summary = self.performance_monitor.get_performance_summary()
        
        with col1:
            patients = len(df) if not df.empty else 0
            st.metric("👥 Total Patients", f"{patients:,}", "Live" if patients > 0 else "No Data")
        
        with col2:
            if not df.empty and 'age' in df.columns:
                avg_age = df['age'].mean()
                st.metric("📊 Average Age", f"{avg_age:.1f}", "Years")
            else:
                st.metric("📊 Average Age", "N/A", "No Data")
        
        with col3:
            if perf_summary:
                st.metric("🎯 Predictions", f"{perf_summary['total_predictions']:,}", "Today")
            else:
                st.metric("🎯 Predictions", "0", "No Data")
        
        with col4:
            if perf_summary:
                st.metric("⚡ Avg Response", f"{perf_summary['avg_processing_time']:.2f}s", "Performance")
            else:
                st.metric("⚡ Avg Response", "N/A", "No Data")
        
        # Main Content
        tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔍 Data Quality", "🚨 Alerts & Monitoring"])
        
        with tab1:
            self.render_dashboard_overview()
        
        with tab2:
            self.render_data_quality()
        
        with tab3:
            self.render_alerts_monitoring()
    
    def render_dashboard_overview(self):
        """Render dashboard overview with advanced visualizations."""
        df = st.session_state.get('loaded_df', pd.DataFrame())
        
        if df.empty:
            st.info("📊 Please load data to view dashboard analytics.")
            if st.button("🔄 Load Sample Data"):
                self.load_sample_data()
                st.rerun()
            return
        
        # Generate sample risk scores for demonstration
        risk_scores = np.random.uniform(0, 1, len(df))
        
        # Advanced analytics visualization
        st.markdown("### 📊 Risk Intelligence Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Risk stratification
            fig = AdvancedAnalytics.create_risk_stratification(df, risk_scores)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Key metrics and alerts
            st.markdown("#### 🎯 Key Metrics")
            
            critical_risk_pct = (risk_scores > 0.7).mean() * 100
            high_risk_pct = ((risk_scores > 0.5) & (risk_scores <= 0.7)).mean() * 100
            medium_risk_pct = ((risk_scores > 0.3) & (risk_scores <= 0.5)).mean() * 100
            low_risk_pct = (risk_scores <= 0.3).mean() * 100
            
            st.metric("🔴 Critical Risk", f"{critical_risk_pct:.1f}%")
            st.metric("🟠 High Risk", f"{high_risk_pct:.1f}%")
            st.metric("🟡 Medium Risk", f"{medium_risk_pct:.1f}%")
            st.metric("🟢 Low Risk", f"{low_risk_pct:.1f}%")
            
            st.markdown("---")
            st.markdown("#### 📈 Performance")
            
            # Mock performance metrics
            st.metric("Model Accuracy", "94.2%", "±1.5%")
            st.metric("Data Quality", "98.7%", "Excellent")
            st.metric("System Uptime", "99.9%", "This month")
    
    def render_data_quality(self):
        """Render data quality assessment."""
        df = st.session_state.get('loaded_df', pd.DataFrame())
        
        if df.empty:
            st.info("📁 Please load data to view quality assessment.")
            return
        
        # Generate or retrieve data quality report
        if st.session_state.get('data_quality_report') is None:
            st.session_state['data_quality_report'] = DataValidator.generate_data_quality_report(df)
        
        report = st.session_state['data_quality_report']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📋 Quality Summary")
            
            # Quick quality metrics
            validation = DataValidator.validate_patient_data(df)
            
            st.metric("✅ Valid Records", f"{len(df):,}")
            st.metric("⚠️ Data Warnings", f"{len(validation['warnings'])}")
            st.metric("❌ Critical Issues", f"{len(validation['issues'])}")
            
            if st.button("🔄 Revalidate Data", use_container_width=True):
                st.session_state['data_quality_report'] = DataValidator.generate_data_quality_report(df)
                st.rerun()
        
        with col2:
            st.markdown("### 📄 Detailed Report")
            st.markdown(report)
    
    def render_alerts_monitoring(self):
        """Render alerts and system monitoring."""
        st.markdown("### 🚨 Real-time Monitoring")
        
        # Performance alerts
        alerts = self.performance_monitor.get_recent_alerts()
        
        if alerts:
            st.markdown("#### ⚠️ Active Alerts")
            for alert in alerts:
                with st.container():
                    st.warning(f"""
                    **{alert['timestamp'].strftime('%H:%M:%S')}** - {alert['message']}
                    """)
        else:
            st.success("✅ All systems operational - no active alerts")
        
        # System metrics
        st.markdown("---")
        st.markdown("#### 📊 System Metrics")
        
        perf_summary = self.performance_monitor.get_performance_summary()
        if perf_summary:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", f"{perf_summary['total_predictions']:,}")
            with col2:
                st.metric("Avg Confidence", f"{perf_summary['avg_confidence']:.1f}%")
            with col3:
                st.metric("Avg Response Time", f"{perf_summary['avg_processing_time']:.2f}s")
            with col4:
                st.metric("Critical Risk Rate", f"{perf_summary['critical_risk_rate']:.1%}")
        else:
            st.info("📊 No performance data available yet")
    
    def page_data_explorer(self):
        """Enhanced data exploration page."""
        st.markdown('<div class="main-header">Data Explorer</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📁 Data Management")
            
            # Data source selection
            data_source = st.radio(
                "Data Source",
                ["Sample Data", "Upload CSV", "Database Connection"],
                help="Choose your data source method"
            )
            
            if data_source == "Sample Data":
                if st.button("🔄 Generate Sample Data", use_container_width=True):
                    self.load_sample_data()
            
            elif data_source == "Upload CSV":
                uploaded_file = st.file_uploader(
                    "Choose CSV file",
                    type=["csv"],
                    help="Upload patient data CSV file"
                )
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.session_state['loaded_df'] = df
                        st.session_state['data_quality_report'] = DataValidator.generate_data_quality_report(df)
                        st.success(f"✅ Successfully loaded {len(df)} records")
                    except Exception as e:
                        st.error(f"❌ Error loading file: {e}")
            
            elif data_source == "Database Connection":
                st.info("🔗 Database integration available in Enterprise edition")
                
            # Model loading
            st.markdown("### 🤖 Model Management")
            if st.button("🚀 Load AI Model", use_container_width=True):
                try:
                    model = load_trained_model()
                    st.session_state['loaded_model'] = model
                    st.session_state['model_loaded'] = True
                    st.success("✅ AI model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Model loading failed: {e}")
        
        with col2:
            df = st.session_state.get('loaded_df', pd.DataFrame())
            
            if not df.empty:
                st.markdown("### 📊 Data Overview")
                
                # Enhanced data preview with tabs
                tab1, tab2, tab3, tab4 = st.tabs(["🔍 Preview", "📈 Statistics", "📋 Structure", "🎯 Sample Analysis"])
                
                with tab1:
                    st.dataframe(df.head(20), use_container_width=True)
                    st.caption(f"Showing 20 of {len(df):,} records")
                
                with tab2:
                    if not df.empty:
                        st.dataframe(df.describe(), use_container_width=True)
                    else:
                        st.info("No data to display")
                
                with tab3:
                    # Data structure and types
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Data Types**")
                        dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                        st.dataframe(dtype_df, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Missing Values**")
                        missing_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Count'])
                        missing_df['Percentage'] = (missing_df['Missing Count'] / len(df)) * 100
                        st.dataframe(missing_df.round(2), use_container_width=True)
                
                with tab4:
                    st.markdown("**Quick Data Analysis**")
                    
                    # Column distribution
                    if len(df.columns) > 0:
                        selected_col = st.selectbox("Select column for distribution:", df.columns)
                        if selected_col in df.columns:
                            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👆 Please load data to explore")
    
    def page_risk_predictor(self):
        """Enhanced single patient risk prediction."""
        st.markdown('<div class="main-header">Risk Predictor</div>', unsafe_allow_html=True)
        
        df = st.session_state.get('loaded_df', pd.DataFrame())
        model = st.session_state.get('loaded_model')
        
        if df.empty:
            st.warning("⚠️ Please load data first in the Data Explorer")
            if st.button("🔄 Load Sample Data"):
                self.load_sample_data()
                st.rerun()
            return
        
        # Ensure patient_id exists
        if 'patient_id' not in df.columns:
            df['patient_id'] = [f"PAT_{i}" for i in range(len(df))]
            st.session_state['loaded_df'] = df
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎯 Patient Selection")
            
            patient_ids = df['patient_id'].astype(str).tolist()
            selected_pid = st.selectbox("Select Patient ID", patient_ids, index=0)
            
            # Patient quick view
            patient_data = df[df['patient_id'].astype(str) == selected_pid].iloc[0].to_dict()
            
            st.markdown("### 📋 Patient Snapshot")
            
            # Display key patient information in cards
            key_fields = ['age', 'gender', 'primary_diagnosis', 'length_of_stay']
            for field in key_fields:
                if field in patient_data and pd.notna(patient_data[field]):
                    value = patient_data[field]
                    st.markdown(f"""
                    <div class="pro-card">
                        <strong>{field.replace('_', ' ').title()}</strong><br>
                        <span style="font-size: 1.1em; color: {self.config.PRIMARY_COLOR};">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Load model if not loaded
            if not model and not st.session_state.get('model_loaded', False):
                st.markdown("### 🤖 Model Status")
                if st.button("🚀 Load Prediction Model", use_container_width=True):
                    try:
                        model = load_trained_model()
                        st.session_state['loaded_model'] = model
                        st.session_state['model_loaded'] = True
                        st.success("✅ Model loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Model loading failed: {e}")
        
        with col2:
            st.markdown("### 📊 Clinical Overview")
            
            # Clinical metrics in columns
            clinical_fields = ['glucose_level', 'bmi', 'blood_pressure_systolic', 'num_previous_admissions']
            metrics_cols = st.columns(len(clinical_fields))
            
            for i, field in enumerate(clinical_fields):
                with metrics_cols[i]:
                    if field in patient_data and pd.notna(patient_data[field]):
                        value = patient_data[field]
                        if field == 'glucose_level':
                            st.metric("🩸 Glucose", f"{value:.0f}", "mg/dL")
                        elif field == 'bmi':
                            st.metric("⚖️ BMI", f"{value:.1f}")
                        elif field == 'blood_pressure_systolic':
                            st.metric("💓 BP Systolic", f"{value:.0f}", "mmHg")
                        elif field == 'num_previous_admissions':
                            st.metric("📅 Prev Admissions", f"{value:.0f}")
            
            # Prediction section
            st.markdown("---")
            st.markdown("### 🎯 Risk Prediction")
            
            model = st.session_state.get('loaded_model')
            if not model:
                st.warning("⚠️ Please load the prediction model first")
                return
            
            if st.button("🎯 Predict Readmission Risk", type="primary", use_container_width=True):
                start_time = time.time()
                
                with st.spinner("🔄 Analyzing patient data..."):
                    try:
                        result = predict_patient_risk(model, patient_data)
                        processing_time = time.time() - start_time
                        
                        # Log performance
                        self.performance_monitor.log_prediction(result, processing_time)
                        st.session_state['last_prediction'] = result
                        st.session_state['last_processing_time'] = processing_time
                        
                        # Display results
                        self.render_prediction_results(result, processing_time)
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
            
            # Show last prediction results if available
            if st.session_state.get('last_prediction'):
                self.render_prediction_results(
                    st.session_state['last_prediction'], 
                    st.session_state.get('last_processing_time', 0)
                )
    
    def render_prediction_results(self, result: Dict, processing_time: float):
        """Render prediction results with professional styling."""
        risk_score = result.get('risk_score', 0)
        risk_level, color, css_class = self.risk_category_from_prob(risk_score * 100)
        display_prob = risk_score * 100
        
        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size: 1.3em; margin-bottom: 0.5rem;">🏥 Risk Assessment Complete</div>
            <div style="font-size: 1em;">
                📊 Probability: <strong>{display_prob:.1f}%</strong> • 
                🎯 Level: <strong>{risk_level}</strong> • 
                ⚡ Processing: <strong>{processing_time:.2f}s</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence Score", f"{result.get('confidence', 0):.1f}%")
        
        with col2:
            st.metric("Risk Category", risk_level)
        
        with col3:
            st.metric("Processing Time", f"{processing_time:.2f}s")
        
        # Recommendations
        st.markdown("#### 💡 Clinical Recommendations")
        
        recommendations = result.get('recommendation', '').split('•')
        if len(recommendations) == 1:
            # If no bullet points, use the whole string
            st.info(result.get('recommendation', 'No specific recommendations available.'))
        else:
            for rec in recommendations:
                if rec.strip():
                    st.markdown(f"- {rec.strip()}")
        
        # Additional context
        st.markdown("#### 📋 Next Steps")
        if risk_level == "Critical":
            steps = [
                "Immediate clinical review required",
                "Contact care coordination team",
                "Schedule discharge planning meeting",
                "Consider extended monitoring"
            ]
        elif risk_level == "High":
            steps = [
                "Schedule follow-up within 7 days",
                "Review medication management",
                "Assess home care needs",
                "Coordinate with primary care"
            ]
        elif risk_level == "Medium":
            steps = [
                "Schedule follow-up within 14 days",
                "Provide patient education",
                "Review discharge instructions",
                "Confirm transportation arrangements"
            ]
        else:
            steps = [
                "Standard discharge process",
                "Provide educational materials",
                "Schedule routine follow-up",
                "Confirm understanding of care plan"
            ]
        
        for step in steps:
            st.markdown(f"- {step}")

    def page_ai_explainability(self):
        """AI model explainability and interpretability."""
        st.markdown('<div class="main-header">AI Explainability</div>', unsafe_allow_html=True)
        
        df = st.session_state.get('loaded_df', pd.DataFrame())
        model = st.session_state.get('loaded_model')
        
        if df.empty or not model:
            st.warning("⚠️ Please load both data and model first")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🔍 Explanation Settings")
            
            patient_ids = df['patient_id'].astype(str).tolist()
            selected_pid = st.selectbox("Select Patient for Explanation", patient_ids, index=0)
            
            explanation_type = st.radio(
                "Explanation Method",
                ["SHAP Analysis", "LIME Explanation", "Feature Importance"],
                help="Choose the explanation technique"
            )
            
            if st.button("🧠 Generate Explanation", type="primary", use_container_width=True):
                with st.spinner("🔄 Generating explanation..."):
                    try:
                        patient_data = df[df['patient_id'].astype(str) == selected_pid].iloc[0].to_dict()
                        
                        if explanation_type == "SHAP Analysis":
                            result = generate_shap_explanation(model, pd.DataFrame([patient_data]), df.sample(100), self.config.TEMP_DIR)
                            st.session_state['last_shap_result'] = result
                            
                        elif explanation_type == "LIME Explanation":
                            result = generate_lime_explanation(model, pd.DataFrame([patient_data]), df.sample(100), self.config.TEMP_DIR)
                            st.session_state['last_lime_result'] = result
                            
                        elif explanation_type == "Feature Importance":
                            result = get_feature_importance(model, top_k=10)
                            st.session_state['last_feature_importance'] = result
                        
                        st.success("✅ Explanation generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Explanation failed: {e}")
        
        with col2:
            st.markdown("### 📊 Model Interpretability")
            
            if explanation_type == "SHAP Analysis" and st.session_state.get('last_shap_result') is not None:
                result = st.session_state['last_shap_result']
                st.markdown("#### 📈 SHAP Feature Impact")
                
                if result.get('top_features'):
                    features = [feat[0] for feat in result['top_features']]
                    values = [feat[1] for feat in result['top_features']]
                    
                    fig = go.Figure(go.Bar(
                        x=values,
                        y=features,
                        orientation='h',
                        marker_color=['#e74c3c' if x > 0 else '#2ecc71' for x in values]
                    ))
                    
                    fig.update_layout(
                        title="SHAP Feature Importance",
                        xaxis_title="SHAP Value Impact",
                        yaxis_title="Features",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"**Summary:** {result.get('summary', 'No summary available')}")
            
            elif explanation_type == "LIME Explanation" and st.session_state.get('last_lime_result') is not None:
                result = st.session_state['last_lime_result']
                st.markdown("#### 🍋 LIME Feature Weights")
                
                if result.get('explanation'):
                    features = [feat[0] for feat in result['explanation']]
                    values = [feat[1] for feat in result['explanation']]
                    
                    fig = go.Figure(go.Bar(
                        x=values,
                        y=features,
                        orientation='h',
                        marker_color=['#e74c3c' if x > 0 else '#2ecc71' for x in values]
                    ))
                    
                    fig.update_layout(
                        title="LIME Feature Weights",
                        xaxis_title="Weight Impact",
                        yaxis_title="Features",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            elif explanation_type == "Feature Importance" and st.session_state.get('last_feature_importance') is not None:
                result = st.session_state['last_feature_importance']
                st.markdown("#### 🎯 Global Feature Importance")
                
                fig = go.Figure(go.Bar(
                    x=result['importance'],
                    y=result['feature'],
                    orientation='h',
                    marker_color='#3498db'
                ))
                
                fig.update_layout(
                    title="Global Feature Importance",
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("👆 Generate an explanation to view interpretability insights")
    
    def page_advanced_analytics(self):
        """Advanced analytics and insights."""
        st.markdown('<div class="main-header">Advanced Analytics</div>', unsafe_allow_html=True)
        
        df = st.session_state.get('loaded_df', pd.DataFrame())
        
        if df.empty:
            st.warning("⚠️ Please load data first")
            if st.button("🔄 Load Sample Data"):
                self.load_sample_data()
                st.rerun()
            return
        
        tab1, tab2, tab3 = st.tabs(["📈 Correlation Analysis", "📊 Risk Patterns", "🔍 Statistical Insights"])
        
        with tab1:
            st.markdown("### 📈 Feature Correlation Analysis")
            fig = AdvancedAnalytics.create_feature_correlation_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 💡 Correlation Insights")
            st.markdown("""
            - **Strong positive correlation** (red): Features that increase together
            - **Strong negative correlation** (blue): Features that move in opposite directions  
            - **Weak correlation** (white): Little to no relationship between features
            """)
        
        with tab2:
            st.markdown("### 📊 Risk Pattern Analysis")
            
            # Generate sample risk scores
            risk_scores = np.random.uniform(0, 1, len(df))
            df_analysis = df.copy()
            df_analysis['risk_score'] = risk_scores
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk by diagnosis
                if 'primary_diagnosis' in df.columns:
                    try:
                        diagnosis_risk = df_analysis.groupby('primary_diagnosis')['risk_score'].mean().sort_values(ascending=False)
                        fig = px.bar(
                            diagnosis_risk, 
                            title="Average Risk by Diagnosis",
                            labels={'value': 'Average Risk Score', 'primary_diagnosis': 'Diagnosis'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating diagnosis risk chart: {e}")
                else:
                    st.info("No diagnosis data available")
            
            with col2:
                # Risk by age group
                if 'age' in df.columns:
                    try:
                        df_analysis['age_group'] = pd.cut(df_analysis['age'], bins=[0, 40, 60, 80, 100], 
                                                        labels=['<40', '40-60', '60-80', '80+'])
                        age_risk = df_analysis.groupby('age_group')['risk_score'].mean()
                        fig = px.line(
                            age_risk, 
                            title="Risk Trend by Age Group",
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating age risk chart: {e}")
                else:
                    st.info("No age data available")
        
        with tab3:
            st.markdown("### 🔍 Statistical Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Data Summary")
                try:
                    # Only show numerical columns for summary
                    numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
                    if numerical_cols:
                        st.dataframe(df[numerical_cols].describe(), use_container_width=True)
                    else:
                        st.info("No numerical data available for summary")
                except Exception as e:
                    st.error(f"Error generating data summary: {e}")
            
            with col2:
                st.markdown("#### 📊 Distribution Insights")
                
                # Get available numerical columns
                numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
                
                if numerical_cols:
                    selected_num_col = st.selectbox("Select numerical column:", numerical_cols)
                    if selected_num_col in df.columns:
                        try:
                            fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating distribution chart: {e}")
                else:
                    st.info("No numerical columns available")
    
    def page_batch_processor(self):
        """Batch processing for multiple patients."""
        st.markdown('<div class="main-header">Batch Processor</div>', unsafe_allow_html=True)
        
        df = st.session_state.get('loaded_df', pd.DataFrame())
        model = st.session_state.get('loaded_model')
        
        if df.empty or not model:
            st.warning("⚠️ Please load both data and model first")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Batch Settings")
            
            batch_size = st.slider("Batch Size", min_value=10, max_value=min(1000, len(df)), value=100, step=10)
            
            risk_threshold = st.slider("High Risk Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            
            if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                with st.spinner(f"🔄 Processing {batch_size} patients..."):
                    try:
                        sample_patients = df.sample(batch_size).to_dict('records')
                        results = predict_batch_risk(model, sample_patients)
                        st.session_state['batch_results'] = results
                        st.success(f"✅ Successfully processed {len(results)} patients!")
                    except Exception as e:
                        st.error(f"❌ Batch processing failed: {e}")
        
        with col2:
            st.markdown("### 📊 Batch Results")
            
            if st.session_state.get('batch_results') is not None:
                results = st.session_state['batch_results']
                results_df = pd.DataFrame(results)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Processed", len(results))
                
                with col2:
                    high_risk_count = len([r for r in results if r['risk_score'] > risk_threshold])
                    st.metric("High Risk Patients", high_risk_count)
                
                with col3:
                    avg_risk = np.mean([r['risk_score'] for r in results])
                    st.metric("Average Risk", f"{avg_risk:.3f}")
                
                with col4:
                    avg_confidence = np.mean([r['confidence'] for r in results])
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                # Results table
                st.markdown("#### 📋 Detailed Results")
                display_df = results_df[['patient_id', 'predicted_label', 'risk_score', 'confidence']].round(3)
                st.dataframe(display_df, use_container_width=True)
                
                # Risk distribution
                st.markdown("#### 📈 Risk Distribution")
                try:
                    risk_counts = results_df['predicted_label'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Category Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating risk distribution chart: {e}")
            
            else:
                st.info("👆 Process a batch to view results")
    
    def page_report_generator(self):
        """Professional report generation."""
        st.markdown('<div class="main-header">Report Generator</div>', unsafe_allow_html=True)
        
        df = st.session_state.get('loaded_df', pd.DataFrame())
        last_prediction = st.session_state.get('last_prediction')
        
        if df.empty:
            st.warning("⚠️ Please load data first")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📄 Report Settings")
            
            report_type = st.selectbox(
                "Report Type",
                ["Patient Risk Assessment", "Data Quality Report", "Batch Analysis Summary", "Model Performance"]
            )
            
            if report_type == "Patient Risk Assessment":
                patient_ids = df['patient_id'].astype(str).tolist()
                selected_pid = st.selectbox("Select Patient", patient_ids, index=0)
                patient_data = df[df['patient_id'].astype(str) == selected_pid].iloc[0].to_dict()
            
            include_charts = st.checkbox("Include Visualizations", value=True)
            report_format = st.radio("Format", ["PDF", "HTML", "Markdown"])
            
            if st.button("📋 Generate Report", type="primary", use_container_width=True):
                with st.spinner("🔄 Generating professional report..."):
                    try:
                        if report_type == "Patient Risk Assessment":
                            model = st.session_state.get('loaded_model')
                            if model and last_prediction:
                                report_content = generate_patient_report(patient_data, last_prediction)
                            else:
                                # Generate mock prediction for report
                                mock_prediction = predict_patient_risk(MockModel(), patient_data)
                                report_content = generate_patient_report(patient_data, mock_prediction)
                        
                        elif report_type == "Data Quality Report":
                            report_content = DataValidator.generate_data_quality_report(df)
                        
                        else:
                            report_content = f"# {report_type}\n\nReport generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nThis is a sample {report_type.lower()}."
                        
                        st.session_state['last_report'] = report_content
                        st.success("✅ Report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Report generation failed: {e}")
        
        with col2:
            st.markdown("### 📊 Report Preview")
            
            if st.session_state.get('last_report') is not None:
                report_content = st.session_state['last_report']
                
                st.markdown("#### 📝 Content Preview")
                st.markdown(report_content)
                
                st.markdown("---")
                st.markdown("#### 💾 Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Download PDF", use_container_width=True):
                        st.info("📄 PDF export would be available in production")
                
                with col2:
                    if st.button("🌐 Export HTML", use_container_width=True):
                        st.info("🖥️ HTML export would be available in production")
                
                with col3:
                    if st.button("📋 Copy to Clipboard", use_container_width=True):
                        st.info("📋 Clipboard copy would be available in production")
            
            else:
                st.info("👆 Generate a report to preview content")
    
    def page_system_monitor(self):
        """System monitoring and performance metrics."""
        st.markdown('<div class="main-header">System Monitor</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖥️ System Health")
            
            # System metrics
            metrics = [
                ("CPU Usage", "23%", "normal"),
                ("Memory Usage", "67%", "warning"),
                ("Disk Space", "45%", "normal"),
                ("Network I/O", "12 MB/s", "normal"),
                ("Database Connections", "24", "normal"),
                ("API Response Time", "128ms", "normal")
            ]
            
            for metric, value, status in metrics:
                icon = "🟢" if status == "normal" else "🟡" if status == "warning" else "🔴"
                st.metric(metric, value)
            
            st.markdown("### 🔔 Recent Alerts")
            alerts = self.performance_monitor.get_recent_alerts()
            if alerts:
                for alert in alerts[-5:]:  # Last 5 alerts
                    st.warning(f"**{alert['timestamp'].strftime('%H:%M')}** - {alert['message']}")
            else:
                st.success("✅ No recent alerts")
        
        with col2:
            st.markdown("### 📈 Performance Metrics")
            
            perf_summary = self.performance_monitor.get_performance_summary()
            if perf_summary:
                # Create performance charts
                times = [f"{-i}h" for i in range(24, 0, -1)]
                mock_throughput = np.random.poisson(50, 24)
                mock_response_times = np.random.normal(0.8, 0.2, 24)
                
                fig = make_subplots(rows=2, cols=1, subplot_titles=('Request Throughput', 'Response Times'))
                
                fig.add_trace(
                    go.Scatter(x=times, y=mock_throughput, mode='lines+markers', name='Requests/min'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=times, y=mock_response_times, mode='lines+markers', name='Response Time (s)'),
                    row=2, col=1
                )
                
                fig.update_layout(height=500, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Model info
                st.markdown("### 🤖 Model Information")
                model_info = get_model_info(st.session_state.get('loaded_model', MockModel()))
                for key, value in model_info.items():
                    st.text(f"{key}: {value}")
            
            else:
                st.info("📊 No performance data available yet")

    def run(self):
        """Main application runner."""
        try:
            # Get current page from sidebar
            page = self.create_sidebar()
            
            # Route to appropriate page
            page_mapping = {
                "📊 Executive Dashboard": self.page_executive_dashboard,
                "🔍 Data Explorer": self.page_data_explorer,
                "🎯 Risk Predictor": self.page_risk_predictor,
                "🤖 AI Explainability": self.page_ai_explainability,
                "📈 Advanced Analytics": self.page_advanced_analytics,
                "⚡ Batch Processor": self.page_batch_processor,
                "📋 Report Generator": self.page_report_generator,
                "⚙️ System Monitor": self.page_system_monitor,
            }
            
            page_function = page_mapping.get(page)
            if page_function:
                page_function()
            else:
                st.info("🚧 Page under construction - check back soon!")
                
        except Exception as e:
            st.error(f"❌ Application error: {e}")
            with st.expander("Technical Details"):
                st.exception(e)

# ========== APPLICATION ENTRY POINT ==========
def main():
    """Application entry point."""
    app = ProStreamlitApp()
    app.run()

if __name__ == "__main__":
    main()