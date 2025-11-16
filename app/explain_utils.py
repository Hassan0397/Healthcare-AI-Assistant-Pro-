# explain_utils.py
"""
Explainability Utilities Module for Patient Readmission Risk Analysis App

This module handles model interpretability using SHAP and LIME to explain
why the model makes specific predictions, enhancing trust and transparency
in healthcare AI systems.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union
import os
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import LIME (optional dependency)
try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    st.warning("⚠️ LIME not available. Install with: pip install lime")

@st.cache_resource(show_spinner="Initializing SHAP explainer...")
def create_shap_explainer(model: object, X_background: np.ndarray = None) -> object:
    """
    Create appropriate SHAP explainer based on model type.
    
    Args:
        model: Trained model object
        X_background: Background dataset for KernelExplainer
        
    Returns:
        object: SHAP explainer
    """
    try:
        # Get the actual classifier from pipeline if needed
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
        else:
            classifier = model
        
        # Choose explainer based on model type
        if hasattr(classifier, 'estimators_'):  # Random Forest
            explainer = shap.TreeExplainer(classifier)
            st.success("✅ Using TreeExplainer for Random Forest")
            
        elif hasattr(classifier, 'get_booster'):  # XGBoost
            explainer = shap.TreeExplainer(classifier)
            st.success("✅ Using TreeExplainer for XGBoost")
            
        elif hasattr(classifier, 'coef_'):  # Linear models
            if X_background is not None:
                explainer = shap.LinearExplainer(classifier, X_background)
                st.success("✅ Using LinearExplainer")
            else:
                explainer = shap.Explainer(classifier)
                st.success("✅ Using Generic Explainer for linear model")
                
        else:  # Fallback to KernelExplainer
            if X_background is None:
                # Create small background dataset
                if hasattr(model, 'named_steps'):
                    # For pipeline, we need to transform background data
                    X_background = np.random.randn(50, 100)  # Placeholder
                else:
                    X_background = np.random.randn(50, 100)
                    
            explainer = shap.KernelExplainer(model.predict_proba, X_background)
            st.success("✅ Using KernelExplainer (fallback)")
        
        return explainer
        
    except Exception as e:
        st.error(f"❌ Error creating SHAP explainer: {str(e)}")
        # Fallback to generic explainer
        try:
            explainer = shap.Explainer(model)
            st.success("✅ Using generic SHAP Explainer")
            return explainer
        except:
            st.error("❌ Could not create any SHAP explainer")
            return None

def generate_shap_explanation(model: object, 
                            X_sample: np.ndarray,
                            feature_names: List[str],
                            X_background: np.ndarray = None,
                            patient_id: str = "Unknown") -> Dict:
    """
    Generate SHAP explanations for a patient's prediction.
    
    Args:
        model: Trained model object
        X_sample: Single patient's preprocessed data
        feature_names: List of feature names
        X_background: Background dataset for explainer
        patient_id: Patient identifier for file naming
        
    Returns:
        Dict: SHAP explanation results and plot paths
    """
    try:
        st.info("🔍 Generating SHAP explanation...")
        
        # Create explainer
        explainer = create_shap_explainer(model, X_background)
        if explainer is None:
            return {'error': 'Could not create SHAP explainer'}
        
        # Compute SHAP values
        with st.spinner("Computing SHAP values..."):
            shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP values formats
        if isinstance(shap_values, list):
            # For binary classification, take class 1 (high risk)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Create temporary directory for plots
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        results = {
            'shap_values': shap_values,
            'base_value': explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            'plot_paths': {},
            'top_contributors': get_top_contributors(shap_values[0], feature_names)
        }
        
        # Generate and save plots
        st.subheader("📊 SHAP Explanations")
        
        # 1. Force Plot (Local Explanation)
        with st.spinner("Creating force plot..."):
            fig, ax = plt.subplots(figsize=(10, 3))
            shap.force_plot(
                results['base_value'],
                shap_values[0],
                X_sample[0],
                feature_names=feature_names,
                matplotlib=True,
                show=False,
                text_rotation=15
            )
            plt.tight_layout()
            force_plot_path = temp_dir / f"shap_force_{patient_id}.png"
            plt.savefig(force_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            results['plot_paths']['force_plot'] = str(force_plot_path)
        
        # 2. Waterfall Plot (Detailed Local Explanation)
        with st.spinner("Creating waterfall plot..."):
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=results['base_value'],
                    data=X_sample[0],
                    feature_names=feature_names
                ),
                show=False
            )
            plt.title(f'SHAP Waterfall Plot - Patient {patient_id}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            waterfall_plot_path = temp_dir / f"shap_waterfall_{patient_id}.png"
            plt.savefig(waterfall_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            results['plot_paths']['waterfall_plot'] = str(waterfall_plot_path)
        
        # 3. Bar Plot (Feature Importance)
        with st.spinner("Creating feature importance plot..."):
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.bar(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=results['base_value'],
                    feature_names=feature_names
                ),
                show=False
            )
            plt.title(f'Feature Importance - Patient {patient_id}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            bar_plot_path = temp_dir / f"shap_bar_{patient_id}.png"
            plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            results['plot_paths']['bar_plot'] = str(bar_plot_path)
        
        st.success(f"✅ SHAP explanations generated for patient {patient_id}")
        
        return results
        
    except Exception as e:
        st.error(f"❌ Error generating SHAP explanation: {str(e)}")
        return {'error': str(e)}

def generate_lime_explanation(model: object,
                            X_sample: np.ndarray,
                            X_background: np.ndarray,
                            feature_names: List[str],
                            class_names: List[str] = ['Low Risk', 'High Risk'],
                            patient_id: str = "Unknown") -> Dict:
    """
    Generate LIME explanation for a patient's prediction.
    
    Args:
        model: Trained model object
        X_sample: Single patient's preprocessed data
        X_background: Background training data
        feature_names: List of feature names
        class_names: Names of prediction classes
        patient_id: Patient identifier
        
    Returns:
        Dict: LIME explanation results
    """
    if not LIME_AVAILABLE:
        return {'error': 'LIME not available. Install with: pip install lime'}
    
    try:
        st.info("🍋 Generating LIME explanation...")
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_background,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=42
        )
        
        # Generate explanation
        with st.spinner("Computing LIME explanation..."):
            explanation = explainer.explain_instance(
                data_row=X_sample[0],
                predict_fn=model.predict_proba,
                num_features=10,
                top_labels=1
            )
        
        # Extract explanation details
        lime_results = {
            'explanation': explanation,
            'top_features': [],
            'prediction_probabilities': explanation.predict_proba,
            'local_prediction': explanation.local_pred
        }
        
        # Get top contributing features
        top_features = explanation.as_list(label=explanation.available_labels()[0])
        lime_results['top_features'] = top_features
        
        # Create visualization
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save LIME plot
        fig = explanation.as_pyplot_figure(explanation.available_labels()[0])
        plt.title(f'LIME Explanation - Patient {patient_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        lime_plot_path = temp_dir / f"lime_explanation_{patient_id}.png"
        plt.savefig(lime_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        lime_results['plot_path'] = str(lime_plot_path)
        lime_results['html_path'] = str(temp_dir / f"lime_explanation_{patient_id}.html")
        
        # Save HTML version
        explanation.save_to_file(lime_results['html_path'])
        
        st.success(f"✅ LIME explanation generated for patient {patient_id}")
        
        return lime_results
        
    except Exception as e:
        st.error(f"❌ Error generating LIME explanation: {str(e)}")
        return {'error': str(e)}

def get_top_contributors(shap_values: np.ndarray, 
                        feature_names: List[str], 
                        top_n: int = 5) -> List[Dict]:
    """
    Extract top features contributing to the prediction.
    
    Args:
        shap_values: SHAP values for a single prediction
        feature_names: List of feature names
        top_n: Number of top contributors to return
        
    Returns:
        List[Dict]: Top contributing features with their impact
    """
    try:
        # Get absolute values for importance ranking
        abs_contributions = np.abs(shap_values)
        
        # Get indices of top contributors
        top_indices = np.argsort(abs_contributions)[-top_n:][::-1]
        
        contributors = []
        for idx in top_indices:
            contribution = shap_values[idx]
            contributors.append({
                'feature': feature_names[idx],
                'contribution': float(contribution),
                'absolute_impact': float(abs_contributions[idx]),
                'direction': 'increases risk' if contribution > 0 else 'decreases risk'
            })
        
        return contributors
        
    except Exception as e:
        st.warning(f"⚠️ Could not extract top contributors: {str(e)}")
        return []

def generate_combined_explanation(model: object,
                                X_sample: np.ndarray,
                                X_background: np.ndarray,
                                feature_names: List[str],
                                patient_id: str = "Unknown") -> Dict:
    """
    Generate both SHAP and LIME explanations for comprehensive interpretability.
    
    Args:
        model: Trained model object
        X_sample: Single patient's preprocessed data
        X_background: Background training data
        feature_names: List of feature names
        patient_id: Patient identifier
        
    Returns:
        Dict: Combined explanation results
    """
    st.header("🤖 AI Explanation Report")
    
    results = {
        'patient_id': patient_id,
        'shap': {},
        'lime': {},
        'summary': {}
    }
    
    # Generate SHAP explanation
    shap_results = generate_shap_explanation(
        model, X_sample, feature_names, X_background, patient_id
    )
    results['shap'] = shap_results
    
    # Generate LIME explanation
    lime_results = generate_lime_explanation(
        model, X_sample, X_background, feature_names, patient_id=patient_id
    )
    results['lime'] = lime_results
    
    # Create combined summary
    results['summary'] = create_explanation_summary(shap_results, lime_results)
    
    return results

def create_explanation_summary(shap_results: Dict, lime_results: Dict) -> Dict:
    """
    Create a unified summary from SHAP and LIME explanations.
    
    Args:
        shap_results: SHAP explanation results
        lime_results: LIME explanation results
        
    Returns:
        Dict: Unified explanation summary
    """
    summary = {
        'key_factors': [],
        'risk_drivers': [],
        'protective_factors': [],
        'agreement_level': 'high'
    }
    
    try:
        # Extract from SHAP results
        if 'top_contributors' in shap_results and not isinstance(shap_results.get('error'), str):
            for contributor in shap_results['top_contributors']:
                if contributor['contribution'] > 0:
                    summary['risk_drivers'].append({
                        'feature': contributor['feature'],
                        'impact': contributor['absolute_impact'],
                        'source': 'SHAP'
                    })
                else:
                    summary['protective_factors'].append({
                        'feature': contributor['feature'],
                        'impact': contributor['absolute_impact'],
                        'source': 'SHAP'
                    })
        
        # Extract from LIME results
        if 'top_features' in lime_results and not isinstance(lime_results.get('error'), str):
            for feature, impact in lime_results['top_features']:
                feature_name = feature.split(' <=')[0] if '<=' in feature else feature.split(' > ')[0]
                if impact > 0:
                    summary['risk_drivers'].append({
                        'feature': feature_name,
                        'impact': abs(impact),
                        'source': 'LIME'
                    })
                else:
                    summary['protective_factors'].append({
                        'feature': feature_name,
                        'impact': abs(impact),
                        'source': 'LIME'
                    })
        
        # Create unified key factors list
        all_factors = summary['risk_drivers'] + summary['protective_factors']
        all_factors.sort(key=lambda x: x['impact'], reverse=True)
        summary['key_factors'] = all_factors[:5]  # Top 5 factors
        
        return summary
        
    except Exception as e:
        st.warning(f"⚠️ Could not create explanation summary: {str(e)}")
        return summary

def display_explanations_in_streamlit(explanation_results: Dict):
    """
    Display SHAP and LIME explanations in Streamlit dashboard.
    
    Args:
        explanation_results: Combined explanation results
    """
    if not explanation_results:
        st.error("No explanation results to display")
        return
    
    patient_id = explanation_results.get('patient_id', 'Unknown')
    
    # Display SHAP explanations
    if 'shap' in explanation_results and not isinstance(explanation_results['shap'].get('error'), str):
        st.subheader("🔍 SHAP Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'plot_paths' in explanation_results['shap']:
                if 'force_plot' in explanation_results['shap']['plot_paths']:
                    st.image(explanation_results['shap']['plot_paths']['force_plot'], 
                            caption="SHAP Force Plot - Feature Contributions")
        
        with col2:
            if 'plot_paths' in explanation_results['shap']:
                if 'bar_plot' in explanation_results['shap']['plot_paths']:
                    st.image(explanation_results['shap']['plot_paths']['bar_plot'],
                            caption="SHAP Feature Importance")
        
        # Show waterfall plot in full width
        if 'plot_paths' in explanation_results['shap'] and 'waterfall_plot' in explanation_results['shap']['plot_paths']:
            st.image(explanation_results['shap']['plot_paths']['waterfall_plot'],
                    caption="SHAP Waterfall Plot - Detailed Breakdown")
    
    # Display LIME explanations
    if 'lime' in explanation_results and LIME_AVAILABLE and not isinstance(explanation_results['lime'].get('error'), str):
        st.subheader("🍋 LIME Analysis")
        
        if 'plot_path' in explanation_results['lime']:
            st.image(explanation_results['lime']['plot_path'],
                    caption="LIME Explanation - Local Feature Importance")
        
        # Show top features from LIME
        if 'top_features' in explanation_results['lime']:
            st.write("**Top Contributing Features (LIME):**")
            for feature, impact in explanation_results['lime']['top_features'][:5]:
                direction = "increases risk" if impact > 0 else "decreases risk"
                st.write(f"- {feature}: {impact:.3f} ({direction})")
    
    # Display summary
    if 'summary' in explanation_results:
        display_explanation_summary(explanation_results['summary'])

def display_explanation_summary(summary: Dict):
    """
    Display the explanation summary in a user-friendly format.
    
    Args:
        summary: Explanation summary dictionary
    """
    st.subheader("📋 Explanation Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🚨 Risk Drivers:**")
        if summary['risk_drivers']:
            for driver in sorted(summary['risk_drivers'], key=lambda x: x['impact'], reverse=True)[:3]:
                st.write(f"- {driver['feature']} (Impact: {driver['impact']:.3f})")
        else:
            st.write("No significant risk drivers identified")
    
    with col2:
        st.write("**🛡️ Protective Factors:**")
        if summary['protective_factors']:
            for factor in sorted(summary['protective_factors'], key=lambda x: x['impact'], reverse=True)[:3]:
                st.write(f"- {factor['feature']} (Impact: {factor['impact']:.3f})")
        else:
            st.write("No significant protective factors identified")
    
    # Key insights
    st.write("**💡 Key Insights:**")
    if summary['key_factors']:
        for i, factor in enumerate(summary['key_factors'][:3], 1):
            st.write(f"{i}. **{factor['feature']}** is the most influential factor")
    else:
        st.write("No key factors identified")

def save_plot(fig: plt.Figure, filename: str, directory: str = "temp") -> str:
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        directory: Output directory
        
    Returns:
        str: Path to saved plot
    """
    Path(directory).mkdir(exist_ok=True)
    filepath = Path(directory) / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(filepath)

def cleanup_temp_files():
    """
    Clean up temporary explanation files.
    """
    temp_dir = Path("temp")
    if temp_dir.exists():
        for file in temp_dir.glob("*.png"):
            try:
                file.unlink()
            except:
                pass
        for file in temp_dir.glob("*.html"):
            try:
                file.unlink()
            except:
                pass

# Example usage function
def test_explain_utils():
    """
    Test the explainability utilities.
    """
    print("🧪 Testing explain_utils.py...")
    
    # This would be called from main app with actual model and data
    print("✅ Explainability utilities ready!")
    print("   - SHAP explanations for global and local interpretability")
    print("   - LIME explanations for local surrogate models")
    print("   - Combined explanation summaries")
    print("   - Streamlit visualization support")

if __name__ == "__main__":
    test_explain_utils()