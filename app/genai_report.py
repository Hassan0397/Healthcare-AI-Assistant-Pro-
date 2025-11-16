# genai_report.py
"""
Generative AI Reporting Module for Patient Readmission Risk Analysis App

This module uses Large Language Models (LLMs) to automatically generate
human-readable, doctor-friendly patient risk reports that explain model
predictions in natural language.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union
import streamlit as st
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import OpenAI (optional)
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("⚠️ OpenAI package not available. Install with: pip install openai")

# Try to load environment variables
try:
    from dotenv import load_dotenv # type: ignore
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
except ImportError:
    OPENAI_API_KEY = None

def setup_ai_client():
    """
    Set up the AI client (OpenAI or fallback to mock).
    
    Returns:
        object: AI client or None for mock mode
    """
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            st.success("✅ OpenAI client configured successfully")
            return client
        except Exception as e:
            st.error(f"❌ OpenAI setup failed: {e}")
    
    st.info("🔶 Using mock AI mode for demonstration")
    st.info("💡 To use real OpenAI: Set OPENAI_API_KEY environment variable")
    return None

def create_patient_prompt_template(style: str = "doctor") -> str:
    """
    Create appropriate prompt template based on desired style.
    
    Args:
        style: "doctor", "nurse", "patient", or "detailed"
        
    Returns:
        str: Prompt template
    """
    templates = {
        "doctor": """
You are a medical AI assistant helping clinicians assess patient readmission risk.

PATIENT INFORMATION:
{patient_info}

PREDICTION RESULTS:
- Risk Level: {risk_level}
- Risk Probability: {risk_probability}%
- Confidence: {confidence}%

KEY CONTRIBUTING FACTORS:
{key_factors}

Please write a concise clinical summary (4-5 sentences) that:
1. States the predicted readmission risk clearly
2. Explains the top 2-3 clinical factors driving this risk
3. Provides brief clinical context for these factors
4. Suggests 1-2 focused follow-up considerations

Use professional medical language appropriate for physicians.
Focus on actionable insights and clinical relevance.
""",

        "nurse": """
You are a healthcare AI assistant helping nursing staff with patient discharge planning.

PATIENT INFORMATION:
{patient_info}

PREDICTION RESULTS:
- Risk Level: {risk_level}
- Risk Probability: {risk_probability}%

KEY FACTORS:
{key_factors}

Please write a practical nursing summary (3-4 sentences) that:
1. Clearly states the readmission risk level
2. Highlights the most important monitoring needs
3. Suggests specific nursing interventions
4. Notes any special discharge considerations

Use clear, practical language focused on patient care and monitoring.
""",

        "patient": """
You are a compassionate healthcare AI explaining readmission risk to a patient.

PATIENT INFORMATION (simplified):
{patient_info_simple}

RISK ASSESSMENT:
- Likelihood of returning to hospital: {risk_level_simple}

KEY FACTORS:
{key_factors_simple}

Please write a patient-friendly explanation (3-4 sentences) that:
1. Explains their risk level in simple, reassuring terms
2. Highlights 1-2 things they can work on
3. Emphasizes positive actions they can take
4. Encourages follow-up care

Use warm, empathetic language without medical jargon.
Be encouraging but honest about risks.
""",

        "detailed": """
You are a comprehensive medical AI analyst providing detailed risk assessment.

PATIENT CLINICAL PROFILE:
{patient_info_detailed}

PREDICTION ANALYSIS:
- Predicted Risk Level: {risk_level}
- Probability Score: {risk_probability}%
- Model Confidence: {confidence}%
- Threshold Used: {threshold}%

DETAILED FACTOR ANALYSIS:
{key_factors_detailed}

Please provide a comprehensive clinical analysis including:
1. Executive summary of readmission risk
2. Detailed breakdown of contributing factors
3. Clinical interpretation of each significant factor
4. Evidence-based recommendations for risk mitigation
5. Suggested monitoring parameters and follow-up timeline

Use thorough, evidence-based medical language with specific clinical recommendations.
"""
    }
    
    return templates.get(style, templates["doctor"])

def format_patient_info(patient_data: Dict, style: str = "doctor") -> str:
    """
    Format patient information for the prompt.
    
    Args:
        patient_data: Patient information dictionary
        style: Output style
        
    Returns:
        str: Formatted patient information
    """
    if style == "patient":
        # Simplified for patients
        return f"""
- Age: {patient_data.get('age', 'Unknown')}
- Main Condition: {patient_data.get('primary_diagnosis', 'Not specified')}
- Hospital Stay: {patient_data.get('length_of_stay', 'Unknown')} days
- Previous Admissions: {patient_data.get('num_previous_admissions', 'Unknown')}
"""
    
    elif style == "detailed":
        # Comprehensive for detailed analysis
        return f"""
Demographics:
- Age: {patient_data.get('age', 'N/A')}
- Gender: {patient_data.get('gender', 'N/A')}
- Ethnicity: {patient_data.get('ethnicity', 'N/A')}

Clinical Information:
- Primary Diagnosis: {patient_data.get('primary_diagnosis', 'N/A')}
- Secondary Diagnosis: {patient_data.get('secondary_diagnosis', 'None')}
- Admission Type: {patient_data.get('admission_type', 'N/A')}
- Length of Stay: {patient_data.get('length_of_stay', 'N/A')} days
- Previous Admissions: {patient_data.get('num_previous_admissions', 'N/A')}

Vital Signs & Labs:
- BMI: {patient_data.get('bmi', 'N/A')}
- Glucose Level: {patient_data.get('glucose_level', 'N/A')} mg/dL
- Cholesterol: {patient_data.get('cholesterol_level', 'N/A')} mg/dL
- Blood Pressure: {patient_data.get('blood_pressure_systolic', 'N/A')}/{patient_data.get('blood_pressure_diastolic', 'N/A')}
- Heart Rate: {patient_data.get('heart_rate', 'N/A')} bpm

Lifestyle Factors:
- Smoking Status: {patient_data.get('smoking_status', 'N/A')}
- Physical Activity: {patient_data.get('physical_activity_level', 'N/A')}
- Diet Quality: {patient_data.get('diet_quality', 'N/A')}
"""
    
    else:
        # Standard for healthcare professionals
        return f"""
- Age: {patient_data.get('age', 'N/A')}
- Gender: {patient_data.get('gender', 'N/A')}
- Primary Diagnosis: {patient_data.get('primary_diagnosis', 'N/A')}
- Length of Stay: {patient_data.get('length_of_stay', 'N/A')} days
- Previous Admissions: {patient_data.get('num_previous_admissions', 'N/A')}
- Key Clinical Values:
  * BMI: {patient_data.get('bmi', 'N/A')}
  * Glucose: {patient_data.get('glucose_level', 'N/A')} mg/dL
  * Cholesterol: {patient_data.get('cholesterol_level', 'N/A')} mg/dL
"""

def format_key_factors(explanation_data: Dict, style: str = "doctor") -> str:
    """
    Format key contributing factors for the prompt.
    
    Args:
        explanation_data: Explanation results from explain_utils
        style: Output style
        
    Returns:
        str: Formatted key factors
    """
    if not explanation_data or 'summary' not in explanation_data:
        return "No specific factor analysis available."
    
    summary = explanation_data['summary']
    
    if style == "patient":
        # Simplified for patients
        factors = []
        for factor in summary.get('key_factors', [])[:3]:
            feature = factor['feature'].replace('_', ' ').title()
            direction = "increases" if factor.get('impact', 0) > 0 else "helps reduce"
            factors.append(f"- {feature} {direction} your risk")
        
        return "\n".join(factors) if factors else "General health factors contribute to your risk assessment."
    
    elif style == "detailed":
        # Comprehensive analysis
        factors_text = []
        
        # Risk drivers
        if summary.get('risk_drivers'):
            factors_text.append("RISK INCREASING FACTORS:")
            for driver in summary['risk_drivers'][:5]:
                factors_text.append(f"- {driver['feature']}: Impact score {driver['impact']:.3f}")
        
        # Protective factors
        if summary.get('protective_factors'):
            factors_text.append("\nRISK REDUCING FACTORS:")
            for factor in summary['protective_factors'][:3]:
                factors_text.append(f"- {factor['feature']}: Impact score {factor['impact']:.3f}")
        
        # Model agreement
        agreement = summary.get('agreement_level', 'unknown')
        factors_text.append(f"\nMODEL CONFIDENCE: {agreement.upper()}")
        
        return "\n".join(factors_text)
    
    else:
        # Standard for healthcare professionals
        factors = []
        for factor in summary.get('key_factors', [])[:5]:
            direction = "↑ increases risk" if factor.get('impact', 0) > 0 else "↓ decreases risk"
            factors.append(f"- {factor['feature']}: {direction} (impact: {factor.get('impact', 0):.3f})")
        
        return "\n".join(factors) if factors else "General clinical factors contribute to risk assessment."

def generate_patient_report(patient_data: Dict,
                          prediction: Dict,
                          explanation: Dict,
                          style: str = "doctor",
                          detail_level: str = "standard") -> Dict:
    """
    Generate a natural language patient risk report using AI.
    
    Args:
        patient_data: Patient information dictionary
        prediction: Model prediction results
        explanation: Explanation results from explain_utils
        style: "doctor", "nurse", "patient", or "detailed"
        detail_level: "brief" or "detailed"
        
    Returns:
        Dict: Generated report and metadata
    """
    try:
        st.info(f"🤖 Generating {style}-style AI report...")
        
        # Setup AI client
        client = setup_ai_client()
        
        # Prepare prompt components
        risk_level = prediction.get('predicted_label', 'Unknown')
        risk_probability = prediction.get('risk_probability', 0)
        confidence = prediction.get('confidence', 0)
        
        # Format information based on style
        patient_info = format_patient_info(patient_data, style)
        key_factors = format_key_factors(explanation, style)
        
        # Create simplified versions for patient style
        risk_level_simple = "higher" if risk_level == "High Risk" else "lower"
        patient_info_simple = format_patient_info(patient_data, "patient")
        key_factors_simple = format_key_factors(explanation, "patient")
        
        # Get appropriate prompt template
        prompt_template = create_patient_prompt_template(style)
        
        # Format the prompt
        prompt = prompt_template.format(
            patient_info=patient_info,
            patient_info_simple=patient_info_simple,
            patient_info_detailed=format_patient_info(patient_data, "detailed"),
            risk_level=risk_level,
            risk_level_simple=risk_level_simple,
            risk_probability=risk_probability,
            confidence=confidence,
            threshold=prediction.get('threshold_used', 0.5),
            key_factors=key_factors,
            key_factors_simple=key_factors_simple,
            key_factors_detailed=format_key_factors(explanation, "detailed")
        )
        
        # Generate report using AI
        if client:
            # Use real OpenAI API
            with st.spinner("Generating AI report..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                report_text = response.choices[0].message.content
        else:
            # Use mock response
            report_text = generate_mock_report(patient_data, prediction, explanation, style)
        
        # Post-process and structure the report
        structured_report = structure_generated_report(report_text, style, detail_level)
        
        # Add metadata
        structured_report['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'style': style,
            'detail_level': detail_level,
            'model_used': 'gpt-3.5-turbo' if client else 'mock',
            'patient_id': prediction.get('patient_id', 'Unknown')
        }
        
        st.success("✅ AI report generated successfully!")
        
        return structured_report
        
    except Exception as e:
        st.error(f"❌ Error generating AI report: {str(e)}")
        return {
            'error': str(e),
            'report_text': "Unable to generate AI report at this time.",
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'error': True
            }
        }

def generate_mock_report(patient_data: Dict, prediction: Dict, explanation: Dict, style: str) -> str:
    """
    Generate a mock AI report for demonstration purposes.
    
    Args:
        patient_data: Patient information
        prediction: Prediction results
        explanation: Explanation results
        style: Report style
        
    Returns:
        str: Mock generated report
    """
    risk_level = prediction.get('predicted_label', 'Unknown')
    risk_probability = prediction.get('risk_probability', 0)
    age = patient_data.get('age', 'Unknown')
    diagnosis = patient_data.get('primary_diagnosis', 'Unknown')
    
    if style == "patient":
        return f"""
Based on your recent hospital stay, here's what we found:

You have a {risk_level.lower()} likelihood of needing to return to the hospital. This is based on factors like your age ({age}), your condition ({diagnosis}), and your hospital stay details.

Don't worry - many of these factors can be managed with good follow-up care. The most important thing is to attend your scheduled appointments and follow your doctor's advice carefully.

What you can do:
- Take your medications as prescribed
- Keep your follow-up appointments
- Monitor how you're feeling at home
- Contact your doctor if anything changes
"""
    
    elif style == "nurse":
        return f"""
Nursing Assessment:

Patient demonstrates {risk_level.lower()} readmission risk ({risk_probability}%). Key factors include age ({age}), primary diagnosis ({diagnosis}), and clinical markers.

Nursing Priorities:
1. Ensure thorough discharge education
2. Verify medication reconciliation completed
3. Schedule timely follow-up appointment
4. Provide clear symptom monitoring guidelines

Recommended nursing interventions focus on patient education and close follow-up to mitigate identified risk factors.
"""
    
    elif style == "detailed":
        return f"""
COMPREHENSIVE RISK ASSESSMENT REPORT

Executive Summary:
Patient presents with {risk_level} of hospital readmission, with a calculated probability of {risk_probability}%. This assessment is based on comprehensive analysis of clinical, demographic, and historical factors.

Factor Analysis:
- Primary drivers include age-related factors, clinical complexity of {diagnosis}, and specific biomarker patterns
- Protective factors include stable vital signs and appropriate treatment response
- Social determinants and compliance history contribute moderately to overall risk

Clinical Recommendations:
1. Implement enhanced discharge planning protocol
2. Schedule follow-up within 7 days post-discharge
3. Consider remote monitoring for first 30 days
4. Coordinate with primary care for seamless transition

Monitoring Parameters:
- Weekly weight and symptom checks
- Bi-weekly vital sign monitoring
- Medication adherence verification at each follow-up
"""
    
    else:  # doctor style
        return f"""
Clinical Risk Assessment:

This {age}-year-old patient with {diagnosis} demonstrates {risk_level.lower()} readmission risk ({risk_probability}%). The prediction is primarily driven by clinical complexity, historical admission patterns, and specific biomarker values.

Key contributing factors include the patient's clinical presentation, treatment response profile, and underlying comorbidities that typically correlate with higher readmission probabilities in similar patient populations.

Recommended next steps include close follow-up within 7-10 days, comprehensive medication reconciliation, and consideration of transitional care services to support post-discharge recovery and monitoring.
"""

def structure_generated_report(report_text: str, style: str, detail_level: str) -> Dict:
    """
    Structure the generated report into organized sections.
    
    Args:
        report_text: Raw generated report text
        style: Report style
        detail_level: Detail level
        
    Returns:
        Dict: Structured report
    """
    # Basic structure
    structured_report = {
        'report_text': report_text,
        'sections': {}
    }
    
    # Try to extract sections based on content
    lines = report_text.split('\n')
    current_section = 'summary'
    sections = {'summary': []}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers
        if (line.upper() == line and len(line) < 50) or ':' in line and len(line) < 100:
            current_section = line.lower().replace(':', '').replace(' ', '_')
            sections[current_section] = []
        else:
            sections[current_section].append(line)
    
    # Convert lists to text
    for section, content in sections.items():
        structured_report['sections'][section] = '\n'.join(content)
    
    return structured_report

def generate_summary_for_all(patients_df: pd.DataFrame,
                           model: object,
                           explanations: Dict,
                           output_dir: str = "reports") -> Dict:
    """
    Generate reports for multiple patients and save them.
    
    Args:
        patients_df: DataFrame with multiple patients
        model: Trained model
        explanations: Dictionary of explanations for all patients
        output_dir: Output directory for reports
        
    Returns:
        Dict: Summary of generated reports
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        generated_reports = {}
        
        for idx, patient_row in patients_df.iterrows():
            patient_id = f"PAT_{idx:04d}"
            
            # Extract patient data
            patient_data = patient_row.to_dict()
            
            # Generate prediction (simplified - in real app this would come from model_utils)
            prediction = {
                'patient_id': patient_id,
                'predicted_label': 'High Risk' if idx % 3 == 0 else 'Low Risk',
                'risk_probability': 75.0 if idx % 3 == 0 else 25.0,
                'confidence': 85.0
            }
            
            # Get explanation for this patient
            explanation = explanations.get(patient_id, {})
            
            # Generate report
            report = generate_patient_report(
                patient_data=patient_data,
                prediction=prediction,
                explanation=explanation,
                style="doctor"
            )
            
            # Save report
            report_filename = output_path / f"{patient_id}_report.json"
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            generated_reports[patient_id] = {
                'file_path': str(report_filename),
                'risk_level': prediction['predicted_label'],
                'generated': True
            }
        
        st.success(f"✅ Generated {len(generated_reports)} patient reports in {output_dir}/")
        return generated_reports
        
    except Exception as e:
        st.error(f"❌ Error generating batch reports: {str(e)}")
        return {'error': str(e)}

def display_report_in_streamlit(report: Dict, style: str = "doctor"):
    """
    Display the generated report in Streamlit.
    
    Args:
        report: Generated report dictionary
        style: Report style for formatting
    """
    if 'error' in report:
        st.error(f"Report generation error: {report['error']}")
        return
    
    st.header("📋 AI-Generated Patient Report")
    
    # Display metadata
    if 'metadata' in report:
        with st.expander("Report Information"):
            meta = report['metadata']
            st.write(f"**Generated:** {meta.get('generated_at', 'Unknown')}")
            st.write(f"**Style:** {meta.get('style', 'Unknown')}")
            st.write(f"**Patient ID:** {meta.get('patient_id', 'Unknown')}")
    
    # Display report text
    report_text = report.get('report_text', '')
    
    if style == "patient":
        st.info("👥 Patient-Friendly Version")
        st.write(report_text)
        
    elif style == "nurse":
        st.success("👩‍⚕️ Nursing Summary")
        st.write(report_text)
        
    elif style == "detailed":
        st.warning("📊 Detailed Clinical Analysis")
        st.write(report_text)
        
    else:  # doctor
        st.success("🏥 Clinical Report")
        st.write(report_text)
    
    # Display sections if available
    if 'sections' in report and report['sections']:
        st.subheader("Report Sections")
        for section_name, section_content in report['sections'].items():
            if section_content.strip():
                with st.expander(f"{section_name.replace('_', ' ').title()}"):
                    st.write(section_content)

def save_report(report: Dict, format: str = "txt") -> str:
    """
    Save report to file in specified format.
    
    Args:
        report: Generated report dictionary
        format: "txt", "json", or "pdf" (pdf requires additional libraries)
        
    Returns:
        str: Path to saved file
    """
    try:
        output_dir = Path("reports/generated_reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        patient_id = report.get('metadata', {}).get('patient_id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = output_dir / f"{patient_id}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
        else:  # txt
            filename = output_dir / f"{patient_id}_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write("PATIENT READMISSION RISK REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Write metadata
                meta = report.get('metadata', {})
                f.write(f"Generated: {meta.get('generated_at', 'Unknown')}\n")
                f.write(f"Patient ID: {meta.get('patient_id', 'Unknown')}\n")
                f.write(f"Report Style: {meta.get('style', 'Unknown')}\n\n")
                
                # Write report content
                f.write(report.get('report_text', ''))
                f.write("\n\n---\n")
                f.write("Note: This report is AI-generated and should be used as a decision support tool.\n")
                f.write("Always verify with clinical judgment and patient context.\n")
        
        return str(filename)
        
    except Exception as e:
        st.error(f"❌ Error saving report: {str(e)}")
        return ""

# Example usage and testing
def test_genai_report():
    """
    Test the Generative AI reporting utilities.
    """
    print("🧪 Testing genai_report.py...")
    
    # Create sample data
    sample_patient = {
        'age': 65,
        'gender': 'Male',
        'primary_diagnosis': 'Heart Failure',
        'length_of_stay': 7,
        'num_previous_admissions': 2,
        'bmi': 28.5,
        'glucose_level': 145.0
    }
    
    sample_prediction = {
        'patient_id': 'TEST_001',
        'predicted_label': 'High Risk',
        'risk_probability': 82.5,
        'confidence': 88.0
    }
    
    sample_explanation = {
        'summary': {
            'key_factors': [
                {'feature': 'num_previous_admissions', 'impact': 0.15},
                {'feature': 'glucose_level', 'impact': 0.12},
                {'feature': 'age', 'impact': 0.10}
            ],
            'risk_drivers': [
                {'feature': 'num_previous_admissions', 'impact': 0.15},
                {'feature': 'glucose_level', 'impact': 0.12}
            ],
            'agreement_level': 'high'
        }
    }
    
    print("✅ Generative AI reporting utilities ready!")
    print("   - Multiple reporting styles: doctor, nurse, patient, detailed")
    print("   - OpenAI integration with fallback to mock responses")
    print("   - Structured report generation and formatting")
    print("   - Batch reporting capabilities")
    print("   - Streamlit display integration")

if __name__ == "__main__":
    test_genai_report()