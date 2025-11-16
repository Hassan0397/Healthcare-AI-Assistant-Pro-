# data_loader.py
"""
Data Loader Module for Patient Readmission Risk Analysis App

This module handles data ingestion, preprocessing, and preparation for the Streamlit dashboard.
It supports both default dataset loading and user-uploaded CSV files.
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
from typing import Tuple, Optional
from pathlib import Path

def get_data_path(filename: str) -> Path:
    """
    Get the correct path to data files, handling different execution contexts.
    
    Args:
        filename: Name of the data file
        
    Returns:
        Path: Full path to the data file
    """
    # Try multiple possible locations
    possible_paths = [
        Path("data") / filename,  # app/data/
        Path("../data") / filename,  # ../data/ (from app directory)
        Path("../../data") / filename,  # ../../data/ (from notebooks)
        Path(filename)  # Direct path
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # If no path found, return the most likely one
    return Path("../data") / filename

@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None) -> pd.DataFrame:
    """
    Load and return the patient dataset.
    
    Args:
        uploaded_file: Streamlit uploaded file object. If None, loads default dataset.
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed patient data
    """
    try:
        if uploaded_file is not None:
            # Load user-uploaded file
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded dataset loaded successfully! Shape: {df.shape}")
        else:
            # Load default dataset - try multiple locations
            default_files = [
                "patient_readmission_risk.csv",
                "patient_readmission_risk_clean.csv"
            ]
            
            df = None
            for data_file in default_files:
                data_path = get_data_path(data_file)
                if data_path.exists():
                    df = pd.read_csv(data_path)
                    st.success(f"✅ Dataset loaded from {data_path}! Shape: {df.shape}")
                    break
            
            if df is None:
                st.error("❌ No dataset found. Please upload a CSV file.")
                return None
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    # Check for missing values
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        st.warning(f"⚠️ Found {missing_count} missing values. Handling them...")
        
        # Handle numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                # Use median for numerical columns (robust to outliers)
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                st.info(f"Filled missing values in {col} with median: {df_clean[col].median():.2f}")
        
        # Handle categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                # Use mode for categorical columns
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                st.info(f"Filled missing values in {col} with mode: {mode_val}")
    
    else:
        st.success("✅ No missing values found!")
        
    return df_clean

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correct data types for all columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with corrected data types
    """
    df_typed = df.copy()
    
    # Expected data types based on EDA
    type_conversions = {
        'numerical': ['age', 'length_of_stay', 'num_previous_admissions', 
                     'num_lab_procedures', 'num_medications', 'num_outpatient_visits',
                     'num_emergency_visits', 'num_inpatient_visits',
                     'blood_pressure_systolic', 'blood_pressure_diastolic',
                     'bmi', 'glucose_level', 'cholesterol_level', 'heart_rate',
                     'oxygen_saturation', 'followup_days'],
        'categorical': ['gender', 'ethnicity', 'admission_type', 'primary_diagnosis',
                       'secondary_diagnosis', 'smoking_status', 'alcohol_intake',
                       'physical_activity_level', 'diet_quality', 'living_alone',
                       'treatment_type', 'discharge_disposition', 
                       'followup_appointment_scheduled']
    }
    
    # Convert numerical columns
    for col in type_conversions['numerical']:
        if col in df_typed.columns:
            df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')
    
    # Convert categorical columns
    for col in type_conversions['categorical']:
        if col in df_typed.columns:
            df_typed[col] = df_typed[col].astype('category')
    
    # Handle binary columns
    binary_cols = ['living_alone', 'followup_appointment_scheduled', 'high_risk_readmission']
    for col in binary_cols:
        if col in df_typed.columns:
            # Convert to string first to handle mixed types, then to int
            df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce').fillna(0).astype(int)
    
    return df_typed

def preprocess_data(df: pd.DataFrame, for_prediction: bool = False) -> pd.DataFrame:
    """
    Main preprocessing function that handles missing values and data types.
    
    Args:
        df: Raw DataFrame
        for_prediction: Whether preprocessing is for model prediction
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame
    """
    if df is None:
        return None
    
    st.info("🔄 Starting data preprocessing...")
    
    # Step 1: Handle missing values
    df_clean = handle_missing_values(df)
    
    # Step 2: Convert data types
    df_clean = convert_data_types(df_clean)
    
    # Step 3: Additional cleaning for prediction
    if for_prediction:
        # Ensure all expected columns are present
        expected_cols = ['age', 'gender', 'ethnicity', 'admission_type', 'length_of_stay',
                        'num_previous_admissions', 'primary_diagnosis', 'bmi', 
                        'glucose_level', 'cholesterol_level', 'heart_rate']
        
        missing_cols = [col for col in expected_cols if col not in df_clean.columns]
        if missing_cols:
            st.warning(f"⚠️ Missing columns for prediction: {missing_cols}")
    
    st.success("✅ Data preprocessing completed!")
    
    return df_clean

def get_data_summary(df: pd.DataFrame) -> None:
    """
    Display data summary in Streamlit.
    
    Args:
        df: DataFrame to summarize
    """
    if df is None:
        return
    
    st.subheader("📊 Dataset Summary")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
    with col2:
        st.metric("Total Features", f"{len(df.columns)}")
    with col3:
        if 'high_risk_readmission' in df.columns:
            high_risk_count = df['high_risk_readmission'].sum()
            high_risk_pct = (high_risk_count / len(df)) * 100
            st.metric("High Risk Patients", f"{high_risk_count} ({high_risk_pct:.1f}%)")
        else:
            st.metric("Target Column", "Not Found")
    
    # Data preview
    with st.expander("View Data Preview"):
        tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Data Types", "Statistical Summary"])
        
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            dtype_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(dtype_info, use_container_width=True)
        
        with tab3:
            if df.select_dtypes(include=[np.number]).shape[1] > 0:
                st.dataframe(df.describe(), use_container_width=True)
            else:
                st.info("No numerical columns to describe")

def load_and_prepare_data(uploaded_file=None, for_prediction: bool = False) -> Optional[pd.DataFrame]:
    """
    Main function to load and prepare data for the app.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        for_prediction: Whether data is for model prediction
        
    Returns:
        Optional[pd.DataFrame]: Prepared DataFrame or None if error
    """
    try:
        # Load data
        df = load_data(uploaded_file)
        
        if df is None:
            return None
        
        # Preprocess data
        df_clean = preprocess_data(df, for_prediction)
        
        # Display summary
        get_data_summary(df_clean)
        
        return df_clean
        
    except Exception as e:
        st.error(f"❌ Error in data preparation: {str(e)}")
        return None

# Test function that doesn't require Streamlit
def test_data_loader():
    """Test the data loader without Streamlit dependencies"""
    print("🧪 Testing data_loader.py...")
    
    # Test path resolution
    test_files = ["patient_readmission_risk.csv", "patient_readmission_risk_clean.csv"]
    
    for test_file in test_files:
        data_path = get_data_path(test_file)
        print(f"Looking for {test_file}: {data_path} -> {'EXISTS' if data_path.exists() else 'NOT FOUND'}")
    
    # Try to load data
    try:
        # Create a simple test without Streamlit
        for data_file in test_files:
            data_path = get_data_path(data_file)
            if data_path.exists():
                df = pd.read_csv(data_path)
                print(f"✅ SUCCESS: Loaded {len(df)} records from {data_path}")
                print(f"   Columns: {len(df.columns)}")
                print(f"   Sample data shape: {df.shape}")
                
                # Test preprocessing
                df_clean = handle_missing_values_silent(df)
                df_typed = convert_data_types_silent(df_clean)
                print(f"✅ Preprocessing successful! Final shape: {df_typed.shape}")
                return df_typed
        
        print("❌ No data files found for testing")
        return None
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return None

def handle_missing_values_silent(df: pd.DataFrame) -> pd.DataFrame:
    """Silent version for testing without Streamlit"""
    df_clean = df.copy()
    
    # Handle numerical columns
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Handle categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
    
    return df_clean

def convert_data_types_silent(df: pd.DataFrame) -> pd.DataFrame:
    """Silent version for testing without Streamlit"""
    df_typed = df.copy()
    
    type_conversions = {
        'numerical': ['age', 'length_of_stay', 'num_previous_admissions', 'bmi', 
                     'glucose_level', 'cholesterol_level', 'heart_rate'],
        'categorical': ['gender', 'ethnicity', 'admission_type', 'primary_diagnosis']
    }
    
    # Convert numerical columns
    for col in type_conversions['numerical']:
        if col in df_typed.columns:
            df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')
    
    # Convert categorical columns
    for col in type_conversions['categorical']:
        if col in df_typed.columns:
            df_typed[col] = df_typed[col].astype('category')
    
    return df_typed

# Example usage (for testing)
if __name__ == "__main__":
    # Test the data loader without Streamlit dependencies
    test_data_loader()