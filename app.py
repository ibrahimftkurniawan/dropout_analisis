'''
File: app.py
File Created: Thursday, 8th May 2025 10:14:19 am
Author: ibrahimftkurniawan
Copyright @ 2025 Ibrahim FT Kurniawan
'''

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io
import csv

# Set page config
st.set_page_config(
    page_title="Jaya Jaya Institut - Sistem Prediksi Dropout",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
    }
    .risk-high {
        color: white;
        background-color: #EF4444;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .risk-medium {
        color: white;
        background-color: #F59E0B;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .risk-low {
        color: white;
        background-color: #10B981;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6B7280;
        font-size: 0.875rem;
    }
    .batch-results-table {
        margin-top: 1rem;
    }
    .csv-info {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to create sample logo
def get_logo():
    # Create a simple logo with PIL
    img = Image.new('RGB', (200, 200), color=(30, 58, 138))
    return img

# Function to load model and other assets
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        # For demonstration, create mock model objects if files don't exist
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        scaler = StandardScaler()
        feature_names = [
            'Marital_status', 'Application_mode', 'Application_order', 
            'Course', 'Daytime_evening_attendance', 'Previous_qualification', 
            'Previous_qualification_grade', 'Nacionality', 'Mothers_qualification', 
            'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation', 
            'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor', 
            'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'Age_at_enrollment', 
            'International', 'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled', 
            'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved', 
            'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations', 
            'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled', 
            'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved', 
            'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations', 
            'Unemployment_rate', 'Inflation_rate', 'GDP'
        ]
        return model, scaler, feature_names

# Function to map categorical values to numeric
def map_categorical_features(input_data):
    mapped_data = input_data.copy()
    
    # Map gender (1: Male, 0: Female)
    if 'Gender' in mapped_data:
        mapped_data['Gender'] = 1 if mapped_data['Gender'] == 1 else 0
    
    # Map marital status 
    if 'Marital_status' in mapped_data:
        # Keep as is - already numeric in the sample data
        pass
    
    # Map scholarship holder
    if 'Scholarship_holder' in mapped_data:
        # Keep as is - already numeric in the sample data (1: Yes, 0: No)
        pass
        
    return mapped_data

# Function for prediction
def predict_dropout(input_data, model, feature_names):
    # Create DataFrame with the right features
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Keep only features used by model
    input_df = input_df[feature_names]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return prediction, probability

# Function to get risk category
def get_risk_category(probability):
    if probability < 0.3:
        return "RENDAH", "risk-low"
    elif probability < 0.7:
        return "SEDANG", "risk-medium"
    else:
        return "TINGGI", "risk-high"

# Function to display gauge chart
def display_gauge(probability):
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Define colors for different risk levels
    colors = [(0.067, 0.729, 0.506), (0.957, 0.620, 0.043), (0.937, 0.267, 0.267)]
    
    # Create gradient colormap
    cmap = plt.cm.RdYlGn_r
    
    # Draw gauge background
    ax.barh([0], [1], color='#F3F4F6', height=0.5)
    
    # Draw gauge value
    ax.barh([0], [probability], color=cmap(probability), height=0.5)
    
    # Customize gauge appearance
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.3, 0.7, 1.0])
    ax.set_xticklabels(['0%', '30%', '70%', '100%'])
    
    # Add vertical lines for risk thresholds
    ax.axvline(0.3, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.7, color='gray', linestyle='--', alpha=0.5)
    
    # Add risk labels
    ax.text(0.15, -0.4, 'Risiko Rendah', ha='center', color='#10B981', fontweight='bold')
    ax.text(0.5, -0.4, 'Risiko Sedang', ha='center', color='#F59E0B', fontweight='bold')
    ax.text(0.85, -0.4, 'Risiko Tinggi', ha='center', color='#EF4444', fontweight='bold')
    
    # Add the probability text
    ax.text(probability, 0, f"{probability:.1%}", ha='center', va='center', 
            color='white' if 0.3 < probability < 0.95 else 'black',
            fontweight='bold')
    
    plt.title('Probabilitas Dropout', fontsize=12, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return fig

# Function to generate action plan
def generate_action_plan(prediction, input_data):
    risk_areas = []
    
    # Check academic issues
    if input_data.get('Curricular_units_1st_sem_grade', 0) < 12 or input_data.get('Curricular_units_2nd_sem_grade', 0) < 12:
        risk_areas.append("akademik")
    
    # Check attendance issues - using evaluations as proxy for attendance
    if input_data.get('Curricular_units_1st_sem_without_evaluations', 0) > 0 or input_data.get('Curricular_units_2nd_sem_without_evaluations', 0) > 0:
        risk_areas.append("kehadiran")
    
    # Check financial issues
    if input_data.get('Debtor', 0) == 1 or input_data.get('Tuition_fees_up_to_date', 0) == 0:
        risk_areas.append("finansial")
    
    # Check economic context
    if input_data.get('Unemployment_rate', 0) > 10:
        risk_areas.append("ekonomi")
    
    # If no specific risk areas identified but prediction is dropout
    if not risk_areas and prediction == 1:
        risk_areas.append("umum")
    
    return risk_areas

# Function to display recommendations
def display_recommendations(risk_areas):
    recommendations = {
        "akademik": [
            "Ikuti program bimbingan akademik intensif",
            "Dapatkan tutor untuk mata kuliah yang sulit",
            "Diskusikan dengan dosen pengampu untuk memahami materi yang tertinggal",
            "Periksa gaya belajar dan sesuaikan metode belajar"
        ],
        "kehadiran": [
            "Buat jadwal perkuliahan dan patuhi secara ketat",
            "Identifikasi faktor yang menyebabkan ketidakhadiran",
            "Gunakan reminder atau alarm untuk jadwal kuliah",
            "Laporkan masalah transportasi atau logistik kepada bagian kemahasiswaan"
        ],
        "finansial": [
            "Ajukan permohonan beasiswa atau bantuan finansial",
            "Diskusikan opsi pembayaran fleksibel dengan bagian keuangan",
            "Cari informasi tentang program kerja paruh waktu di kampus",
            "Dapatkan konseling keuangan untuk pengelolaan anggaran"
        ],
        "ekonomi": [
            "Cari program magang berbayar untuk menambah pengalaman dan pendapatan",
            "Konsultasikan dengan pusat karir tentang peluang kerja part-time",
            "Pertimbangkan untuk mengikuti program co-op education",
            "Manfaatkan layanan konseling karir untuk perencanaan masa depan"
        ],
        "umum": [
            "Ikuti program mentoring dengan dosen pembimbing",
            "Dapatkan layanan konseling untuk kesejahteraan psikologis",
            "Tetapkan tujuan akademik jangka pendek yang realistis",
            "Evaluasi kemajuan secara berkala dengan penasihat akademik"
        ]
    }
    
    for area in risk_areas:
        st.markdown(f"<div class='metric-card'><h3>Area Risiko: {area.title()}</h3>", unsafe_allow_html=True)
        for rec in recommendations[area]:
            st.markdown(f"- {rec}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

# Function to preprocess CSV data for prediction
def preprocess_csv_data(df, feature_names):
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Ensure all required features exist in the dataframe
    for feature in feature_names:
        if feature not in processed_df.columns:
            processed_df[feature] = 0
    
    # Convert all columns to appropriate numeric types
    for col in processed_df.columns:
        if col in feature_names:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    return processed_df

# Function to batch predict from CSV
def batch_predict(df, model, feature_names):
    # Handle semicolon-separated format
    if len(df.columns) == 1 and ';' in df.iloc[0, 0]:
        # The file is semicolon separated but was read as comma separated
        # Re-read with correct separator
        csv_data = '\n'.join([','.join(row.split(';')) for row in df.iloc[:, 0].str.strip().values])
        buffer = io.StringIO(csv_data)
        df = pd.read_csv(buffer)
    
    # Preprocess the data
    processed_df = preprocess_csv_data(df, feature_names)
    
    # Select only the features used by the model
    input_df = processed_df[feature_names]
    
    # Make predictions
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[:, 1]
    
    # Add results back to original dataframe
    results_df = df.copy()
    results_df['prediction'] = predictions
    results_df['probability'] = probabilities
    results_df['risk_level'] = pd.Series(probabilities).apply(lambda p: "TINGGI" if p >= 0.7 else ("SEDANG" if p >= 0.3 else "RENDAH"))

    return results_df

# Function to generate batch report
def generate_batch_report(results_df):
    # Gunakan mode 'w' untuk string text
    text_buffer = io.StringIO()
    
    # Add headers for CSV file
    headers = list(results_df.columns)
    
    # Create CSV in memory as text first
    writer = csv.writer(text_buffer)
    writer.writerow(headers)
    writer.writerows(results_df.values)
    
    # Dapatkan string dari text_buffer
    text_content = text_buffer.getvalue()
    
    # Konversi ke bytes untuk BytesIO
    bytes_buffer = io.BytesIO()
    bytes_buffer.write(text_content.encode('utf-8'))
    
    # Pastikan buffer dikembalikan ke posisi awal
    bytes_buffer.seek(0)
    return bytes_buffer

    '''
    buffer = io.BytesIO()
    
    # Add headers for CSV file
    headers = list(results_df.columns)
    
    # Create CSV in memory
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(results_df.values)
    
    buffer.seek(0)
    return buffer
    '''

# Function to create a template CSV file based on the new structure
def create_template_csv():
    # Create a template dataframe with example data based on the new structure
    template_data = {
        'Marital_status': [1, 2],
        'Application_mode': [17, 1],
        'Application_order': [5, 1],
        'Course': [171, 172],
        'Daytime_evening_attendance': [1, 0],
        'Previous_qualification': [1, 2],
        'Previous_qualification_grade': [122.0, 130.5],
        'Nacionality': [1, 1],
        'Mothers_qualification': [19, 22],
        'Fathers_qualification': [12, 19],
        'Mothers_occupation': [5, 7],
        'Fathers_occupation': [9, 5],
        'Admission_grade': [127.3, 135.4],
        'Displaced': [1, 0],
        'Educational_special_needs': [0, 0],
        'Debtor': [0, 1],
        'Tuition_fees_up_to_date': [1, 0],
        'Gender': [1, 0],  # 1: Male, 0: Female
        'Scholarship_holder': [0, 1],
        'Age_at_enrollment': [20, 19],
        'International': [0, 0],
        'Curricular_units_1st_sem_credited': [0, 2],
        'Curricular_units_1st_sem_enrolled': [0, 6],
        'Curricular_units_1st_sem_evaluations': [0, 12],
        'Curricular_units_1st_sem_approved': [0, 5],
        'Curricular_units_1st_sem_grade': [0.0, 13.5],
        'Curricular_units_1st_sem_without_evaluations': [0, 0],
        'Curricular_units_2nd_sem_credited': [0, 0],
        'Curricular_units_2nd_sem_enrolled': [0, 6],
        'Curricular_units_2nd_sem_evaluations': [0, 12],
        'Curricular_units_2nd_sem_approved': [0, 6],
        'Curricular_units_2nd_sem_grade': [0.0, 14.2],
        'Curricular_units_2nd_sem_without_evaluations': [0, 0],
        'Unemployment_rate': [10.8, 9.4],
        'Inflation_rate': [1.4, 1.6],
        'GDP': [1.74, 1.92]
    }
    
    df = pd.DataFrame(template_data)
    
    # Create buffer and save to CSV
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)  # Use semicolon separator to match input format
    buffer.seek(0)
    
    return buffer

# Main App
def main():
    # Load model and resources
    model, scaler, feature_names = load_model()
    
    # Initialize session state variables if not exist
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    if 'generate_report' not in st.session_state:
        st.session_state.generate_report = False
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    
    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<h1 class='main-header'>SISTEM PREDIKSI DROPOUT</h1>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Jaya Jaya Institut</h2>", unsafe_allow_html=True)
    
    # Information about the system
    st.info("""
    Sistem ini menggunakan model machine learning untuk mengidentifikasi mahasiswa yang berisiko dropout
    berdasarkan informasi akademik dan non-akademik. Tujuannya adalah untuk memberikan intervensi tepat waktu
    dan dukungan yang sesuai untuk membantu mahasiswa menyelesaikan studi mereka.
    """)
    
    # Navigation
    tab1, tab2, tab3= st.tabs(["Input Data Mahasiswa", "Upload CSV", "Tentang Sistem"])
    
    # Tab 1: Input Data
    with tab1:
        st.markdown("<h3>Masukkan Data Mahasiswa</h3>", unsafe_allow_html=True)
        
        # Create multi-column form for better layout
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Informasi Dasar")
                gender = st.selectbox("Jenis Kelamin (Gender)", ["Laki-laki", "Perempuan"])
                age = st.number_input("Usia (Age_at_enrollment)", min_value=16, max_value=60, value=20)
                marital_status = st.selectbox("Status Perkawinan (Marital_status)", ["Single (1)", "Married (2)", "Other (3)"])
                international = st.selectbox("Mahasiswa Internasional (International)", ["Tidak (0)", "Ya (1)"])
                scholarship = st.selectbox("Penerima Beasiswa (Scholarship_holder)", ["Tidak (0)", "Ya (1)"])
                
                st.markdown("#### Pendidikan & Aplikasi")
                previous_qualification = st.selectbox("Kualifikasi Sebelumnya (Previous_qualification)", 
                                             [1, 2, 3, 4, 5, 6], 
                                             help="Kode kualifikasi pendidikan sebelumnya")
                prev_qual_grade = st.number_input("Nilai Kualifikasi (Previous_qualification_grade)", 
                                              min_value=0.0, max_value=200.0, value=122.0)
                admission_grade = st.number_input("Nilai Masuk (Admission_grade)", 
                                              min_value=0.0, max_value=200.0, value=127.3)
                course = st.number_input("Kode Program Studi (Course)", min_value=1, max_value=999, value=171)
                
            with col2:
                st.markdown("#### Keluarga")
                mothers_qual = st.number_input("Kualifikasi Pendidikan Ibu (Mothers_qualification)", 
                                            min_value=0, max_value=34, value=19, 
                                            help="Kode level pendidikan ibu")
                fathers_qual = st.number_input("Kualifikasi Pendidikan Ayah (Fathers_qualification)", 
                                            min_value=0, max_value=34, value=12, 
                                            help="Kode level pendidikan ayah")
                mothers_occ = st.number_input("Kode Pekerjaan Ibu (Mothers_occupation)", 
                                           min_value=0, max_value=46, value=5)
                fathers_occ = st.number_input("Kode Pekerjaan Ayah (Fathers_occupation)", 
                                           min_value=0, max_value=46, value=9)
                
                st.markdown("#### Kondisi Keuangan")
                debtor = st.selectbox("Status Debitur (Debtor)", ["Tidak (0)", "Ya (1)"], 
                                   help="Apakah mahasiswa memiliki hutang?")
                tuition_up_to_date = st.selectbox("SPP Terbayar (Tuition_fees_up_to_date)", ["Tidak (0)", "Ya (1)"], 
                                               help="Apakah pembayaran SPP tepat waktu?")
                displaced = st.selectbox("Tinggal Jauh dari Kampus (Displaced)", ["Tidak (0)", "Ya (1)"])
                special_needs = st.selectbox("Kebutuhan Khusus (Educational_special_needs)", ["Tidak (0)", "Ya (1)"])
                
            with col3:
                st.markdown("#### Semester 1")
                sem1_credited = st.number_input("Mata Kuliah Diakui Sem 1 (Credited)", min_value=0, max_value=20, value=0)
                sem1_enrolled = st.number_input("Mata Kuliah Terdaftar Sem 1 (Enrolled)", min_value=0, max_value=20, value=0)
                sem1_evaluations = st.number_input("Evaluasi Sem 1 (Evaluations)", min_value=0, max_value=40, value=0)
                sem1_approved = st.number_input("Mata Kuliah Lulus Sem 1 (Approved)", min_value=0, max_value=20, value=0)
                sem1_grade = st.number_input("Nilai Rata-rata Sem 1 (Grade)", min_value=0.0, max_value=20.0, value=0.0)
                sem1_without_eval = st.number_input("MK Tanpa Evaluasi Sem 1", min_value=0, max_value=20, value=0)
                
                st.markdown("#### Semester 2")
                sem2_credited = st.number_input("Mata Kuliah Diakui Sem 2 (Credited)", min_value=0, max_value=20, value=0)
                sem2_enrolled = st.number_input("Mata Kuliah Terdaftar Sem 2 (Enrolled)", min_value=0, max_value=20, value=0)
                sem2_evaluations = st.number_input("Evaluasi Sem 2 (Evaluations)", min_value=0, max_value=40, value=0)
                sem2_approved = st.number_input("Mata Kuliah Lulus Sem 2 (Approved)", min_value=0, max_value=20, value=0)
                sem2_grade = st.number_input("Nilai Rata-rata Sem 2 (Grade)", min_value=0.0, max_value=20.0, value=0.0)
                sem2_without_eval = st.number_input("MK Tanpa Evaluasi Sem 2", min_value=0, max_value=20, value=0)
                
                st.markdown("#### Ekonomi")
                unemployment = st.number_input("Tingkat Pengangguran (Unemployment_rate)", min_value=0.0, max_value=30.0, value=10.8)
                inflation = st.number_input("Tingkat Inflasi (Inflation_rate)", min_value=0.0, max_value=10.0, value=1.4)
                gdp = st.number_input("GDP", min_value=0.0, max_value=5.0, value=1.74)
            
            submit_button = st.form_submit_button("Analisis Risiko Dropout")
        
        if submit_button:
            # Prepare input data
            input_data = {
                'Marital_status': int(marital_status.split("(")[1].split(")")[0]),
                'Application_mode': 17,  # Default value
                'Application_order': 5,  # Default value
                'Course': course,
                'Daytime_evening_attendance': 1,  # Default value
                'Previous_qualification': previous_qualification,
                'Previous_qualification_grade': prev_qual_grade,
                'Nacionality': 1,  # Default value
                'Mothers_qualification': mothers_qual,
                'Fathers_qualification': fathers_qual,
                'Mothers_occupation': mothers_occ,
                'Fathers_occupation': fathers_occ,
                'Admission_grade': admission_grade,
                'Displaced': int(displaced.split("(")[1].split(")")[0]),
                'Educational_special_needs': int(special_needs.split("(")[1].split(")")[0]),
                'Debtor': int(debtor.split("(")[1].split(")")[0]),
                'Tuition_fees_up_to_date': int(tuition_up_to_date.split("(")[1].split(")")[0]),
                'Gender': 1 if gender == "Laki-laki" else 0,
                'Scholarship_holder': int(scholarship.split("(")[1].split(")")[0]),
                'Age_at_enrollment': age,
                'International': int(international.split("(")[1].split(")")[0]),
                'Curricular_units_1st_sem_credited': sem1_credited,
                'Curricular_units_1st_sem_enrolled': sem1_enrolled,
                'Curricular_units_1st_sem_evaluations': sem1_evaluations,
                'Curricular_units_1st_sem_approved': sem1_approved,
                'Curricular_units_1st_sem_grade': sem1_grade,
                'Curricular_units_1st_sem_without_evaluations': sem1_without_eval,
                'Curricular_units_2nd_sem_credited': sem2_credited,
                'Curricular_units_2nd_sem_enrolled': sem2_enrolled,
                'Curricular_units_2nd_sem_evaluations': sem2_evaluations,
                'Curricular_units_2nd_sem_approved': sem2_approved,
                'Curricular_units_2nd_sem_grade': sem2_grade,
                'Curricular_units_2nd_sem_without_evaluations': sem2_without_eval,
                'Unemployment_rate': unemployment,
                'Inflation_rate': inflation,
                'GDP': gdp
            }
            
            # Make prediction
            prediction, probability = predict_dropout(input_data, model, feature_names)
            
            # Store results in session state
            st.session_state.prediction = prediction
            st.session_state.probability = probability
            st.session_state.input_data = input_data
            
            # Display result
            st.markdown("### Hasil Analisis")
            
            risk_level, risk_class = get_risk_category(probability)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"<h2>Risiko Dropout: <span class='{risk_class}'>{risk_level}</span></h2>", 
                           unsafe_allow_html=True)
                st.markdown(f"Probabilitas: **{probability:.1%}**")
                # Display gauge chart
                fig = display_gauge(probability)
                st.pyplot(fig)
                
            with col2:
                # Display student status
                status = "BERISIKO DROPOUT" if prediction == 1 else "TIDAK BERISIKO"
                status_color = "red" if prediction == 1 else "green"
                st.markdown(f"<h3 style='color:{status_color};'>Status: {status}</h3>", unsafe_allow_html=True)
                
                # Display recommendation button
                if prediction == 1:
                    if st.button("Lihat Rekomendasi Tindakan"):
                        st.session_state.show_recommendations = True
            
            # Show recommendations if applicable
            if prediction == 1 and st.session_state.get('show_recommendations', False):
                st.markdown("### Rekomendasi Tindakan")
                risk_areas = generate_action_plan(prediction, input_data)
                display_recommendations(risk_areas)
                
                # Option to download report
                if st.button("Unduh Laporan"):
                    st.session_state.generate_report = True
    
    # Tab 2: Upload CSV
    with tab2:
        st.markdown("<h3>Prediksi Batch dengan File CSV</h3>", unsafe_allow_html=True)
        
        # Add info about CSV format
        st.markdown("""
        <div class='csv-info'>
            <h4>Format CSV</h4>
            <p>Upload file CSV dengan kolom-kolom berikut (urutan tidak harus sama):</p>
            <ul>
                <li>Status Perkawinan (Marital_status)</li>
                <li> (Application_mode)</li>
                <li> (Application_order)</li>
                <li>Kode Program Studi (Course)</li>
                <li> (Daytime_evening_attendance)</li>
                <li>Kualifikasi Sebelumnya (Previous_qualification)</li>
                <li>Nilai Kualifikasi (Previous_qualification_grade)</li>
                <li> (Nacionality)</li>
                <li>Kualifikasi Pendidikan Ibu (Mothers_qualification)</li>
                <li>Kualifikasi Pendidikan Ayah (Fathers_qualification)</li>
                <li>Kode Pekerjaan Ibu (Mothers_occupation)</li>
                <li>Kode Pekerjaan Ayah (Fathers_occupation)</li>
                <li>Nilai Masuk (Admission_grade)</li>
                <li>Tinggal Jauh dari Kampus (Displaced)</li>
                <li>Kebutuhan Khusus (Educational_special_needs)</li>
                <li>Status Debitur (Debtor)</li>
                <li>SPP Terbayar (Tuition_fees_up_to_date)</li>
                <li>Jenis Kelamin (Gender)</li>
                <li>Penerima Beasiswa (Scholarship_holder)</li>
                <li>Usia (Age_at_enrollment)</li>
                <li>Mahasiswa Internasional (International)</li>
                <li> (Curricular_units_1st_sem_credited)</li>
                <li> (Curricular_units_1st_sem_enrolled)</li>
                <li> (Curricular_units_1st_sem_evaluations)</li>
                <li> (Curricular_units_1st_sem_approved)</li>
                <li> (Curricular_units_1st_sem_grade)</li>
                <li> (Curricular_units_1st_sem_without_evaluations)</li>
                <li> (Curricular_units_2nd_sem_credited)</li>
                <li> (Curricular_units_2nd_sem_enrolled)</li>
                <li> (Curricular_units_2nd_sem_evaluations)</li>
                <li> (Curricular_units_2nd_sem_approved)</li>
                <li> (Curricular_units_2nd_sem_grade)</li>
                <li> (Curricular_units_2nd_sem_without_evaluations)</li>
                <li>Tingkat Pengangguran (Unemployment_rate)</li>
                <li>Tingkat Inflasi (Inflation_rate)</li>
                <li>GDP</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Option to download template
        template_col1, template_col2 = st.columns([1, 2])
        with template_col1:
            template_buffer = create_template_csv()
            st.download_button(
                label="Unduh Template CSV",
                data=template_buffer,
                file_name="template_data_mahasiswa.csv",
                mime="text/csv"
            )
        
        # File uploader
        uploaded_file = st.file_uploader("Upload file CSV data mahasiswa", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
                # Display preview of the data
                st.markdown("#### Preview Data")
                st.dataframe(df.head())
                
                # Batch prediction
                if st.button("Proses Prediksi Batch"):
                    with st.spinner("Memproses data..."):
                        # Perform batch prediction
                        results_df = batch_predict(df, model, feature_names)
                        
                        # Store results in session state
                        st.session_state.batch_results = results_df
                        
                        # Display summary
                        st.success(f"Prediksi selesai untuk {len(results_df)} data mahasiswa!")
                        
                        # Summary statistics
                        high_risk_count = (results_df['risk_level'] == 'TINGGI').sum()
                        medium_risk_count = (results_df['risk_level'] == 'SEDANG').sum()
                        low_risk_count = (results_df['risk_level'] == 'RENDAH').sum()
                        
                        # Display summary cards
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.markdown(f"""
                            <div style='background-color:#EF4444; color:white; padding:10px; border-radius:5px; text-align:center;'>
                                <h3>{high_risk_count}</h3>
                                <p>Risiko Tinggi</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with stat_col2:
                            st.markdown(f"""
                            <div style='background-color:#F59E0B; color:white; padding:10px; border-radius:5px; text-align:center;'>
                                <h3>{medium_risk_count}</h3>
                                <p>Risiko Sedang</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with stat_col3:
                            st.markdown(f"""
                            <div style='background-color:#10B981; color:white; padding:10px; border-radius:5px; text-align:center;'>
                                <h3>{low_risk_count}</h3>
                                <p>Risiko Rendah</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Display results if available
                if st.session_state.get('batch_results') is not None:
                    # Display filtered results
                    st.markdown("#### Hasil Prediksi")
                    
                    # Add filter options
                    risk_filter = st.selectbox(
                        "Filter berdasarkan tingkat risiko:",
                        ["Semua", "TINGGI", "SEDANG", "RENDAH"]
                    )
                    
                    filtered_results = st.session_state.batch_results
                    if risk_filter != "Semua":
                        filtered_results = filtered_results[filtered_results['risk_level'] == risk_filter]
                    
                    # Show filtered results
                    st.markdown("<div class='batch-results-table'>", unsafe_allow_html=True)
                    st.dataframe(
                        # filtered_results[['nim', 'nama', 'probability', 'risk_level']].rename(
                        filtered_results[['probability', 'risk_level']].rename(
                            columns={
                                'probability': 'Probabilitas',
                                'risk_level': 'Tingkat Risiko'
                            }
                        ).style.format({'Probabilitas': '{:.1%}'})
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Option to download results
                    csv_buffer = generate_batch_report(filtered_results)
                    st.download_button(
                        label="Unduh Hasil Prediksi",
                        data=csv_buffer,
                        file_name="hasil_prediksi_dropout.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization of results
                    st.markdown("#### Visualisasi Hasil")
                    
                    # Pie chart for risk distribution
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    risk_counts = st.session_state.batch_results['risk_level'].value_counts()
                    colors = ['#10B981', '#F59E0B', '#EF4444'] if 'RENDAH' in risk_counts.index else ['#F59E0B', '#EF4444']
                    
                    # Pie chart
                    ax1.pie(
                        risk_counts, 
                        labels=risk_counts.index, 
                        autopct='%1.1f%%',
                        colors=colors,
                        startangle=90,
                        explode=[0.05] * len(risk_counts)
                    )
                    ax1.set_title('Distribusi Tingkat Risiko')
                    
                    # Histogram of probabilities
                    sns.histplot(
                        st.session_state.batch_results['probability'], 
                        bins=10, 
                        kde=True, 
                        ax=ax2,
                        color='#3B82F6'
                    )
                    ax2.set_title('Distribusi Probabilitas Dropout')
                    ax2.set_xlabel('Probabilitas')
                    ax2.set_ylabel('Jumlah Mahasiswa')
                    
                    # Add vertical lines for thresholds
                    ax2.axvline(0.3, color='gray', linestyle='--')
                    ax2.axvline(0.7, color='gray', linestyle='--')
                    
                    # Add text for thresholds
                    ax2.text(0.15, ax2.get_ylim()[1]*0.9, 'Rendah', ha='center')
                    ax2.text(0.5, ax2.get_ylim()[1]*0.9, 'Sedang', ha='center')
                    ax2.text(0.85, ax2.get_ylim()[1]*0.9, 'Tinggi', ha='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Feature analysis for high-risk students
                    if (st.session_state.batch_results['risk_level'] == 'TINGGI').sum() > 0:
                        st.markdown("#### Analisis untuk Mahasiswa Berisiko Tinggi")
                        
                        high_risk_df = st.session_state.batch_results[st.session_state.batch_results['risk_level'] == 'TINGGI']
                        
                        # Find most common risk factors
                        risk_factors = []
                        
                        # Check academic performance
                        if 'Curricular_units_1st_sem_grade' in high_risk_df.columns and 'Curricular_units_2nd_sem_grade' in high_risk_df.columns:
                            low_gpa_count = ((high_risk_df['Curricular_units_1st_sem_grade'] < 5) | (high_risk_df['Curricular_units_2nd_sem_grade'] < 5)).sum()
                            risk_factors.append(('IPK Rendah', low_gpa_count))
                        
                        # Check financial issues
                        if 'Debtor' in high_risk_df.columns:
                            financial_issues_count = (high_risk_df['Debtor'] == 1).sum()
                            risk_factors.append(('Masalah Keuangan', financial_issues_count))

                        if 'Displaced' in high_risk_df.columns:
                            Displaced_issues_count = (high_risk_df['Displaced'] == 1).sum()
                            risk_factors.append(('Masalah Tempat Tinggal', Displaced_issues_count))
                        
                        other_issues_count = high_risk_count - (risk_factors[0][1]+risk_factors[1][1]+risk_factors[2][1])
                        risk_factors.append(('Masalah Lainnya', other_issues_count))
                        
                        
                        # Create bar chart for risk factors
                        if risk_factors:
                            risk_factors.sort(key=lambda x: x[1], reverse=True)
                            labels = [factor[0] for factor in risk_factors]
                            values = [factor[1] for factor in risk_factors]
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(labels, values, color='#EF4444')
                            ax.set_ylabel('Jumlah Mahasiswa')
                            ax.set_title('Faktor Risiko pada Mahasiswa Berisiko Tinggi')
                            
                            # Add values on top of bars
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                        f'{int(height)}', ha='center', va='bottom')
                            
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")
                st.info("Pastikan format dan struktur file CSV sesuai dengan template.")
    
    
    # Tab 3: About
    with tab3:
        st.markdown("<h3>Tentang Sistem Prediksi Dropout</h3>", unsafe_allow_html=True)
        
        st.write("""
        Sistem prediksi dropout mahasiswa ini dikembangkan untuk membantu institusi pendidikan tinggi
        dalam mengidentifikasi mahasiswa yang berisiko putus kuliah sejak dini, sehingga tindakan
        pencegahan yang tepat dapat diambil untuk meningkatkan retensi mahasiswa.
        """)
        
        st.markdown("#### Model Prediksi")
        st.write("""
        Model ini menggunakan algoritma Random Forest yang dilatih menggunakan data historis mahasiswa.
        Beberapa faktor kunci yang dipertimbangkan dalam model ini meliputi:
        - Performa akademik
        - Latar belakang sosial ekonomi
        - Keterlibatan dalam kegiatan kampus
        - Kebiasaan belajar
        - Faktor demografi
        """)
        
        st.markdown("#### Metrik Performa Model")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "77.17%")
        with col2:
            st.metric("Precision", "72.74%")
        with col3:
            st.metric("Recall", "68.49%")
        with col4:
            st.metric("F1 Score", "69.62%")
        
        st.markdown("#### Fitur Utama")
        st.write("""
        1. **Prediksi Individual** - Analisis risiko untuk satu mahasiswa berdasarkan data yang dimasukkan
        2. **Prediksi Batch** - Analisis risiko untuk banyak mahasiswa melalui file CSV
        3. **Visualisasi** - Grafik dan chart untuk memahami faktor-faktor risiko
        4. **Laporan** - Unduh hasil analisis dalam format yang siap digunakan
        """)
        
        st.markdown("#### Tim Pengembang")
        st.write("""
        Sistem ini dikembangkan oleh Tim Data Science Jaya Jaya Institut, 
        bekerja sama dengan Bagian Kemahasiswaan dan Fakultas Ilmu Komputer.
        """)
        
        st.markdown("#### Kontak")
        st.write("""
        Untuk informasi lebih lanjut atau bantuan teknis, silakan hubungi:
        - Email: ibrahimftkurniawan@gmail.com
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 Tim Data Science Jaya Jaya Institut - Sistem Prediksi Dropout v1.0</p>
    </div>
    """, unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    main()