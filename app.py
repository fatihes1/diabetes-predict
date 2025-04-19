import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Başlık ve açıklama
st.title('Diyabet Hastalarının Yeniden Yatış Tahmini')
st.write('Bu uygulama diyabet hastalarının hastaneye yeniden yatış ihtimalini tahmin eder.')

@st.cache_resource
def load_models():
    """Modelleri ve gerekli bileşenleri yükle"""
    model = joblib.load('diyabet_tahmin_modeli.pkl')
    pca = joblib.load('pca_model.pkl')
    scaler = joblib.load('scaler_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, pca, scaler, feature_names

# Modelleri yükle
model, pca, scaler, feature_names = load_models()

# Form oluştur
st.header('Hasta Bilgilerini Girin')

# Bazı önemli özellikleri formda göster
# Bu formda tüm özellikleri göstermek karmaşık olacağından, en önemli birkaç tanesini seçin
col1, col2 = st.columns(2)

with col1:
    age = st.selectbox('Yaş Aralığı', 
                      options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      format_func=lambda x: f"[{(x-1)*10}-{x*10}) yaş")
    
    time_in_hospital = st.slider('Hastanede Kalma Süresi (gün)', 1, 14, 1)
    
    num_lab_procedures = st.slider('Laboratuvar İşlem Sayısı', 1, 120, 45)
    
    num_medications = st.slider('İlaç Sayısı', 1, 80, 15)

with col2:
    num_procedures = st.slider('Prosedür Sayısı', 0, 6, 1)
    
    num_diagnoses = st.slider('Teşhis Sayısı', 1, 16, 8)
    
    diabetesMed = st.selectbox('Diyabet İlacı Verildi mi?', 
                              options=[0, 1], 
                              format_func=lambda x: "Hayır" if x == 0 else "Evet")
    
    A1Cresult = st.selectbox('A1C Testi Sonucu', 
                            options=[-99, 0, 1], 
                            format_func=lambda x: "Test Yok" if x == -99 else ("Normal" if x == 0 else ">7 veya >8"))

# Tahmin butonu
if st.button('Tahmin Et'):
    # Boş bir dataframe oluştur
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Kullanıcının girdiği değerleri ekle
    input_data['age'] = age
    input_data['time_in_hospital'] = time_in_hospital
    input_data['num_lab_procedures'] = num_lab_procedures
    input_data['num_procedures'] = num_procedures
    input_data['num_medications'] = num_medications
    input_data['number_diagnoses'] = num_diagnoses
    input_data['diabetesMed'] = diabetesMed
    input_data['A1Cresult'] = A1Cresult
    
    # Ölçeklendirme ve PCA dönüşümü
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    
    # Tahmin yapma
    prediction = model.predict(input_pca)
    prediction_proba = model.predict_proba(input_pca)
    
    # Sonuçları göster
    st.header('Tahmin Sonucu')
    if prediction[0] == 1:
        st.error('Hastanın 30 gün içinde yeniden yatış riski YÜKSEK!')
        st.write(f'Yeniden yatış olasılığı: {prediction_proba[0][1]:.2%}')
    else:
        st.success('Hastanın 30 gün içinde yeniden yatış riski DÜŞÜK.')
        st.write(f'Yeniden yatış olmaması olasılığı: {prediction_proba[0][0]:.2%}')
    
    # Ek bilgiler
    st.subheader('Model Güven Değerleri')
    st.write(f'Yeniden yatış YOK olasılığı: {prediction_proba[0][0]:.2%}')
    st.write(f'Yeniden yatış VAR olasılığı: {prediction_proba[0][1]:.2%}')