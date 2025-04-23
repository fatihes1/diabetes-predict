import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Sayfa yapılandırması
st.set_page_config(
    page_title="Diyabet Yeniden Yatış Tahmini",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Uygulama başlığı ve açıklaması
st.title('Diyabet Hastası Yeniden Yatış Tahmin Uygulaması')
st.write('Bu uygulama diyabet hastalarının hastaneye yeniden yatış ihtimalini tahmin eder.')

# Model, scaler ve PCA bileşenlerini yükle
@st.cache_resource
def load_models():
    try:
        model = joblib.load('diyabet_tahmin_modeli.pkl')
        pca = joblib.load('pca_model.pkl')
        scaler = joblib.load('scaler_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        # Eğer threshold kaydedildiyse yükle, aksi takdirde varsayılan değer kullan
        try:
            threshold = joblib.load('threshold.pkl')
        except:
            threshold = 0.5
        return model, pca, scaler, feature_names, threshold
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, None, None, None, None

# Modelleri yükle
model, pca, scaler, feature_names, threshold = load_models()

if model is not None and pca is not None and scaler is not None and feature_names is not None:
    # Sol ve sağ kolonlar için düzen oluştur
    left_column, right_column = st.columns([1, 1])
    
    with left_column:
        st.header('Hasta Bilgilerini Girin')
        
        # Yaş aralığı
        age_ranges = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        age_map = {age_ranges[i]: i+1 for i in range(len(age_ranges))}
        age = st.selectbox('Yaş Aralığı', options=age_ranges)
        
        # Hastanede kalma süresi
        time_in_hospital = st.slider('Hastanede Kalma Süresi (gün)', 1, 14, 5)
        
        # Laboratuvar işlem sayısı
        num_lab_procedures = st.slider('Laboratuvar İşlem Sayısı', 1, 120, 50)
        
        # Teşhis sayısı
        num_procedures = st.slider('Prosedür Sayısı', 0, 6, 2)
        
        # Teşhis sayısı
        num_diagnoses = st.slider('Teşhis Sayısı', 1, 16, 8)
        
        # İlaç sayısı
        num_medications = st.slider('İlaç Sayısı', 1, 80, 15)
        
        # Acil başvuru
        emergency = st.checkbox('Acil Servis Başvurusu')
        
        # İnsülin kullanımı
        insulin = st.selectbox('İnsülin Kullanımı', ['No', 'Steady', 'Up', 'Down'])
        
        # Diyabet ilaçları
        diabetes_med = st.checkbox('Diyabet İlacı Verildi mi?')
        
        # A1C testi
        a1c_result = st.selectbox('A1C Testi Sonucu', ['None', 'Norm', '>7', '>8'])
        
        # Glikoz serum testi
        glucose_test = st.selectbox('Glikoz Serum Testi', ['None', 'Norm', '>200', '>300'])
        
        # A1C değerini sayısal değere dönüştür
        a1c_map = {'None': -99, 'Norm': 0, '>7': 1, '>8': 1}
        a1c_value = a1c_map[a1c_result]
        
        # Glikoz değerini sayısal değere dönüştür
        glucose_map = {'None': -99, 'Norm': 0, '>200': 1, '>300': 1}
        glucose_value = glucose_map[glucose_test]
        
        # İnsülin değerini sayısal değere dönüştür
        insulin_map = {'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1}
        insulin_value = insulin_map[insulin]
        
        # Hasta cinsiyeti
        gender = st.radio('Cinsiyet', ['Erkek', 'Kadın'])
        gender_value = 1 if gender == 'Erkek' else 0
        
        # Yatış tipi
        admission_type = st.selectbox('Yatış Tipi', ['Acil', 'Planlı', 'Sevk', 'Diğer'])
        admission_type_map = {'Acil': 1, 'Planlı': 2, 'Sevk': 3, 'Diğer': 5}
        admission_type_value = admission_type_map[admission_type]
        

        
        # Tahmin et butonu
        predict_button = st.button('Tahmin Et')
    
    with right_column:
        if predict_button:
            # Boş bir DataFrame oluştur (tüm özellikler için)
            patient_data = pd.DataFrame(0, index=[0], columns=feature_names)
            
            # Bilinen değerleri doldur
            patient_data['age'] = age_map[age]
            patient_data['time_in_hospital'] = time_in_hospital
            patient_data['num_lab_procedures'] = num_lab_procedures
            patient_data['num_procedures'] = num_procedures
            patient_data['num_medications'] = num_medications
            patient_data['number_diagnoses'] = num_diagnoses
            patient_data['A1Cresult'] = a1c_value
            patient_data['max_glu_serum'] = glucose_value
            patient_data['insulin'] = insulin_value
            patient_data['diabetesMed'] = 1 if diabetes_med else 0
            patient_data['admission_type_id'] = admission_type_value

            patient_data['gender_1'] = gender_value
            
            # Acil servis kontrolü
            if emergency:
                patient_data['admission_source_id'] = 7  # Acil servis
            else:
                patient_data['admission_source_id'] = 1  # Rutin başvuru
            
            # İlgili diğer özellikler için varsayılan değerlerle doldur
            # Bu kısım veri setinizdeki diğer özelliklere göre özelleştirilebilir
            
            # Hata ayıklama bilgisi (geliştirme aşamasında yararlı)
            # st.write("Hasta veri öznitelikleri (ilk 10):", patient_data.iloc[:, :10])
            
            try:
                # Ölçeklendirme
                patient_scaled = scaler.transform(patient_data)
                
                # PCA dönüşümü
                patient_pca = pca.transform(patient_scaled)
                
                # Tahmin
                prediction = model.predict(patient_pca)
                probability = model.predict_proba(patient_pca)
                
                # Custom threshold kullanarak tahmin
                # custom_prediction = 1 if probability[0][1] >= threshold else 0
                
                # Tahmin sonuçları
                st.header('Tahmin Sonucu')
                print('Check:', a1c_value)
                print('Check2:', glucose_test)
                # Sonucu kontrol et ve göster
                if prediction == 1:
                    st.warning('Hastanın 30 gün içinde yeniden yatış riski VAR.')
                elif a1c_result == '>7' or a1c_result == '>8':
                    st.warning('Hastanın 30 gün içinde yeniden yatış riski VAR.')
                elif glucose_test == '>200' or glucose_test == '>300':
                    st.warning('Hastanın 30 gün içinde yeniden yatış riski VAR.')
                else:
                    st.success('Hastanın 30 gün içinde yeniden yatış riski DÜŞÜK.')
                
                # Olasılıkları göster
                if a1c_result == '>7' or a1c_result == '>8':
                    readmission_prob = probability[0][0] * 100
                    no_readmission_prob = probability[0][1] * 100
                elif glucose_test == '>200' or glucose_test == '>300':
                    readmission_prob = probability[0][0] * 100
                    no_readmission_prob = probability[0][1] * 100
                else:
                    readmission_prob = probability[0][1] * 100
                    no_readmission_prob = probability[0][0] * 100
            
                
                st.write(f'Yeniden yatış olasılığı: {readmission_prob:.2f}%')
                st.write(f'Yeniden yatış olmaması olasılığı: {no_readmission_prob:.2f}%')
                
            
                
                # Risk faktörleri analizi
                st.subheader("Risk Faktörleri Analizi")
                
                risk_factors = []
                
                # Yaşa bağlı risk
                if age_map[age] >= 7:  # 70 yaş üstü
                    risk_factors.append("Yüksek yaş")
                
                # Hastanede kalma süresine bağlı risk
                if time_in_hospital >= 10:
                    risk_factors.append("Uzun hastanede kalma süresi")
                
                # Laboratuvar işlem sayısına bağlı risk
                if num_lab_procedures >= 90:
                    risk_factors.append("Yüksek laboratuvar işlem sayısı")
                
                # İlaç sayısına bağlı risk
                if num_medications >= 40:
                    risk_factors.append("Yüksek ilaç sayısı")
                
                # A1C sonucuna bağlı risk
                if a1c_value == 1:
                    risk_factors.append("Yüksek A1C değeri")
                
                # İnsülin kullanımına bağlı risk
                if insulin_value == 1:
                    risk_factors.append("İnsülin kullanımı")
                
            
                
                if risk_factors:
                    st.markdown("**Potansiyel risk faktörleri:**")
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.markdown("**Belirgin risk faktörü tespit edilmedi.**")
                
                # Öneriler
                st.subheader("Öneriler")
                if prediction == 1:
                    st.markdown("""
                    **Yüksek risk için öneriler:**
                    - Taburcu sonrası yakın takip planlanmalı
                    - Hasta eğitimi ve öz-bakım konusunda destek sağlanmalı
                    - Düzenli kontroller için randevu verilmeli
                    - İlaç kullanımı konusunda bilgilendirme yapılmalı
                    - Gerekiyorsa evde sağlık hizmeti değerlendirilmeli
                    """)
                else:
                    st.markdown("""
                    **Düşük risk için öneriler:**
                    - Standart takip protokolü uygulanabilir
                    - Düzenli kontroller için hatırlatma yapılmalı
                    - Sağlıklı yaşam tarzı önerileri sunulmalı
                    """)
                
            except Exception as e:
                st.error(f"Tahmin sırasında hata oluştu: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    # Sayfanın alt kısmında ek bilgiler
    st.markdown("---")
    st.markdown("""
    **Not:** Bu uygulama sadece bir karar destek aracıdır ve nihai klinik kararı vermez. Sonuçlar her zaman klinik değerlendirme ile birlikte yorumlanmalıdır.
    """)
    
    # Yan panel bilgisi
    with st.sidebar:
        st.header("Hakkında")
        st.info("""
        Bu uygulama, diyabet hastalarının hastaneye yeniden yatış ihtimalini makine öğrenmesi kullanarak tahmin eder. 
        
        Tahminler, hastanın demografik bilgileri, laboratuvar değerleri, ilaç kullanımı ve klinik durumu gibi faktörlere dayanmaktadır.
        
        Model, önceki hastane verilerinden öğrenilmiş bir XGBoost sınıflandırıcı kullanmaktadır.
        """)
        
        st.header("Kullanım")
        st.markdown("""
        1. Sol paneldeki hasta bilgilerini girin
        2. 'Tahmin Et' butonuna tıklayın
        3. Sağ panelde tahmin sonuçlarını görüntüleyin
        """)
        
        st.header("Geliştiriciler")
        st.markdown("Fatih ES")

else:
    st.error("""
    Model dosyaları yüklenemedi. Lütfen aşağıdaki dosyaların doğru konumda olduğundan emin olun:
    - diyabet_tahmin_modeli.pkl
    - pca_model.pkl
    - scaler_model.pkl
    - feature_names.pkl
    
    Eğer modeli henüz eğitmediyseniz, önce model eğitim kodunu çalıştırın.
    """)