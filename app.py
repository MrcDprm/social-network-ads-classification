import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix
)

# --- Configuration & Constants ---
# Uygulama sayfasının genel ayarlarını yapıyoruz. Tarayıcı sekmesindeki başlık, ikon ve sayfa genişliği (layout) belirleniyor.
st.set_page_config(page_title="Social Network Ads AI", page_icon="🛍️", layout="wide")

# --- i18n & Theme imports ---
# 'translations.py' dosyamızdan İngilizce ve Türkçe metin sözlüklerini (dictionaries) içe aktarıyoruz.
from translations import translations

def t(key: str) -> str:
    """
    Streamlit session state'de (oturum durumunda) seçili olan geçerli dile göre 
    arayüz metinlerini çeviren yardımcı (helper) fonksiyon.
    Eğer anahtar (key) sözlükte bulunamazsa varsayılan olarak anahtarın kendisini döndürür.
    """
    lang = st.session_state.get("lang", "en")
    return translations.get(lang, translations["en"]).get(key, key)

# Veritabanı dosyasının adını sabit olarak tanımlıyoruz. SQLite kullanıyoruz.
DB_NAME = 'predictions.db'

# Veri setini bulmak için işletim sistemi dosya yollarını (paths) sırasıyla kontrol edeceğimiz olası yollar.
# Projenin çalıştırıldığı dizine göre dosya yolunu esnek yönetebilmek için bu yapı tercih edilmiştir.
POTENTIAL_PATHS = [
    'uts/data/Social_Network_Ads.csv',
    'data/Social_Network_Ads.csv',
    'Social_Network_Ads.csv'
]

# --- SQLite Database Helper Functions ---
def init_db():
    """
    Veritabanını başlatır. Eğer 'prediction_history' adında bir tablo henüz yoksa,
    cinsiyet, yaş, maaş, kullanılan model ve tahmin sonucunu saklayacak olan bu tabloyu oluşturur.
    Bu, uygulama ilk açıldığında veya veritabanı yoksa çağrılır.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender TEXT,
            age INTEGER,
            estimated_salary REAL,
            model_used TEXT,
            prediction INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(gender, age, estimated_salary, model_used, prediction):
    """
    Kullanıcı arayüzünden alınan veya tahmini yapılan verileri veritabanına ekleyen fonksiyon.
    Parametrik sorgu (Parameterized query) kullanılarak SQL injection açıklarına karşı önlem alınmıştır.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prediction_history (gender, age, estimated_salary, model_used, prediction)
        VALUES (?, ?, ?, ?, ?)
    ''', (gender, age, estimated_salary, model_used, prediction))
    conn.commit()
    conn.close()

def load_prediction_history():
    """
    Veritabanındaki geçmiş tahminleri Pandas DataFrame olarak çeker (SQL SELECT sorgusu).
    En son yapılan tahminler en üstte görünsün diye 'id DESC' kullanılarak tersine sıralı okuma yapılır.
    """
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query('SELECT * FROM prediction_history ORDER BY id DESC', conn)
    conn.close()
    return df

# --- Data Loading & Preprocessing ---
# st.cache_data kullanarak verilerin bellekte (cache) tutulmasını sağlıyoruz.
# Böylece uygulama her yeniden çalıştığında verileri diskten tekrar tekrar okuyup zaman kaybetmez.
@st.cache_data
def load_data():
    """
    POTENTIAL_PATHS içindeki yolları tarayarak CSV dosyasını arar ve bulduğu an DataFrame
    olarak yükler. Eğer hiçbir yolu bulamazsa arayüze bir hata mesajı basar.
    """
    for path in POTENTIAL_PATHS:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    st.error(t("no_data"))
    return None

def preprocess_data(df):
    """
    Makine öğrenimi algoritmalarının eğitilmesi için ham veriyi işleme stüdyosu.
    'User ID' modeli yanıltabilecek ve anlam taşımayan bir kimlik verisi olduğundan düşürülür (drop).
    Kategorik veri olan 'Gender' sütunu LabelEncoder ile (0 veya 1 gibi) sayısal verilere dönüştürülür.
    X (özellikler/bağımsız değişkenler) ve y (hedef/bağımlı değişken) olarak veri ikiye ayrılır.
    """
    if df is None: return None, None, None, None, None
    
    # Eksik veri (NaN/Null) kontrolü yapılıyor
    miss_vals = df.isnull().sum()
    shape = df.shape
    
    # User ID eğitimde bir anlam ifade etmediği için (gereksiz boyut) veri kümesinden çıkartılır.
    if 'User ID' in df.columns:
        df = df.drop('User ID', axis=1)
        
    encode_mapping = {}
    
    # 'Gender' sütunu sayısal olmadığı için modeller onu anlayamaz.
    # LabelEncoder ile kategorik string veriyi tamsayı temelli (0, 1) kodlarına eşleştiriyoruz.
    if 'Gender' in df.columns:
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        encode_mapping['Gender'] = le

    # 'Purchased' bizim öğrenmeye çalıştığımız hedef sütunumuz. Yani etiket (label).
    X = df.drop('Purchased', axis=1) # Bağımsız Değişkenler
    y = df['Purchased']              # Bağımlı (Hedef) Değişken
    
    return X, y, shape, miss_vals, encode_mapping


# --- Training & Prediction View ---
@st.dialog("📊 Model Performance")
def show_metrics_dialog(acc, prec, rec, f1, ap, model_choice, cm_fig, cv_info=None):
    """
    Model eğitildikten sonra performans sonuçlarını gösteren modern pop-up/modal penceresi.
    Eğitim metrikleri ve Karmaşıklık Matrisi (Confusion Matrix) bu diyalog içinde sunulur.
    """
    if cv_info:
        st.success(f"**{t('best_params')}** {cv_info['best_params']}")
        st.info(f"**{t('mean_cv_acc')}** {cv_info['best_score']:.4f}")
        st.divider()

    st.subheader(t("perf_metrics"))
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric(t("accuracy"), f"{acc:.4f}")
    metric_col2.metric(t("precision"), f"{prec:.4f}")
    metric_col3.metric(t("recall"), f"{rec:.4f}")
    metric_col4.metric(t("f1"), f"{f1:.4f}")
    metric_col5.metric(t("avg_precision"), f"{ap:.4f}")

    st.subheader(t("conf_matrix"))
    st.pyplot(cm_fig)

def show_training_view(df_raw):
    """
    Veri yükleme, modeli eğitme, geçmişi görme gibi eylemleri barındıran Ana Uygulama Safhası.
    Bu kısımda tek bir model yapılandırılıp eğitilebilir.
    """
    # Section 1: Preprocessing & Data Overview
    st.header(t("data_section"))
    
    # Textbook EDA Feature: Sayfa 161 & 162 Entegrasyonu
    st.subheader(t("eda_insights_title"))
    eda_col1, eda_col2, eda_col3 = st.columns(3)
    
    # Her sütundaki benzersiz ögeleri say (unique().size) ve ilk 10'unu göster
    with eda_col1:
        st.markdown(f"**{t('unique_vals')}**")
        for col in df_raw.columns:
            u_size = df_raw[col].unique().size
            with st.expander(f"{col} ({u_size})"):
                st.write(df_raw[col].unique()[:10])

    # Sütunlarda Regex kullanarak sayısal olmayan, tanımsız harfleri vs. ara
    with eda_col2:
        st.markdown(f"**{t('non_numeric_vals')}**")
        non_num_counts = {}
        for col in df_raw.columns:
            # re.search ve re.match kullanarak numerik/harf analizleri gerçekleştirilir (Sayfa 161)
            count = sum(1 for v in df_raw[col].astype(str) if re.search('[a-zA-Z]', v) and not re.match('nan', v.lower()))
            non_num_counts[col] = count
        st.write(non_num_counts)

    # Denetimli Öğrenme Süreçleri - Veri Ön İşleme Kontrol Listesi
    with eda_col3:
        st.markdown(f"**{t('eda_checklist_title')}**")
        st.checkbox(t('eda_outlier_check'), value=True, disabled=True)
        st.checkbox(t('eda_null_check'), value=True, disabled=True)
        st.checkbox(t('eda_fill_delete'), value=True, disabled=True)
        st.checkbox(t('eda_viz_check'), value=True, disabled=True)
        st.checkbox(t('eda_fix_error'), value=True, disabled=True)

    st.divider()

    X, y, df_shape, miss_vals, encode_mappings = preprocess_data(df_raw.copy())

    # Streamlit üzerinde görsel hiyerarşi oluşturmak için alan yaratımı
    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        st.subheader(t("raw_data_sample"))
        st.dataframe(df_raw.head(), use_container_width=True) # İlk 5 satırı göster
    with col2:
        st.subheader(t("dataset_shape"))
        st.info(f"**{t('rows')}:** {df_shape[0]}\n\n**{t('columns')}:** {df_shape[1]}") # Satır/Sütun özeti
    with col3:
        st.subheader(t("missing_values"))
        st.dataframe(miss_vals.rename(t("missing_count"))) # Boş veri sayımı (pd.isnull().sum())

    # Train/Test Split & Scaling
    # Veri setini %80 eğitim, %20 test olarak ayırıyoruz. Test setiyle modelin başarısını ölçeceğiz.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Modelin özellikleri doğru ağırlıklarla öğrenebilmesi için Yaş ve Maaş gibi çok farklı
    # aralıklara sahip değerleri standart bir matrise indirger. (Z-score normalizasyonu)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.subheader(t("scaled_features"))
    st.dataframe(pd.DataFrame(X_train_scaled, columns=X.columns).head())

    # Sidebar: Algorithm & Hyperparameters Selection
    # Kullanıcı yan menü (sidebar) üzerinden makine öğrenimi algoritmalarını ve bunların parametrelerini seçer.
    st.sidebar.header(t("model_config"))
    model_choice = st.sidebar.selectbox(t("select_algo"), (t("knn"), t("svm"), t("lr"), t("mlp")))

    model = None
    
    # Seçilen algoritmaya göre ilgili hiperparametre ayar kontrolleri (slider vb.) gösterilir 
    # ve Scikit-Learn modelleri dinamik olarak (initialize) yapılandırılır.
    if model_choice == t("knn"):
        st.sidebar.subheader(t("knn_params"))
        
        # K-Fold CV Optimizasyonu için Toggle (Checkbox)
        kfold_enabled = st.sidebar.checkbox(t("enable_kfold"), value=False)
        
        if kfold_enabled:
            # Kullanıcı K-Fold'u etkinleştirdiyse, cv (katlama) sayısını seçmesi için slider göster.
            cv_folds = st.sidebar.slider(t("cv_folds"), 2, 10, 5, 1)
            # Parametre ızgarası (Grid Search) kullanılacağı için manuel model oluşturulamıyor, 
            # bunu eğitim aşamasında GridSearchCV kullanarak oluşturacağız.
            model = "GridSearchKNN" 
            st.session_state["knn_cv_folds"] = cv_folds
        else:
            # n_neighbors (K) en yakın komşu sayısını kontrol eder.
            n_neighbors = st.sidebar.slider(t("n_neighbors"), 1, 30, 5, 1)
            weights = st.sidebar.selectbox(t("weights"), ("uniform", "distance"))
            p = st.sidebar.selectbox(t("power_param"), (1, 2))
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)

    elif model_choice == t("svm"):
        st.sidebar.subheader(t("svm_params"))
        # C (Karmaşıklık parametresi): Yanlış sınıflandırmaya ne kadar izin verileceğini seçer.
        C = st.sidebar.slider(t("regularization_c"), 0.01, 10.0, 1.0, 0.01)
        # kernel: Verilerin hangi tür bir düzleme yansıtılarak ayrıştırılacağını seçer (linear, rbf vb).
        kernel = st.sidebar.selectbox(t("kernel"), ("linear", "poly", "rbf", "sigmoid"))
        model = SVC(C=C, kernel=kernel, probability=True, random_state=42)

    elif model_choice == t("lr"):
        st.sidebar.subheader(t("lr_params"))
        C_lr = st.sidebar.slider(t("inverse_reg_c"), 0.01, 10.0, 1.0, 0.01)
        solver = st.sidebar.selectbox(t("solver"), ("lbfgs", "liblinear", "sag", "saga"))
        model = LogisticRegression(C=C_lr, solver=solver, max_iter=2000, random_state=42)

    elif model_choice == t("mlp"):
        st.sidebar.subheader(t("mlp_params"))
        hidden_layer_sizes = st.sidebar.text_input(t("hidden_layers"), "100, 50")
        activation = st.sidebar.selectbox(t("activation"), ("relu", "tanh", "logistic", "identity"))
        solver_mlp = st.sidebar.selectbox(t("solver"), ("adam", "sgd", "lbfgs"))
        learning_rate_init = st.sidebar.slider(t("learning_rate"), 0.0001, 0.1, 0.001, format="%.4f")
        
        try:
            # Virgülle ayrılmış string yapı tabanlı giriş (ör: '100, 50') demetlere/tuple'a (100, 50) çevriliyor.
            layers = tuple(int(x.strip()) for x in hidden_layer_sizes.split(','))
        except:
            st.sidebar.error(t("invalid_hidden_layers"))
            layers = (100,)
            
        model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, solver=solver_mlp, 
                              learning_rate_init=learning_rate_init, max_iter=2000, random_state=42)

    # Section 2: Model Training & Evaluation
    st.header(t("model_section"))
    
    # Kullanıcı "Eğit" tuşuna (button) tıkladığında eğitim döngüsü başlar.
    if st.button(t("train_button"), type="primary"):
        # Eğer K-Fold GridSearch seçiliyse spinner metnini değiştirelim.
        is_grid_search = (model_choice == t("knn") and model == "GridSearchKNN")
        spinner_text = t("optimizing_knn") if is_grid_search else t("training_spinner")
        
        with st.spinner(spinner_text): # Yükleniyor dairesi animasyonu gösterilir.
            try:
                cv_info = None
                if is_grid_search:
                    # GridSearchCV ve K-Katlamalı (K-Fold) Çapraz Doğrulama Mantığı:
                    # K-Fold, eğitim verisini belirlenen cv sayısında katlamalara (folds) böler.
                    # Modeli bu hiperparametre kombinasyonlarıyla olası her katlama ayrımında (k-1 eğitim, 1 test)
                    # defalarca eğitip ortalama geçerlilik (validation) skoru elde eder.
                    # Bu sayede modelin veriyi ezberlemesini (overfitting) önleyip genelleştirme yeteneği yüksek 
                    # en ideal hiperparametre konfigürasyonunu otomatik buluruz.
                    param_grid = {
                        'n_neighbors': list(range(1, 21)),
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']
                    }
                    cv_folds = st.session_state.get("knn_cv_folds", 5)
                    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
                    grid_search.fit(X_train_scaled, y_train)
                    
                    # Optimizasyon sonucunda elde edilen en iyi modeli atıyoruz.
                    model = grid_search.best_estimator_
                    
                    # En iyi parametreleri ve ortalama skoru dialog'da sunmak için sakla
                    cv_info = {
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_
                    }
                else:
                    # Modeli X_train ve y_train üzerinde fit() ile öğretiyoruz (Eğitim kısmı).
                    model.fit(X_train_scaled, y_train)
                
                # Modeli test verisi üzerinde denenmesi için predict() kullanıyoruz.
                y_pred = model.predict(X_test_scaled)
                
                # Çıkan test sonuçlarına göre modelin performans istatistiklerini (Metrikler) hesaplıyoruz.
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Ortalama Hassasiyet (Average Precision) genelde skor/olasılık üzerinden hesaplanır.
                # Bazı modeller class tahmininden (0/1) öte probability sunar, ona göre koşullu okuma yapılır.
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                    ap = average_precision_score(y_test, y_prob)
                else:
                    y_score = model.decision_function(X_test_scaled)
                    ap = average_precision_score(y_test, y_score)
                
                st.success(t("training_complete_msg"))
                
                # Hata Yansıtma/Karmaşıklık Matrisi hesaplama
                # (Gerçekte satınalan ama satın almadı dediğimiz ya da doğru bildiğimiz durumların dökümü)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                # Seaborn ile bu matrisi renkli bir ısı haritasına (heatmap) dökerek görselleştiriyoruz.
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                            xticklabels=[t('cm_not_purchased'), t('cm_purchased')], 
                            yticklabels=[t('cm_not_purchased'), t('cm_purchased')])
                plt.ylabel(t('actual_label'))
                plt.xlabel(t('predicted_label'))
                plt.title(f"{t('cm_title')}{model_choice}")
                
                # Pop-up (dialog) butonu tıklandığında metriklerin kaybolmaması için bunları Session State içine alıyoruz.
                st.session_state['last_metrics'] = (acc, prec, rec, f1, ap, model_choice, fig, cv_info)
                
                # Başarıyla eğitilmiş modeli, scaler'ı ve encoder'ı (iç bellekte) Session State tutuyoruz.
                # Bu şekilde diğer form alanlarında yeni veri gelirken modele hızlıca erişip sınıflandırma yaptırabiliriz.
                st.session_state['model_trained'] = True
                st.session_state['current_model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['encode_mappings'] = encode_mappings
            except Exception as e:
                st.error(f"{t('error_training')}: {e}")

    # Eğitim sonrasında model ve metrikler hazırsa "Metrikleri Göster" dialog butonunu render et.
    # Ana sayfada karmaşıklığı engellemek için metrikler bir pop-up içine gizlendi.
    if st.session_state.get('model_trained') and 'last_metrics' in st.session_state:
        if st.button(t("show_metrics_popup"), type="secondary"):
            # Tuple de-structing (Parçalama)
            last_m = st.session_state['last_metrics']
            if len(last_m) == 8:
                acc, prec, rec, f1, ap, m_choice, fig, cv_info = last_m
            else:
                acc, prec, rec, f1, ap, m_choice, fig = last_m
                cv_info = None
            show_metrics_dialog(acc, prec, rec, f1, ap, m_choice, fig, cv_info)

    st.divider()

    # Section 3: Make New Predictions
    st.header(t("prediction_section"))
    st.markdown(t("pred_desc"))

    # Yeni tahminler yapmak için arayüz formu.
    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        gender_input = st.selectbox(t("gender"), (t("male"), t("female")))
    with col_input2:
        age_input = st.number_input(t("age"), min_value=18, max_value=100, value=30, step=1)
    with col_input3:
        salary_input = st.number_input(t("est_salary"), min_value=10000.0, max_value=250000.0, value=50000.0, step=5000.0)

    # Tahmin Et (Predict) butonuna basılınca çalışacak blok.
    if st.button(t("predict_button")):
        # Eğer henüz üstteki kısımda hiçbir model "Eğit"ilmemişse uyarı verip tahmin yapılmasını engelliyoruz.
        if not st.session_state.get('model_trained'):
            st.warning(t("warning_train"))
        else:
            # Model, scaler ve haritalar Session State üzerinden geri çekiliyor (yükleniyor).
            loaded_model = st.session_state['current_model']
            d_scaler = st.session_state['scaler']
            d_encoder = st.session_state['encode_mappings'].get('Gender')
            
            try:
                # İngilizce (Male) haricinde olan çevrilmiş veriler LabelEncoder hata vereceği için mapping yapıyoruz.
                # Eğitim seti 'Male'/'Female' içerdiği için geri dönüştürmek önemli.
                actual_gender = "Male" if gender_input in ["Male", "Erkek"] else "Female"
                gender_encoded = d_encoder.transform([actual_gender])[0]
            except Exception as e:
                st.error(t("error_encoder"))
                gender_encoded = 0

            # Formdan alınan kullanıcı girişini DataFrame olarak şekillendiriyoruz (modelin anlayacağı boyutta).
            input_df = pd.DataFrame([{
                'Gender': gender_encoded,
                'Age': age_input,
                'EstimatedSalary': salary_input
            }])
            
            try:
                # Kullanıcıdan gelen tahmini değeri, eğitim sırasındaki gibi ölçeklendiriyoruz ki
                # aynı ağırlıklarla hesaplamalara tabi tutulabilsin ve saçma skor üretmesin.
                input_scaled = d_scaler.transform(input_df)
                
                # Artık eğitilmiş modele predict() ile tahmin verdiriyoruz.
                prediction_result = loaded_model.predict(input_scaled)[0]
                pred_label = t("purchased") if prediction_result == 1 else t("not_purchased")
                
                st.success(f"**{t('model_predicts')}:** {pred_label}")
                
                # Tahmin yapıldı. Log tutulabilmesi için SQLite insert fonksiyonunu çağırıp veritabanına ekliyoruz.
                insert_prediction(actual_gender, age_input, salary_input, loaded_model.__class__.__name__, int(prediction_result))
                st.info(t("log_success"))
            except Exception as e:
                st.error(f"{t('error_prediction')}: {e}")

    st.divider()

    # Section 4: History Table
    st.header(t("history_section"))
    st.markdown(t("hist_desc"))
    
    # SQLite veritabanından en güncel tahmin olaylarının loglarını okutuyoruz.
    history_df = load_prediction_history()
    if history_df.empty:
        st.write(t("no_history"))
    else:
        # Sonuçlar ve etiketleri, i18n sistemine göre okunaklı (1 yerine Purchased) yapıya çeviriyoruz.
        history_df['prediction'] = history_df['prediction'].apply(lambda x: t("purchased") if x == 1 else t("not_purchased"))
        
        history_df.rename(columns={
            'id': t("id_col"),
            'gender': t("gender_col"),
            'age': t("age_col"),
            'estimated_salary': t("salary_col"),
            'model_used': t("model_col"),
            'prediction': t("pred_col"),
            'timestamp': t("time_col")
        }, inplace=True)
        
        # 'Male' / 'Female' metinlerini lokal formata ('Erkek' vs.) tablo içerisinde mapliyoruz.
        history_df[t("gender_col")] = history_df[t("gender_col")].apply(lambda g: t("male") if g == "Male" else (t("female") if g == "Female" else g))

        st.dataframe(history_df, use_container_width=True, hide_index=True)


# --- All Models Results View ---
def show_all_models_results_view(df_raw):
    """
    Bu menü, veri setini tüm makine öğrenimi modellerine aynı anda katar (KNN, SVM, LR, MLP)
    ve bu modellerin performans metriklerini Plotly üzerinde şematize ederek karşılaştırmasını sağlar. 
    """
    st.header(t("compare_section"))
    st.markdown(t("compare_desc"))
    
    # Tüm modelleri test etmek için tekrar baştan ön işleme standartları yapıyoruz
    X, y, df_shape, miss_vals, encode_mappings = preprocess_data(df_raw.copy())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dictionary yapısıyla her bir algoritmayı standart parametreleri (default) ile ayağa kaldırıyoruz.
    models = {
        t("knn"): KNeighborsClassifier(),
        t("svm"): SVC(probability=True, random_state=42),
        t("lr"): LogisticRegression(max_iter=2000, random_state=42),
        t("mlp"): MLPClassifier(max_iter=2000, random_state=42)
    }
    
    results = [] # Değerlenmiş metriklerin listesi tutulacak (Tablo grafik için)
    cms = {}     # Her modelin kendi Karmaşıklık (Confusion) matris bilgileri tutulacak 
    
    with st.spinner(t("compare_spinner")):
        # Gelen sözlükteki her bir model sırasıyla öğretilip predict ile sonuç test ediliyor
        for name, clf in models.items():
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            
            # Her bir modelin doğruluğu (Acc), kesinliği (Prec), geri dönüşü (Rec) ve F1 Skoru hesaplanıyor.
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test_scaled)[:, 1]
                ap = average_precision_score(y_test, y_prob)
            else:
                y_score = clf.decision_function(X_test_scaled)
                ap = average_precision_score(y_test, y_score)
                
            results.append({
                "Model": name,
                t("accuracy"): acc,
                t("precision"): prec,
                t("recall"): rec,
                t("f1"): f1,
                t("avg_precision"): ap
            })
            cms[name] = confusion_matrix(y_test, y_pred)
            
    # Döngü sonucunda toplanan tüm farklı metrikleri DataFrame yapısına alıp Pandas'la biçimlendik.
    results_df = pd.DataFrame(results)
    results_df_reset = results_df.copy()
    results_df.set_index("Model", inplace=True)
    
    st.subheader(t("results_table"))
    # "%98.24" gibi yüzdelik forma büründürüp Pandas style property'si ile tabloluyoruz.
    st.dataframe(results_df.style.format("{:.2%}"), use_container_width=True)
    
    # Plotly ile yukarıda hesaplanan bu tablo grafiğe/sütun formata (bar char) dökülerek karşılaştırma sunuluyor.
    # Grafik yığılmış (stacked) olmaması ve net grup (group) olması için pd.melt ile format düzenlemesi yapıyoruz.
    st.subheader(t("results_chart"))
    
    # 'Model' sütunu sabit kalacak şekilde tüm metrik sütunlarını tek bir 'Metric' kolonunda birleştiriyoruz (Unpivot işlemi).
    df_melted = results_df_reset.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig_bar = px.bar(
        df_melted, 
        x="Model", 
        y="Score",
        color="Metric",
        barmode='group',
        text_auto=".2f" # Format skorlarını bar üzerine yerleştir
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.subheader(t("conf_matrix"))
    
    # 4 model de kendi konfüzyon (karmaşıklık) matrisini oluşturdu, ekranı iki kolona ayırıp
    # 1. kolon ve 2. kolon (grid/modüler stil) olacak şekilde döngüde çizdiriyoruz.
    c1, c2 = st.columns(2)
    for i, (name, cm) in enumerate(cms.items()):
        col = c1 if i % 2 == 0 else c2
        with col:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=[t('cm_not_purchased'), t('cm_purchased')], 
                        yticklabels=[t('cm_not_purchased'), t('cm_purchased')])
            plt.ylabel(t('actual_label'))
            plt.xlabel(t('predicted_label'))
            plt.title(f"{t('cm_title')} {name}")
            st.pyplot(fig)


# --- Main Application ---
def main():
    """
    Tüm uygulamanın başlangıç fonksiyonudur.
    Sidebar navigasyon işlemleri, Dil seçimi oturumu (session) ve sayfa yönlendirilmeleri buradan yapılır.
    """
    
    # Uygulama ilk kez yüklendiğinde 'lang' session variable'ı (Oturum Değişkeni) ayarlanmamışsa, en/İngilizce'yle oluşturulur.
    # Bu oturum değişkeni "t" isimli yardımcı fonksiyonda çokça kullanılıyor.
    if "lang" not in st.session_state:
        st.session_state.lang = "en"

    # Yanda (Sidebar) beliren modüller ve navigasyon düğmeleri çizdiriliyor.
    with st.sidebar:
        st.header(t("nav_title"))
        
        # Radyo butonuyla (radio) kullanıcı Sayfa (View) seçimini yapar.
        selected_view = st.radio(
            "Navigation", 
            [t("nav_train"), t("nav_compare")], 
            label_visibility="collapsed"
        )
        
        st.divider()

        st.header(t("language_toggle"))
        # Tüm arayüz dilini en-tr arasında tetikleyen Language box.
        lang_choice = st.selectbox(
            "Language", 
            options=[("en", "English"), ("tr", "Türkçe")], 
            format_func=lambda x: x[1],
            label_visibility="collapsed"
        )
        # Eğer kullanıcı Selectbox içerisindeki dili şimdikinden farklı bir dile çevirirse:
        if lang_choice[0] != st.session_state.lang:
            st.session_state.lang = lang_choice[0] # Oturum state'ini güncelle
            st.rerun() # Tüm uygulamayı baştan (yeni dil referansı ile) yenile

    # Üst Kısım / Başlık ve açıklamalar.
    st.title(t("title"))
    st.markdown(t("app_desc"))
    st.divider()

    # Uygulama başlarken bir kez veritabanının varlığını yordamak amacıyla fonksiyon tetiklenir.
    init_db()

    # Uygulama verisi memory'ye alınıyor
    df_raw = load_data()
    if df_raw is None:
        return

    # Route to the appropriate view
    # Seçili rotaya (sayfa menüsüne) göre hangi form grubunun yükleneceğine dair yönlendirmeler yapılır.
    if selected_view == t("nav_train"):
        show_training_view(df_raw)
    elif selected_view == t("nav_compare"):
        show_all_models_results_view(df_raw)

if __name__ == '__main__':
    main()
