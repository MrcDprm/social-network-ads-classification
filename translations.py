# translations.py
"""
Bu modül, uygulamanın uluslararasılaştırma (i18n) yapısını desteklemek amacıyla İngilizce ('en') ve Türkçe ('tr')
arayüz metinlerini içeren sözlükleri barındırır.
Metinlerin doğrudan kod içine gömülmesi (hardcoding) yerine bu sözlüklerin kullanılması,
kullanıcının Streamlit arayüzünde diller arasında anında ve dinamik olarak geçiş yapabilmesini sağlar.
Anahtarlar (keys), app.py içindeki mantıksal arayüz elemanlarına karşılık gelir.
"""
en = {
    # General & Navigation
    "title": "🛍️ Social Network Ads Purchase Predictor",
    "theme_toggle": "Toggle Theme",
    "language_toggle": "Language",
    "nav_title": "Navigation",
    "nav_train": "Training & Prediction",
    "nav_compare": "Model Comparison & Results",
    "app_desc": "Act as an expert Data Scientist! Tune hyperparameters for various ML models to predict whether a user will buy a product based on their Social Network Ad exposure.",
    
    # Section 1: Data
    "data_section": "1. Data Preprocessing & Overview",
    "no_data": "Dataset not found. Please ensure 'Social_Network_Ads.csv' is present.",
    "raw_data_sample": "Raw Dataset Sample",
    "dataset_shape": "Dataset Shape",
    "rows": "Rows",
    "columns": "Columns",
    "missing_values": "Missing Values",
    "missing_count": "Missing Count",
    "scaled_features": "Scaled Features Sample (Training Data)",

    # Sidebar Options
    "model_config": "🛠️ Model Configuration",
    "select_algo": "Select Classification Algorithm",
    "knn": "K-Nearest Neighbors (KNN)",
    "svm": "Support Vector Machine (SVM)",
    "lr": "Logistic Regression",
    "mlp": "Multilayer Perceptron (MLP)",
    
    # Hyperparameters
    "knn_params": "KNN Hyperparameters",
    "n_neighbors": "Number of Neighbors (k)",
    "weights": "Weights",
    "power_param": "Power parameter (p)",
    "svm_params": "SVM Hyperparameters",
    "regularization_c": "Regularization (C)",
    "kernel": "Kernel",
    "lr_params": "Logistic Regression Hyperparameters",
    "inverse_reg_c": "Inverse of Regularization Strength (C)",
    "solver": "Solver",
    "mlp_params": "MLP Hyperparameters",
    "hidden_layers": "Hidden Layer Sizes (comma-separated)",
    "activation": "Activation Function",
    "learning_rate": "Initial Learning Rate",
    "invalid_hidden_layers": "Invalid hidden layer sizes. Will default to (100,). Example: 100,50",

    # Section 2: Model Training
    "model_section": "2. Model Training & Evaluation",
    "train_button": "🚀 Train Model & View Performance",
    "training_spinner": "Training the model...",
    "perf_metrics": "Performance Metrics (Test Set)",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "avg_precision": "Average Precision (AP)",
    "conf_matrix": "Confusion Matrix",
    "actual_label": "Actual Label",
    "predicted_label": "Predicted Label",
    "cm_not_purchased": "Not Purchased (0)",
    "cm_purchased": "Purchased (1)",
    "cm_title": "Confusion Matrix: ",
    "error_training": "Error occurred during Model Training",
    "training_success_title": "Training Successful",
    "hyperparameters_used": "Hyperparameters",
    "show_metrics_popup": "📊 Show Metrics Pop-up",
    "prediction_result_title": "Prediction Result",
    "training_complete_msg": "Model trained successfully!",

    # Section 3: Prediction
    "prediction_section": "3. Make a New Prediction",
    "pred_desc": "Use the model currently in memory to predict unseen data and log it in the database.",
    "gender": "Gender",
    "male": "Male",
    "female": "Female",
    "age": "Age",
    "est_salary": "Estimated Salary ($)",
    "predict_button": "🔮 Predict & Save to Database",
    "warning_train": "⚠️ Please configure and train a model before making predictions.",
    "error_encoder": "Error internally encoding gender.",
    "model_predicts": "The model predicts",
    "purchased": "Purchased ✅",
    "not_purchased": "Not Purchased ❌",
    "log_success": "Log successfully saved to the integrated SQLite database.",
    "error_prediction": "Error making prediction",

    # Section 4: History
    "history_section": "4. Database Prediction History",
    "hist_desc": "This local SQLite database automatically logs inputs, algorithms utilized, and outputs.",
    "no_history": "No queries cached in database yet. Make a prediction above!",
    "id_col": "ID",
    "gender_col": "Gender",
    "age_col": "Age",
    "salary_col": "Estimated Salary",
    "model_col": "Model Utilized for Output",
    "pred_col": "Final Prediction",
    "time_col": "Timestamp",

    # Model Comparison Page
    "compare_section": "Model Comparison & Results",
    "compare_desc": "Training all classification models on the dataset to compare their performance metrics on the test set.",
    "compare_spinner": "Training all models and computing metrics...",
    "results_table": "Comparison Evaluation Metrics",
    "results_chart": "Metrics Comparison Chart",

    # EDA & Textbook Logic
    "eda_checklist_title": "EDA Checklist",
    "eda_outlier_check": "Outlier Check",
    "eda_null_check": "Null Data Check",
    "eda_fill_delete": "Fill/Delete Data",
    "eda_viz_check": "Visualization Check",
    "eda_fix_error": "Fixing Error Check",
    "unique_vals": "Unique Values",
    "non_numeric_vals": "Non-Numeric Values",
    "eda_insights_title": "Dataset Analysis (Pages 161-162)",
    
    # KNN K-Fold Optimization
    "enable_kfold": "Enable K-Fold CV Optimization",
    "cv_folds": "Number of Folds (cv)",
    "optimizing_knn": "Optimizing KNN via K-Fold CV...",
    "best_params": "Best Parameters:",
    "mean_cv_acc": "Mean CV Accuracy:",
}

tr = {
    # General & Navigation
    "title": "🛍️ Sosyal Ağ Reklamları Satın Alma Tahmincisi",
    "theme_toggle": "Tema Değiştir",
    "language_toggle": "Dil",
    "nav_title": "Gezinme",
    "nav_train": "Eğitim ve Tahmin",
    "nav_compare": "Model Karşılaştırması ve Sonuçlar",
    "app_desc": "Uzman bir Veri Bilimcisi gibi davranın! Bir kullanıcının Sosyal Ağ Reklamı etkileşimine göre bir ürün satın alıp almayacağını tahmin etmek için çeşitli ML modellerinin hiperparametrelerini ayarlayın.",
    
    # Section 1: Data
    "data_section": "1. Veri Ön İşleme & Genel Bakış",
    "no_data": "Veri seti bulunamadı. Lütfen 'Social_Network_Ads.csv' dosyasının mevcut olduğundan emin olun.",
    "raw_data_sample": "Ham Veri Seti Örneği",
    "dataset_shape": "Veri Seti Boyutu",
    "rows": "Satır",
    "columns": "Sütun",
    "missing_values": "Eksik Değerler",
    "missing_count": "Eksik Sayısı",
    "scaled_features": "Ölçeklendirilmiş Özellik Örneği (Eğitim Verisi)",

    # Sidebar Options
    "model_config": "🛠️ Model Yapılandırması",
    "select_algo": "Sınıflandırma Algoritmasını Seçin",
    "knn": "K-En Yakın Komşu (KNN)",
    "svm": "Destek Vektör Makinesi (SVM)",
    "lr": "Lojistik Regresyon",
    "mlp": "Çok Katmanlı Algılayıcı (MLP)",
    
    # Hyperparameters
    "knn_params": "KNN Hiperparametreleri",
    "n_neighbors": "Komşu Sayısı (k)",
    "weights": "Ağırlıklar (Weights)",
    "power_param": "Güç parametresi (p)",
    "svm_params": "SVM Hiperparametreleri",
    "regularization_c": "Düzenlileştirme (C)",
    "kernel": "Çekirdek (Kernel)",
    "lr_params": "Lojistik Regresyon Hiperparametreleri",
    "inverse_reg_c": "Ters Düzenlileştirme Gücü (C)",
    "solver": "Çözücü (Solver)",
    "mlp_params": "MLP Hiperparametreleri",
    "hidden_layers": "Gizli Katman Boyutları (virgülle ayırın)",
    "activation": "Aktivasyon Fonksiyonu",
    "learning_rate": "Başlangıç Öğrenme Oranı",
    "invalid_hidden_layers": "Geçersiz gizli katman boyutları. (100,) kullanılacak. Örnek: 100,50",

    # Section 2: Model Training
    "model_section": "2. Model Eğitimi & Değerlendirme",
    "train_button": "🚀 Modeli Eğit & Performansı Gör",
    "training_spinner": "Model eğitiliyor...",
    "perf_metrics": "Performans Metrikleri (Test Seti)",
    "accuracy": "Doğruluk (Accuracy)",
    "precision": "Kesinlik (Precision)",
    "recall": "Duyarlılık (Recall)",
    "f1": "F1-Skoru",
    "avg_precision": "Ortalama Kesinlik (AP)",
    "conf_matrix": "Karmaşıklık Matrisi",
    "actual_label": "Gerçek Etiket",
    "predicted_label": "Tahmin Edilen Etiket",
    "cm_not_purchased": "Satın Alınmadı (0)",
    "cm_purchased": "Satın Alındı (1)",
    "cm_title": "Karmaşıklık Matrisi: ",
    "error_training": "Model eğitimi sırasında hata oluştu",
    "training_success_title": "Eğitim Başarılı",
    "hyperparameters_used": "Hiperparametreler",
    "show_metrics_popup": "📊 Metrikler Açılır Seçeneğini Göster",
    "prediction_result_title": "Tahmin Sonucu",
    "training_complete_msg": "Model başarıyla eğitildi!",

    # Section 3: Prediction
    "prediction_section": "3. Yeni Bir Tahmin Yap",
    "pred_desc": "Yeni verileri tahmin etmek ve veritabanına kaydetmek için şu anda hafızada olan modeli kullanın.",
    "gender": "Cinsiyet",
    "male": "Erkek",
    "female": "Kadın",
    "age": "Yaş",
    "est_salary": "Tahmini Maaş ($)",
    "predict_button": "🔮 Tahmin Et & Veritabanına Kaydet",
    "warning_train": "⚠️ Lütfen tahmin yapmadan önce bir model yapılandırıp eğitin.",
    "error_encoder": "Cinsiyet kodlanırken hata oluştu.",
    "model_predicts": "Model tahmini",
    "purchased": "Satın Alındı ✅",
    "not_purchased": "Satın Alınmadı ❌",
    "log_success": "Kayıt, entegre SQLite veritabanına başarıyla kaydedildi.",
    "error_prediction": "Tahmin yapılırken hata oluştu",

    # Section 4: History
    "history_section": "4. Veritabanı Tahmin Geçmişi",
    "hist_desc": "Bu yerel SQLite veritabanı girdileri, algoritmayı ve çıktıları otomatik olarak kaydeder.",
    "no_history": "Veritabanında henüz kayıt yok. Yukarıdan bir tahmin yapın!",
    "id_col": "ID",
    "gender_col": "Cinsiyet",
    "age_col": "Yaş",
    "salary_col": "Tahmini Maaş",
    "model_col": "Kullanılan Model",
    "pred_col": "Son Tahmin",
    "time_col": "Zaman Damgası",

    # Model Comparison Page
    "compare_section": "Model Karşılaştırması ve Sonuçlar",
    "compare_desc": "Tüm sınıflandırma modellerini veri seti üzerinde eğiterek test seti üzerindeki performans metriklerini karşılaştırma.",
    "compare_spinner": "Tüm modeller eğitiliyor ve metrikler hesaplanıyor...",
    "results_table": "Karşılaştırmalı Değerlendirme Metrikleri",
    "results_chart": "Metrik Karşılaştırma Grafiği",

    # EDA & Textbook Logic
    "eda_checklist_title": "KKA Kontrol Listesi",
    "eda_outlier_check": "Aykırı Değer Kontrolü",
    "eda_null_check": "Boş Veri Kontrolü",
    "eda_fill_delete": "Eksik Veri Doldurma/Silme",
    "eda_viz_check": "Görselleştirme Kontrolü",
    "eda_fix_error": "Hata Düzeltme Kontrolü",
    "unique_vals": "Benzersiz (Unique) Değerler",
    "non_numeric_vals": "Sayısal Olmayan Değerler",
    "eda_insights_title": "Veriseti Analizi (Sayfa 161-162)",

    # KNN K-Fold Optimization
    "enable_kfold": "K-Katlamalı CV Optimizasyonunu Etkinleştir",
    "cv_folds": "Katlama Sayısı (cv)",
    "optimizing_knn": "K-Katlamalı CV ile KNN Optimize Ediliyor...",
    "best_params": "En İyi Parametreler:",
    "mean_cv_acc": "Ortalama CV Doğruluğu:",
}

# Export a mapping for easy lookup
translations = {"en": en, "tr": tr}
