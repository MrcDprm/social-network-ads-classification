# Social Network Ads Classification - Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57)
![Build](https://img.shields.io/badge/Status-Success-brightgreen)

Bu proje, **Bilgisayar Mühendisliği 4. Sınıf "Makine Öğrenmesi"** dersi kapsamında geliştirilmiş kapsamlı ve interaktif bir makine öğrenmesi uygulamasıdır. Projenin temel amacı, kullanıcıların sosyal medya reklam verilerini kullanarak farklı sınıflandırma algoritmalarını eğitmesi, performanslarını değerlendirmesi ve interaktif bir web arayüzü üzerinden canlı tahminler yapabilmesini sağlamaktır.

## 🚀 Proje Hakkında

Bu uygulama, makine öğrenmesi süreçlerini son kullanıcı için ulaşılabilir kılar. Kullanılan temel teknolojiler ve sağlanan özellikler sayesinde veri ön işleme, model optimizasyonu (GridSearchCV, K-Fold Cross Validation), performans karşılaştırması ve veritabanı kayıt işlemleri dinamik olarak yapılabilmektedir.

Arayüz tamamen **Streamlit** üzerinden geliştirilmiş olup, profesyonel bir deneyim sunması amacıyla uluslararasılaştırma (i18n), karanlık/aydınlık tema desteği, ve gelişmiş veri görselleştirme tekniklerini barındırmaktadır.

### ✨ Temel Özellikler

- **Çoklu Model Desteği:** K-En Yakın Komşu (KNN), Destek Vektör Makineleri (SVM), Lojistik Regresyon ve Çok Katmanlı Algılayıcı (MLP - Yapay Sinir Ağları) modelleri.
- **Model Optimizasyonu:** KNN modeli için GridSearchCV algoritmaları ve K-Fold Cross Validation ile dinamik hiperparametre optimizasyonu.
- **İnteraktif Veri Görselleştirme:** Çapraz tablo (Confusion Matrix), ısı haritaları (Heatmaps), Dağılım grafikleri ve dinamik çubuk grafikler (Plotly & Seaborn).
- **SQLite Veritabanı Entegrasyonu:** Tüm tahmin sonuçlarının geçmişi ve kullanıcı logları yerel `predictions.db` SQLite veritabanına otomatik olarak kaydedilir.
- **Çift Dil Desteği (i18n):** Uygulama anlık olarak Türkçe ve İngilizce dil seçenekleri arasında pürüzsüz (seamless) bir şekilde geçiş yapabilir.
- **Dinamik UI/UX:** "Eğitim ve Tahmin" (Training & Prediction) sayfası ile "Model Karşılaştırma" (Model Comparison) sayfası arasında geçiş yapabilen "Multi-page" (çoklu sayfa) mimarisi.

## 🛠 Kullanılan Teknolojiler

- **Programlama Dili:** Python 3.8+
- **Veri Bilimi ve ML:** `scikit-learn`, `pandas`, `numpy`
- **Web Arayüzü (Frontend/Backend):** `streamlit`
- **Görselleştirme:** `matplotlib`, `seaborn`, `plotly`
- **Veritabanı:** `sqlite3`

## 📂 Proje Dosya Dizini

```
social-network-ads-classification/
│
├── app.py                   # Ana Streamlit uygulama dosyası, tüm ML ve UI mantığı
├── translations.py          # i18n dil dosyaları (Türkçe / İngilizce Sözlük)
├── theme.py                 # Tema motoru (Açık ve Koyu tema enjeksiyonu)
├── requirements.txt         # Proje bağımlılıkları ve modüller
├── predictions.db           # SQLite veritabanı dosyası (Dinamik oluşturulur)
└── data/
    └── Social_Network_Ads.csv # Kullanılan örnek makine öğrenmesi veri seti
```

## ⚙️ Kurulum ve Çalıştırma

Projenin yerel bilgisayarınızda çalıştırılabilmesi için aşağıdaki adımları sırasıyla uygulayınız:

1. **Repoyu Klonlayın:**
   ```bash
   git clone https://github.com/MrcDprm/social-network-ads-classification.git
   cd social-network-ads-classification
   ```

2. **Gerekli Kütüphaneleri Yükleyin:**
   Sanal ortam (virtual environment) kullanılması önerilir.
   ```bash
   pip install -r requirements.txt
   ```

3. **Uygulamayı Başlatın:**
   ```bash
   streamlit run app.py
   ```
   *Bu komut çalıştırıldıktan sonra uygulama standart tarayıcınızda veya `http://localhost:8501` adresinde açılacaktır.*

## 🧠 Makine Öğrenmesi Süreçleri ve Modeller

Projeye entegre edilen modeller ve işleyişleri:
1. **KNN (K-Nearest Neighbors):** Komşuluk sayısına göre sınıflandırma yapar. Arayüzden Cross-Validation ve ağırlıklandırma (uniform/distance) metrikleri GridSearch ile entegre bir şekilde manipüle edilebilir.
2. **SVM (Support Vector Machines):** Doğrusal olmayan ayrımlar için RBF (Radial Basis Function) çekirdeği ile verileri ayırmaktır.
3. **Logistic Regression:** İhtimal tabanlı temel sınıflandırma algoritmasıdır.
4. **MLP Classifier:** Multi-layer Perceptron, karmaşık desenleri öğrenen yapay sinir ağı sınıflandırıcısıdır.

**Değerlendirme Metrikleri:**
- Accuracy (Doğruluk)
- Precision (Hassasiyet)
- Recall (Duyarlılık)
- F1-Score

## 👨‍💻 Geliştirici

Bu proje, makine öğrenmesi konseptlerinin pekiştirilmesi ve pratiğe dönüştürülmesi amacıyla geliştirilmiştir. Modern yazılım prensiplerini ve pratiklerini benimseyerek akademik bilgiyi harmanlanmıştır.

---
*Bilgisayar Mühendisliği 4. Sınıf Makine Öğrenmesi Dersi Projesi*
