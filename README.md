# 💸 Engelsiz Cüzdan

Görme engelliler için para tanıma ve sesli geri bildirim sistemi.

## 📋 Proje Hakkında

**Engelsiz Cüzdan**, görme engelli bireylerin elindeki Türk Lirası banknotlarını (5, 10, 20, 50, 100, 200 TL) kameradan okuyup **sesli olarak bildiren** bir derin öğrenme tabanlı sistemdir.

Kullanıcı sadece parayı kameraya gösteriyor, sistem hem görsel olarak parayı tanıyor hem de üzerindeki rakamı okuyup, **"200 Türk Lirası"** gibi net bir sesli geri bildirim veriyor.

## 🎯 Özellikler

- ✅ **Yüksek Doğruluk**: %98 validation accuracy ile güvenilir tahmin
- ✅ **Hibrit Sistem**: Derin öğrenme modeli + OCR kombinasyonu
- ✅ **Gerçek Zamanlı**: Mobil cihazlarda hızlı tahmin
- ✅ **Erişilebilir**: Otomatik Türkçe seslendirme
- ✅ **Dayanıklı**: Her ışık ve koşulda çalışır
- ✅ **Şeffaf**: Detaylı performans metrikleri ve test senaryosu

## 🛠️ Teknolojiler

- **Python 3.11**
- **TensorFlow/Keras**: Derin öğrenme modeli
- **MobileNetV2**: Transfer learning mimarisi
- **OpenCV**: Görüntü işleme
- **EasyOCR**: Optik karakter tanıma
- **Streamlit**: Web arayüzü
- **gTTS**: Text-to-Speech (Türkçe)
- **scikit-learn**: Performans metrikleri

## 📁 Proje Yapısı

```
EngelsizCüzdan/
├── egitim.py                  # Model eğitim script'i
├── app.py                     # Streamlit web arayüzü
├── requirements.txt           # Python bağımlılıkları
├── README.md                  # Bu dosya
├── banknote_efficientnetv2b0.keras  # Eğitilmiş model
├── class_names.npy            # Sınıf isimleri
├── training_metrics.json     # Eğitim metrikleri
└── dataset/                   # Veri seti
    ├── 5/
    ├── 10/
    ├── 20/
    ├── 50/
    ├── 100/
    └── 200/
```

## 🚀 Kurulum

### 1. Gereksinimler

- Python 3.11
- Virtual environment (önerilir)

### 2. Adımlar

```bash
# Proje klasörüne git
cd EngelsizCüzdan

# Virtual environment oluştur
python3.11 -m venv .venv

# Virtual environment'ı aktifleştir
source .venv/bin/activate  # Mac/Linux
# veya
.venv\Scripts\activate  # Windows

# Bağımlılıkları yükle
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Veri Seti

- **Toplam Görüntü**: ~6,000
- **Sınıf Sayısı**: 6 (5, 10, 20, 50, 100, 200 TL)
- **Dağılım**: Her sınıf için ~1,000 görüntü
- **Bölünme**: %80 eğitim, %20 doğrulama

Veri seti yapısı:
```
dataset/
├── 5/     (1001 görüntü)
├── 10/    (1000 görüntü)
├── 20/    (1002 görüntü)
├── 50/    (1000 görüntü)
├── 100/   (1001 görüntü)
└── 200/   (1000 görüntü)
```

## 🧠 Model Mimarisi

### Transfer Learning + Fine-Tuning

Model, **MobileNetV2** tabanlı iki aşamalı bir eğitim stratejisi kullanır:

#### Aşama 1: Transfer Learning
- **Base Model**: MobileNetV2 (ImageNet ağırlıkları, frozen)
- **Öğrenen Katmanlar**: GlobalAveragePooling2D → Dropout(0.2) → Dense(6, softmax)
- **Learning Rate**: 1e-3
- **Epochs**: 30 (veya early stopping)

#### Aşama 2: Fine-Tuning
- **Koşul**: Aşama 1'de validation accuracy > %50
- **Öğrenen Katmanlar**: MobileNetV2'nin son 20 katmanı
- **Learning Rate**: 1e-5 (düşük, ince ayar için)
- **Epochs**: 15 (veya early stopping)

### Model Özellikleri

- **Mimari**: MobileNetV2 (hafif, mobil uyumlu)
- **Input Boyutu**: 224×224×3
- **Output**: 6 sınıf (softmax)
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam

## 🎓 Model Eğitimi

```bash
python3 egitim.py
```

Eğitim süreci:
1. Veri seti yüklenir ve %80/%20 split yapılır
2. Aşama 1: Transfer learning (base frozen)
3. Aşama 2: Fine-tuning (son 20 katman)
4. Validation seti üzerinde detaylı metrikler hesaplanır
5. Model ve metrikler kaydedilir

**Çıktılar**:
- `banknote_efficientnetv2b0.keras`: Eğitilmiş model
- `class_names.npy`: Sınıf isimleri
- `training_metrics.json`: Eğitim metrikleri

## 🌐 Web Arayüzü

```bash
streamlit run app.py
```

Arayüz özellikleri:
- **Sol Sidebar**: Model bilgileri, veri seti istatistikleri, performans metrikleri
- **Ana İçerik**: 
  - Kamera ile para tanıma
  - Test senaryosu (dataset'ten rastgele görsel)

## 🔍 Hibrit Tahmin Sistemi

Sistem, **derin öğrenme modeli** ve **OCR** kombinasyonu kullanır:

1. **Model Tahmini**: MobileNetV2 ile görsel sınıflandırma
2. **OCR Tahmini**: EasyOCR ile banknot üzerindeki rakam okuma
3. **Karar Mekanizması**:
   - Model ve OCR uyumlu → "Model + OCR (Uyumlu)"
   - Çelişki varsa ve OCR güvenilir → "OCR (Öncelikli)"
   - OCR başarısız → "Model"

### OCR Görüntü İyileştirme

OCR performansı için:
- **Gaussian Blur**: Gürültü azaltma
- **CLAHE**: Kontrast iyileştirme (HSV-V kanalı)
- **Unsharp Mask**: Keskinleştirme

## 📈 Performans Metrikleri

Model performansı:
- **Validation Accuracy**: ~%98
- **Macro Average F1-Score**: Yüksek
- **Sınıf Bazlı Metrikler**: Her banknot için precision, recall, F1-score

Metrikler `training_metrics.json` dosyasında saklanır ve Streamlit arayüzünde gösterilir.

## 🎯 Kullanım Senaryoları

1. **Günlük Alışveriş**: Kasiyerden para üstü alırken kontrol
2. **Toplu Taşıma**: Bilet alırken para seçimi
3. **Bağımsızlık**: Görme engelli bireylerin günlük hayatta bağımsız hareket etmesi

## 🔧 Teknik Detaylar

### Preprocessing

**Eğitim**:
- Resize: 224×224
- Normalizasyon: [0,255] → [0,1]
- Augmentation: Yatay flip (%50 olasılık)

**Inference**:
- BGR → RGB dönüşümü
- Resize: 224×224
- Normalizasyon: [0,255] → [0,1]
- Batch dimension ekleme

**Önemli**: Eğitim ve inference preprocessing'i **tamamen aynı** (distribution shift önleme)

### Callbacks

- **ModelCheckpoint**: En iyi validation accuracy'yi kaydet
- **EarlyStopping**: Overfitting önleme (patience: 15/10)
- **ReduceLROnPlateau**: Dinamik learning rate azaltma

### Optimizasyon

- **tf.data API**: Verimli veri yükleme
- **Cache**: Veriyi bellekte tutma
- **Prefetch**: GPU/CPU paralel çalışma
- **Streamlit Cache**: Model ve OCR reader tek sefer yükleme

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 👤 Geliştirici

Makine Öğrenmesi Final Ödevi - Engelsiz Cüzdan

## 🙏 Teşekkürler

- TensorFlow/Keras ekibine
- EasyOCR geliştiricilerine
- Streamlit ekibine

---

**Not**: Bu proje görme engelliler için erişilebilir teknoloji geliştirme amacıyla yapılmıştır.

