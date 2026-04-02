import os
import json
import random
import re

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from gtts import gTTS
import easyocr

# --- AYARLAR ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
MODEL_PATH = "banknote_efficientnetv2b0.keras"
CLASS_NAMES_PATH = "class_names.npy"
METRICS_PATH = "training_metrics.json"
DATASET_PATH = "dataset"

# EasyOCR reader'ı cache'le (ilk yüklemede biraz zaman alabilir)
@st.cache_resource
def load_ocr_reader():
    """EasyOCR reader'ı yükler (Türkçe ve İngilizce desteği)"""
    try:
        reader = easyocr.Reader(['tr', 'en'], gpu=False)
        return reader
    except Exception as e:
        st.warning(f"EasyOCR yüklenemedi: {e}. OCR özelliği devre dışı.")
        return None

st.set_page_config(page_title="Engelsiz Cüzdan", page_icon="💸")

st.markdown(
    """
    <style>
    .sonuc { font-size: 50px; font-weight: bold; color: #4CAF50; text-align: center;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💸 Engelsiz Cüzdan")
st.write("Parayı kameraya gösterin, model tahmin etsin ve size sesli söylesin.")


@st.cache_resource
def load_model_and_classes():
    """
    Eğitim script'inde kaydedilen Keras modelini ve sınıf isimlerini yükler.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        return None, None

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()
    return model, class_names


@st.cache_resource
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


model, class_names = load_model_and_classes()
metrics = load_metrics()
ocr_reader = load_ocr_reader()

if model is None or class_names is None:
    st.error(
        f"⚠️ Model veya sınıf isimleri bulunamadı.\n\n"
        f"Lütfen önce `egitim.py` dosyasını çalıştırarak '{MODEL_PATH}' ve '{CLASS_NAMES_PATH}' dosyalarını oluşturun."
    )
    st.stop()


def resize_with_padding(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """
    Görüntünün en-boy oranını koruyarak resize yapar, kenarlara siyah padding ekler.
    
    Args:
        image: Görüntü (numpy array, BGR veya RGB)
        target_height: Hedef yükseklik
        target_width: Hedef genişlik
    
    Returns:
        Resize edilmiş ve padding eklenmiş görüntü
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h
    
    # Hedef en-boy oranına göre resize boyutlarını hesapla
    if aspect_ratio > target_width / target_height:
        # Görüntü daha geniş, genişliği hedefe göre ayarla
        new_w = target_width
        new_h = int(target_width / aspect_ratio)
    else:
        # Görüntü daha yüksek, yüksekliği hedefe göre ayarla
        new_h = target_height
        new_w = int(target_height * aspect_ratio)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Siyah padding ekle
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left
    
    # BGR veya RGB kontrolü
    if len(resized.shape) == 3:
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
    
    return padded


def apply_clahe_and_sharpen(image_bgr: np.ndarray) -> np.ndarray:
    """
    CLAHE (Kontrast iyileştirme) ve Unsharp Mask (netleştirme) uygular.
    
    Args:
        image_bgr: BGR formatında görüntü
    
    Returns:
        İyileştirilmiş görüntü
    """
    # 1) Gürültüyü biraz azaltmak için hafif blur
    image_bgr = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    
    # 2) HSV uzayında V kanalına CLAHE (aydınlık/kontrast iyileştirme)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    hsv_clahe = cv2.merge([h, s, v_clahe])
    image_bgr = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    
    # 3) Unsharp mask ile netleştirme
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), 1.0)
    sharpened = cv2.addWeighted(image_bgr, 1.5, blurred, -0.5, 0)
    
    return sharpened


def preprocess_image_opencv(image_bgr: np.ndarray) -> np.ndarray:
    """
    OpenCV ile okunan BGR görüntüyü,
    Model'e uygun (224x224, RGB, [0,1]) formata çevirir.
    
    ÖNEMLİ: Eğitim sırasında kullanılan preprocessing ile AYNISI olmalı!
    Eğitimde sadece normalizasyon kullanıldı, burada da aynısını yapıyoruz.
    """
    # 1) BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 2) Resize (eğitimde tf.image.resize kullanılıyor, burada cv2.resize)
    # image_dataset_from_directory otomatik olarak resize yapıyor, burada manuel yapıyoruz
    image_resized = cv2.resize(image_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    
    # 3) [0, 255] -> [0, 1] (eğitimdeki normalizasyon ile aynı)
    image_norm = image_resized.astype("float32") / 255.0
    
    # 4) Batch dimension ekle
    return np.expand_dims(image_norm, axis=0)


def extract_number_from_text(text: str) -> int:
    """
    Metinden rakam çıkarır (örn: "200", "İKİYÜZ", "200 TL" -> 200)
    """
    # Önce direkt rakam ara
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    
    # Türkçe sayı kelimelerini kontrol et
    turkish_numbers = {
        'beş': 5, 'on': 10, 'yirmi': 20, 'elli': 50,
        'yüz': 100, 'ikiyüz': 200, 'iki yüz': 200
    }
    text_lower = text.lower().replace(' ', '')
    for key, value in turkish_numbers.items():
        if key in text_lower:
            return value
    
    return None


def ocr_predict(image_bgr: np.ndarray) -> tuple:
    """
    EasyOCR ile görüntüden rakam okur.
    
    Returns:
        (tahmin_edilen_değer, güven_skoru) veya (None, 0.0)
    """
    if ocr_reader is None:
        return None, 0.0
    
    try:
        # OCR için görüntüyü iyileştir
        image_enhanced = apply_clahe_and_sharpen(image_bgr)
        
        # OCR uygula
        results = ocr_reader.readtext(image_enhanced)
        
        # Tüm tespit edilen metinleri birleştir
        all_text = ' '.join([result[1] for result in results])
        
        # Rakam çıkar
        detected_value = extract_number_from_text(all_text)
        
        if detected_value is not None:
            # Geçerli banknot değerleri kontrolü
            valid_values = [5, 10, 20, 50, 100, 200]
            if detected_value in valid_values:
                # En yüksek güven skorunu al
                max_confidence = max([result[2] for result in results], default=0.0)
                return detected_value, max_confidence
        
        return None, 0.0
    except Exception as e:
        st.warning(f"OCR hatası: {e}")
        return None, 0.0


def predict_tensor(input_tensor: np.ndarray):
    """Model tahmini + skor döndürür."""
    preds = model.predict(input_tensor, verbose=0)
    pred_index = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]) * 100.0)
    predicted_label = str(class_names[pred_index])
    return predicted_label, confidence


def hybrid_predict(image_bgr: np.ndarray) -> tuple:
    """
    Hybrid tahmin: Model + OCR kombinasyonu.
    Çelişki durumunda OCR'a öncelik verir.
    
    Returns:
        (final_tahmin, model_confidence, ocr_value, ocr_confidence, kullanılan_yöntem)
    """
    # Model tahmini
    input_tensor = preprocess_image_opencv(image_bgr)
    model_label, model_conf = predict_tensor(input_tensor)
    model_value = int(model_label)
    
    # OCR tahmini
    ocr_value, ocr_conf = ocr_predict(image_bgr)
    
    # Karar mekanizması
    if ocr_value is not None and ocr_conf > 0.5:
        # OCR güvenilir bir sonuç verdi
        if model_value == ocr_value:
            # Her ikisi de aynı sonucu verdi, güvenilir
            return ocr_value, model_conf, ocr_value, ocr_conf, "Model + OCR (Uyumlu)"
        else:
            # Çelişki var, OCR'a öncelik ver
            return ocr_value, model_conf, ocr_value, ocr_conf, "OCR (Öncelikli)"
    else:
        # OCR sonuç vermedi veya güvenilir değil, sadece model kullan
        return model_value, model_conf, None, 0.0, "Model"


def pick_random_dataset_image():
    """Dataset klasöründen rastgele bir görsel ve gerçek etiketini döndürür."""
    if not os.path.isdir(DATASET_PATH):
        return None, None

    # Önce sınıf klasörlerini bul
    class_folders = [
        d
        for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ]
    if not class_folders:
        return None, None

    cls = random.choice(class_folders)
    cls_path = os.path.join(DATASET_PATH, cls)
    images = [
        f
        for f in os.listdir(cls_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not images:
        return None, None

    img_name = random.choice(images)
    img_path = os.path.join(cls_path, img_name)
    return img_path, cls


# --- Sol Sidebar: Metrikler ve Bilgiler ---
with st.sidebar:
    st.header("📊 Model Bilgileri")
    
    if metrics:
        # Veri Seti Bilgileri
        st.markdown("### 📁 Veri Seti")
        st.write(f"**Toplam:** {metrics.get('total_images', 0):,}")
        st.write(f"**Eğitim:** {metrics.get('train_images', 0):,} ({metrics.get('train_images', 0)/metrics.get('total_images', 1)*100:.0f}%)")
        st.write(f"**Doğrulama:** {metrics.get('val_images', 0):,} ({metrics.get('val_images', 0)/metrics.get('total_images', 1)*100:.0f}%)")
        st.write(f"**Sınıf Sayısı:** {metrics.get('num_classes', 6)}")
        
        st.markdown("---")
        
        # Eğitim Yöntemi (Akış Şeması Tarzı)
        st.markdown("### 🔄 Eğitim Yöntemi")
        st.write("**1. Transfer Learning**")
        st.write("  → MobileNetV2 (ImageNet)")
        st.write("  → Frozen base model")
        
        if 'best_val_accuracy_stage1' in metrics:
            stage1_val_acc = metrics.get('best_val_accuracy_stage1', 0.0) * 100
            st.write(f"  → Val Acc: {stage1_val_acc:.1f}%")
        
        if metrics.get('epochs_run_stage2', 0) > 0:
            st.write("**2. Fine-tuning**")
            st.write("  → Son 20 katman")
            st.write("  → Lower learning rate")
            stage2_val_acc = metrics.get('best_val_accuracy_stage2', 0.0) * 100
            st.write(f"  → Val Acc: {stage2_val_acc:.1f}%")
        
        st.markdown("---")
        
        # Performans Metrikleri
        st.markdown("### 📈 Performans")
        final_val_acc = metrics.get('best_val_accuracy', 0.0) * 100
        st.metric("**Validation Accuracy**", f"{final_val_acc:.2f}%")
        
        # Detaylı metrikler varsa göster
        if 'detailed_metrics' in metrics:
            detailed = metrics['detailed_metrics']
            st.markdown("#### Genel Metrikler")
            
            macro_precision = detailed.get('macro_avg', {}).get('precision', 0.0) * 100
            macro_recall = detailed.get('macro_avg', {}).get('recall', 0.0) * 100
            macro_f1 = detailed.get('macro_avg', {}).get('f1_score', 0.0) * 100
            
            st.write(f"**Precision:** {macro_precision:.1f}%")
            st.write(f"**Recall:** {macro_recall:.1f}%")
            st.write(f"**F1-Score:** {macro_f1:.1f}%")
            
            # Sınıf bazında özet (kompakt)
            st.markdown("#### Sınıf Bazında F1-Score")
            if 'class_metrics' in detailed:
                class_metrics_data = detailed['class_metrics']
                for class_name in sorted(class_metrics_data.keys(), key=lambda x: int(x) if x.isdigit() else 999):
                    f1_val = class_metrics_data[class_name].get('f1_score', 0.0) * 100
                    st.write(f"{class_name} TL: {f1_val:.1f}%")
        
        st.markdown("---")
        
        # Model Detayları (Minimal)
        st.markdown("### ⚙️ Model Detayları")
        st.write("**Mimari:** MobileNetV2")
        st.write("**Preprocessing:** Normalizasyon")
        st.write("**Optimizer:** Adam")
        st.write("**Loss:** Sparse Categorical Crossentropy")
        
    else:
        st.warning("⚠️ Metrikler bulunamadı")
        st.info("`egitim.py` çalıştırın")

# --- Ana İçerik: Kamera ve Test Senaryosu ---
st.subheader("📷 Kamera ile Para Tanıma")
img_file = st.camera_input("Para görselini çekin")

if img_file is not None:
        # Resmi oku
        bytes_data = img_file.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Görüntü okunamadı. Lütfen tekrar deneyin.")
        else:
            # Hybrid tahmin (Model + OCR)
            final_value, model_conf, ocr_value, ocr_conf, method = hybrid_predict(img_bgr)

            # Ekranda göster
            st.markdown(
                f'<p class="sonuc">{final_value} TL</p>',
                unsafe_allow_html=True,
            )
            
            # Detaylı bilgi (kompakt)
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Model Güven", f"{model_conf:.1f}%")
            if ocr_value is not None:
                with info_col2:
                    st.metric("OCR Güven", f"{ocr_conf*100:.1f}%")
                st.caption(f"Yöntem: {method}")
            
            # Hata analizi ve uyarı
            if model_conf < 50:
                st.warning(
                    "⚠️ **Lütfen parayı daha net gösterin.** "
                    "Güven skoru düşük. Parayı daha iyi ışıkta, tam kadraja alarak tekrar deneyin."
                )
            elif model_conf < 70:
                st.warning(
                    "⚠️ Tahmin güvenilirliği orta seviyede. "
                    "Daha iyi sonuç için parayı daha net çekin."
                )
            else:
                # Sesli oku
                text = f"{final_value} Türk Lirası"
                try:
                    tts = gTTS(text=text, lang="tr")
                    tts.save("ses.mp3")
                    st.audio("ses.mp3", format="audio/mp3", autoplay=True)
                except Exception:
                    st.error("Ses dosyası oluşturulamadı.")

st.markdown("---")
st.subheader("🧪 Test Senaryosu")


if "random_test" not in st.session_state:
    st.session_state.random_test = None

if st.button("📸 Dataset'ten Rastgele Görsel Seç"):
    img_path, true_label = pick_random_dataset_image()
    if img_path is None:
        st.error("Dataset'ten görsel seçilemedi. Klasör yapısını kontrol edin.")
    else:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            st.error("Seçilen görsel okunamadı.")
        else:
            # Hybrid tahmin
            final_value, model_conf, ocr_value, ocr_conf, method = hybrid_predict(img_bgr)
            
            # Sonucu session_state'te tut
            st.session_state.random_test = {
                "img_path": img_path,
                "true_label": str(true_label),
                "pred_label": str(final_value),
                "model_conf": model_conf,
                "ocr_value": ocr_value,
                "ocr_conf": ocr_conf,
                "method": method,
            }

test_info = st.session_state.get("random_test")
if test_info:
    col_test1, col_test2 = st.columns([1, 1])
    with col_test1:
        st.image(test_info["img_path"], caption=f"Gerçek: {test_info['true_label']} TL")
    
    with col_test2:
        is_correct = test_info["true_label"] == test_info["pred_label"]
        st.metric("Tahmin", f"{test_info['pred_label']} TL", 
                  delta="Doğru ✅" if is_correct else "Yanlış ❌",
                  delta_color="normal" if is_correct else "inverse")
        st.metric("Güven", f"{test_info['model_conf']:.1f}%")
        if test_info["ocr_value"] is not None:
            st.caption(f"OCR: {test_info['ocr_value']} TL ({test_info['ocr_conf']*100:.1f}%)")
            st.caption(f"Yöntem: {test_info['method']}")
