"""
Derin öğrenme tabanlı eğitim script'i 

Amaç:
- `dataset/5,10,20,50,100,200` klasörlerinden görüntüleri okuyup
  basit ön işlemeden geçirmek
- MobileNetV2 tabanlı basit ve etkili model
- En iyi modeli `.keras` formatında kaydetmek
"""

import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# --- GENEL AYARLAR ---
DATASET_PATH = "dataset"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30

MODEL_PATH = "banknote_efficientnetv2b0.keras"
CLASS_NAMES_PATH = "class_names.npy"
METRICS_PATH = "training_metrics.json"


def create_datasets():
    """Basit ve etkili dataset oluşturma"""
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"Dataset klasörü bulunamadı: {DATASET_PATH}")

    print("📁 Dataset okunuyor...")

    # Toplam görüntü sayısını hesapla
    total_images = 0
    for root, _, files in os.walk(DATASET_PATH):
        image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        total_images += len(image_files)

    # Basit dataset oluştur - resize with pad kullan
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    print(f"🔢 Sınıflar: {class_names}")
    print(f"📊 Toplam görüntü: {total_images}")
    print(f"📊 Eğitim görüntüsü: {int(total_images * 0.8)}")
    print(f"📊 Doğrulama görüntüsü: {int(total_images * 0.2)}")
    
    # Sınıf isimlerini kaydet
    np.save(CLASS_NAMES_PATH, np.array(class_names))
    print(f"💾 Sınıf isimleri kaydedildi.")

    AUTOTUNE = tf.data.AUTOTUNE
    normalization_layer = layers.Rescaling(1.0 / 255.0)

    # ÇOK BASİT preprocessing - sadece normalizasyon ve minimal augmentasyon
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = normalization_layer(image)
        # Sadece yatay flip - başka bir şey yok!
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
        return image, label

    def preprocess_val(image, label):
        image = tf.cast(image, tf.float32)
        image = normalization_layer(image)
        return image, label

    train_ds = (
        train_ds.shuffle(2000)
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )
    
    val_ds = (
        val_ds.map(preprocess_val, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    train_images = int(total_images * 0.8)
    val_images = total_images - train_images

    return train_ds, val_ds, len(class_names), train_images, val_images, total_images, class_names


def build_model(num_classes: int, base_trainable: bool = False) -> tf.keras.Model:
    """MobileNetV2 tabanlı basit ve etkili model"""
    print(f"🧠 MobileNetV2 model oluşturuluyor... (base_trainable={base_trainable})")

    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = base_trainable

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=base_trainable)
    x = layers.GlobalAveragePooling2D()(x)
    # Çok basit mimari
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="banknote_mobilenetv2")

    # Optimal learning rates
    if base_trainable:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)  # Yüksek LR - öğrenmeye başlasın

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if not base_trainable:
        print(model.summary())
    return model


def calculate_detailed_metrics(model, val_ds, class_names):
    """
    Validation seti üzerinde detaylı metrikler hesaplar:
    - Confusion Matrix
    - Precision, Recall, F1-score (her sınıf için ve macro/weighted average)
    """
    print("\n📊 Detaylı metrikler hesaplanıyor (validation seti üzerinde)...")
    
    # Tüm validation seti üzerinde tahmin yap
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(pred_classes)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Precision, Recall, F1-score (her sınıf için)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro average (tüm sınıfların ortalaması)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted average (sınıf sayılarına göre ağırlıklı ortalama)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Sınıf bazında metrikleri dictionary olarak kaydet
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i])
        }
    
    # Confusion matrix'i list olarak kaydet (JSON serializable için)
    cm_list = cm.tolist()
    
    metrics_dict = {
        "confusion_matrix": cm_list,
        "class_metrics": class_metrics,
        "macro_avg": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1_score": float(f1_macro)
        },
        "weighted_avg": {
            "precision": float(precision_weighted),
            "recall": float(recall_weighted),
            "f1_score": float(f1_weighted)
        }
    }
    
    # Konsola özet yazdır
    print("\n📈 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    return metrics_dict


def train():
    print("🚀 Eğitim başlıyor...")
    print("=" * 60)

    train_ds, val_ds, num_classes, train_images, val_images, total_images, class_names = create_datasets()
    
    # Learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    # AŞAMA 1: Transfer Learning
    print("\n📌 AŞAMA 1: Transfer Learning (Base model frozen)")
    print("=" * 60)
    
    model = build_model(num_classes=num_classes, base_trainable=False)

    checkpoint_cb_stage1 = ModelCheckpoint(
        "model_stage1.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_accuracy",
        patience=15,  # Çok sabırlı
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb_stage1, early_stop_cb, reduce_lr],
        verbose=1,
    )

    print("\n✅ Aşama 1 tamamlandı.")
    best_val_acc_stage1 = max(history_stage1.history.get("val_accuracy", [0.0]))
    print(f"📊 En iyi validation accuracy (Aşama 1): %{best_val_acc_stage1 * 100:.2f}")
    
    # AŞAMA 2: Fine-tuning (sadece iyi sonuç varsa)
    history_stage2 = None
    if best_val_acc_stage1 > 0.5:  # %50'den fazlaysa fine-tuning yap
        print("\n📌 AŞAMA 2: Fine-tuning")
        print("=" * 60)
        
        try:
            model = tf.keras.models.load_model("model_stage1.keras")
            base_model = model.layers[1]
            base_model.trainable = True
            
            # Son 20 katmanı aç
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            
            print(f"🔓 Fine-tuning başlıyor...")

            checkpoint_cb_stage2 = ModelCheckpoint(
                MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            )

            early_stop_cb_stage2 = EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                mode="max",
                restore_best_weights=True,
                verbose=1,
            )

            reduce_lr_stage2 = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1,
            )

            history_stage2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=15,
                callbacks=[checkpoint_cb_stage2, early_stop_cb_stage2, reduce_lr_stage2],
                verbose=1,
            )

            best_val_acc_stage2 = max(history_stage2.history.get("val_accuracy", [0.0]))
            print(f"📊 En iyi validation accuracy (Aşama 2): %{best_val_acc_stage2 * 100:.2f}")
        except Exception as e:
            print(f"\n⚠️ Fine-tuning sırasında hata: {e}")
            print("Stage1 modelini final model olarak kullanıyoruz.")
            import shutil
            if os.path.exists("model_stage1.keras"):
                shutil.copy("model_stage1.keras", MODEL_PATH)
            history_stage2 = None
    else:
        # Stage1 modelini kullan
        print("\n⚠️ Aşama 1 doğruluğu düşük, fine-tuning atlanıyor.")
        print("Stage1 modelini final model olarak kullanıyoruz.")
        import shutil
        if os.path.exists("model_stage1.keras"):
            shutil.copy("model_stage1.keras", MODEL_PATH)

    # Geçici dosyayı sil (sadece final model kaydedildiyse)
    if os.path.exists("model_stage1.keras") and os.path.exists(MODEL_PATH):
        try:
            os.remove("model_stage1.keras")
        except:
            pass

    print(f"\n💾 Model '{MODEL_PATH}' dosyasına kaydedildi.")

    # Final modeli yükle ve detaylı metrikleri hesapla
    print("\n" + "=" * 60)
    detailed_metrics = {}
    try:
        final_model = tf.keras.models.load_model(MODEL_PATH)
        
        # Validation seti üzerinde detaylı metrikler hesapla
        detailed_metrics = calculate_detailed_metrics(final_model, val_ds, class_names)
        print("✅ Detaylı metrikler hesaplandı.")
    except Exception as e:
        print(f"⚠️ Detaylı metrikler hesaplanamadı: {e}")

    # Metrikleri kaydet
    metrics = {}
    if history_stage1 is not None and history_stage1.history:
        hist1 = history_stage1.history
        hist2 = history_stage2.history if history_stage2 is not None and history_stage2.history else {}
        
        metrics = {
            "epochs_run_stage1": len(hist1.get("loss", [])),
            "epochs_run_stage2": len(hist2.get("loss", [])) if hist2 else 0,
            "epochs_run_total": len(hist1.get("loss", [])) + (len(hist2.get("loss", [])) if hist2 else 0),
            "best_train_accuracy_stage1": float(max(hist1.get("accuracy", [0.0]))),
            "best_val_accuracy_stage1": float(max(hist1.get("val_accuracy", [0.0]))),
            "best_train_accuracy_stage2": float(max(hist2.get("accuracy", [0.0]))) if hist2 else 0.0,
            "best_val_accuracy_stage2": float(max(hist2.get("val_accuracy", [0.0]))) if hist2 else 0.0,
            "best_train_accuracy": float(max(hist2.get("accuracy", [0.0]))) if hist2 and hist2.get("accuracy") else float(max(hist1.get("accuracy", [0.0]))),
            "best_val_accuracy": float(max(hist2.get("val_accuracy", [0.0]))) if hist2 and hist2.get("val_accuracy") else float(max(hist1.get("val_accuracy", [0.0]))),
        }

    metrics.update({
        "total_images": int(total_images),
        "train_images": int(train_images),
        "val_images": int(val_images),
        "num_classes": int(num_classes),
    })
    
    # Detaylı metrikleri ekle
    if detailed_metrics:
        metrics["detailed_metrics"] = detailed_metrics

    try:
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"📊 Metrikler '{METRICS_PATH}' dosyasına kaydedildi.")
    except Exception as exc:
        print(f"⚠️ Metrikler kaydedilemedi: {exc}")

    return history_stage2 if history_stage2 else history_stage1


if __name__ == "__main__":
    train()
