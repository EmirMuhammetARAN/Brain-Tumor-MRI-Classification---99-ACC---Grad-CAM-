import gradio as gr
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.models import Model
import cv2

# Sınıf isimleri
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Model yapısını reconstruct et
def build_model():
    img_size = (224, 224)
    inputs = tf.keras.Input(shape=img_size + (3,))
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(
        include_top=False, 
        weights="imagenet", 
        input_tensor=inputs, 
        pooling='max'
    )
    base_model.trainable = True
    
    x = base_model.output
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(class_names), activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# Model oluştur ve weights yükle
model = build_model()
model.load_weights("best_weights_balanced.h5")

# Son conv layer'ı otomatik bul
def get_last_conv_layer_name(model):
    """Find the last convolutional layer in the model"""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    return None

# Grad-CAM fonksiyonu
def get_gradcam(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize to 0-1
    heatmap_min = tf.math.reduce_min(heatmap)
    heatmap_max = tf.math.reduce_max(heatmap)
    heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + K.epsilon())
    
    return heatmap.numpy(), pred_index.numpy()

def predict_and_explain(img):
    # Görüntüyü hazırla
    img_resized = cv2.resize(img, (224, 224))
    
    # Gradio'dan gelen image 0-255 range'de
    # preprocess_input bu range'i normalize ediyor
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array.astype(np.float32))
    
    # Tahmin
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]
    
    # Grad-CAM - son conv layer'ı bul
    last_conv_layer_name = get_last_conv_layer_name(model)
    heatmap, _ = get_gradcam(img_array, model, last_conv_layer_name)
    heatmap = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
    # Heatmap'ı ters çevir: kırmızı = model odaklandığı yer
    heatmap = 1 - heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Original image for overlay (normalize to 0-255)
    img_for_display = cv2.resize(img, (224, 224))
    if img_for_display.max() <= 1.0:
        img_for_display = (img_for_display * 255).astype(np.uint8)
    
    # Overlay
    superimposed = cv2.addWeighted(img_for_display, 0.6, heatmap_colored, 0.4, 0)
    
    # Sonuçlar
    results = {class_names[i]: float(predictions[0][i]) for i in range(4)}
    
    return results, superimposed

# Gradio arayüzü
demo = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(label="MRI Görüntüsü Yükle"),
    outputs=[
        gr.Label(num_top_classes=4, label="Tahmin"),
        gr.Image(label="Grad-CAM Açıklaması")
    ],
    title="🧠 Beyin Tümörü MRI Sınıflandırma",
    description="""
    **EfficientNetB3 + Grad-CAM ile Açıklanabilir AI**
    
    Bu model beyin MRI görüntülerini 4 kategoride sınıflandırır:
    - **Glioma** - Glial hücrelerden kaynaklanan tümör
    - **Meningioma** - Meninks zarından kaynaklanan tümör  
    - **Pituitary** - Hipofiz bezi tümörü
    - **No Tumor** - Tümör yok
    
    Grad-CAM, modelin hangi bölgelere odaklandığını gösterir.
    
    ⚠️ *Bu araç sadece eğitim amaçlıdır, tıbbi teşhis için kullanılamaz.*
    """,
    examples=[],  # Örnek görüntü ekleyebilirsin
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()