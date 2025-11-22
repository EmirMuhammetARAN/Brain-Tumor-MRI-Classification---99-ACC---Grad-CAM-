import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Model yükle
model = load_model("best_weights_balanced.h5")

# Sınıf isimleri
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Grad-CAM fonksiyonu
def get_gradcam(img_array, model, last_conv_layer_name="top_conv"):
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
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy(), pred_index.numpy()

def predict_and_explain(img):
    # Görüntüyü hazırla
    img_resized = cv2.resize(img, (300, 300))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    
    # Tahmin
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]
    
    # Grad-CAM
    heatmap, _ = get_gradcam(img_array, model)
    heatmap = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    superimposed = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
    
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