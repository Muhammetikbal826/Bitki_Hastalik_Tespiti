from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# ✅ Modeli yükle (model.h5 dosyan aynı klasörde olmalı!)
model = tf.keras.models.load_model("model.h5")

# ✅ Sınıf etiketleri (labels.txt'den çekebilirsin)
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Tomato_Bacterial_spot",
    "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus", "Tomato_healthy"
]

# ✅ Görseli işle
def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ✅ API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Görsel dosyası eksik!"}), 400

    file = request.files['file']
    image_data = file.read()
    processed_image = prepare_image(image_data)

    prediction = model.predict(processed_image)[0]  # tek örnek olduğu için [0] ekledik
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[predicted_index])

    # ✅ Tüm sınıfların skorlarını {label: score} sözlüğü olarak hazırla
    scores = {
        class_names[i]: float(prediction[i])
        for i in range(len(class_names))
    }

    return jsonify({
        "label": predicted_class,
        "confidence": confidence,
        "scores": scores  # 📊 Bar chart için
    })


# ✅ Sunucuyu başlat
if __name__ == "__main__":
    app.run(debug=True)
