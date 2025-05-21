import requests

# ✅ Test etmek istediğin görselin yolu
image_path = "test.JPG"

# ✅ API URL'si
url = "http://127.0.0.1:5000/predict"

# ✅ Görseli POST isteği olarak gönder
with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# ✅ Sonucu yazdır
print("🔍 Tahmin:", response.json())
