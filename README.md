# 🌿 Bitki Hastalığı Tespit ve Tedavi Öneri Uygulaması

Bu mobil uygulama, tarım alanında bitki hastalıklarını hızlı ve doğru bir şekilde tespit etmek amacıyla Flutter ile geliştirilmiştir. Yapay zekâ destekli bu sistem, 15 farklı bitki hastalığı sınıfında eğitilmiş bir modeli sunucu üzerinden çalıştırarak, kullanıcıdan alınan görsel veriye göre anlık teşhis ve ilaç önerisinde bulunur.

## 🚀 Özellikler

- 📷 **Kamera ile Anlık Tespit:** Kullanıcılar bitkinin hastalıklı kısmını kamerayla tarayarak hızlıca analiz alabilir.
- 🖼️ **Galeriden Yükleme:** Kullanıcılar cihazlarından görsel seçerek de analiz gerçekleştirebilir.
- 🤖 **Yapay Zekâ Entegrasyonu:** Eğitim verisi ile %97 doğruluk oranına sahip 15 sınıflı bir model sunucu tarafında barındırılmakta ve REST API üzerinden mobil uygulamaya entegre edilmiştir.
- 💊 **Zirai İlaç Önerisi:** Hastalık tespiti sonrası uygun zirai ilaç önerisi kullanıcıya sunulur.
- 🌐 **Gerçek Zamanlı Sonuçlar:** Model tahmini saniyeler içinde yapılır, kullanıcıya anında gösterilir.

## 🧠 Yapay Zekâ Modeli

- **Model Tipi:** CNN tabanlı özel sınıflandırma modeli
- **Sınıf Sayısı:** 15 farklı bitki hastalığı
- **Doğruluk Oranı:** %97
- **Entegrasyon:** REST API ile Flutter uygulamasına bağlanmıştır

## ⚙️ Kullanılan Teknolojiler

| Katman               | Teknolojiler                  |
|----------------------|-------------------------------|
| Mobil Uygulama       | Flutter, Dart                 |
| Backend              | Python (Flask veya Django)    |
| AI Model             | TensorFlow/Keras              |
| Veri Entegrasyonu    | REST API, HTTP                |
| Kamera/Galeri        | image_picker                  |
| Local Storage        | shared_preferences (opsiyonel) |
| Durum Yönetimi       | Provider                      |



