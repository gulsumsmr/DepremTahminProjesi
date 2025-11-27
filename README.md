# Yapay Zeka Destekli İnteraktif Deprem Tahmin Projesi

Bu proje, AFAD'dan alınan canlı deprem verilerini kullanarak, belirli bir coğrafi konum için potansiyel deprem büyüklüğünü tahmin eden yapay zeka tabanlı bir web uygulamasıdır. Proje, bir LSTM (Long Short-Term Memory) sinir ağı modeli ile geliştirilmiş ve sonuçları interaktif bir harita üzerinde sunan bir React arayüzüne sahiptir.

<img width="1917" height="908" alt="image" src="https://github.com/user-attachments/assets/984001c5-5051-4a37-ba23-73c065da3454" />


---

## Projenin Özellikleri

- **Canlı Veri Entegrasyonu:** AFAD'ın resmi API'sinden anlık olarak son deprem verilerini çeker.
- **Gelişmiş Yapay Zeka Modeli:** Zaman serisi verileri için özel olarak tasarlanmış bir **LSTM (Long Short-Term Memory)** modeli kullanarak, geçmiş deprem desenlerine dayalı tahminler yapar.
- **İnteraktif Harita:** Kullanıcıların harita üzerinde istedikleri bir noktayı seçerek o bölge için tahmin alabilmelerini sağlar.

---

##  Kullanılan Teknolojiler

- **Backend & AI:**
  - Python
  - TensorFlow / Keras (LSTM Modeli için)
  - Flask (API Servisi için)
  - Pandas & NumPy (Veri işleme için)
- **Frontend:**
  - JavaScript
  - React
  - Leaflet.js (İnteraktif harita için)
  - Material-UI (Arayüz bileşenleri için)
  - Axios (API istekleri için)

---

## Modelin Çalışma Mantığı

1.  Kullanıcı haritadan bir nokta seçtiğinde, bu konum bilgisi (enlem/boylam ) Flask API'sine gönderilir.
2.  API, AFAD sunucusundan son 60 güne ait deprem verilerini çeker.
3.  Çekilen veriler arasından, kullanıcının seçtiği noktaya en yakın deprem bulunur.
4.  LSTM modeli, tek bir deprem yerine bir "desen" aradığı için, bulunan bu en yakın deprem ve ondan önceki 9 deprem alınarak **10 adımlık bir zaman serisi penceresi** oluşturulur.
5.  Bu 10'luk pencere, eğitilmiş `deprem_modeli.h5` dosyasına girdi olarak verilir.
6.  Model, bu desene dayanarak bir sonraki potansiyel depremin büyüklüğünü tahmin eder.
7.  Tahmin sonucu ve referans alınan en yakın depremin bilgileri, React arayüzüne geri gönderilerek kullanıcıya gösterilir.

