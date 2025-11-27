# model_egitim.py (Strateji 2: LSTM Mimarisi ile)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

print("Kütüphaneler yüklendi, LSTM modeli için işlem başlıyor...")

# --- 1. HAZIRLANMIŞ VERİYİ YÜKLE ---
try:
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
except FileNotFoundError:
    print("HATA: '.npy' dosyaları bulunamadı. Önce veri hazırlık adımını tamamlamalısın.")
    exit()

print(f"Eğitim Verisi Şekli: {X_train.shape}")
print(f"Test Verisi Şekli: {X_test.shape}")

# --- 2. LSTM MODEL MİMARİSİ ---
# Bu mimari, sıralı verilerdeki desenleri öğrenmek için tasarlanmıştır.
model = Sequential()

# İlk LSTM Katmanı: Girdi şeklini (10, 4) olarak belirtiyoruz.
# return_sequences=True: Bir sonraki LSTM katmanına da bir dizi göndermesi gerektiğini belirtir.
model.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

# İkinci LSTM Katmanı: return_sequences=False (veya belirtilmemiş) çünkü sonraki katman normal bir Dense katmanı.
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))

# Tam Bağlantılı (Dense) Katman: LSTM'den gelen çıktıyı işlemek için.
model.add(Dense(50, activation='relu'))

# Çıkış Katmanı: Tek bir değer (Büyüklük) tahmin edecek.
model.add(Dense(1))

# Modeli Derle
optimizer = Adam(learning_rate=0.001) # LSTM için başlangıçta biraz daha yüksek bir oran deneyebiliriz.
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

model.summary()

print("\n--- LSTM Model Eğitimi Başlıyor ---")

# --- 3. EĞİTİM ---
gecmis = model.fit(
    X_train, 
    y_train, 
    epochs=50,  # LSTM daha hızlı öğrendiği için başlangıçta 50 epoch yeterli olabilir.
    batch_size=64,
    validation_data=(X_test, y_test), 
    verbose=1
)

# --- 4. PERFORMANS TESTİ ---
print("\n--- Test Sonuçları ---")
loss, mae = model.evaluate(X_test, y_test)
print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")

# --- 5. MODELİ KAYDET ---
model.save('deprem_modeli.h5')
print("\nBAŞARILI! LSTM modeli 'deprem_modeli.h5' olarak kaydedildi.")
