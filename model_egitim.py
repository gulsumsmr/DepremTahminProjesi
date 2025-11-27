import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

print("Kütüphaneler yüklendi, LSTM modeli için işlem başlıyor...")

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

model = Sequential()

model.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu'))

model.add(Dense(1))

optimizer = Adam(learning_rate=0.001) 
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

model.summary()

print("\n--- LSTM Model Eğitimi Başlıyor ---")

gecmis = model.fit(
    X_train, 
    y_train, 
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test), 
    verbose=1
)


print("\n--- Test Sonuçları ---")
loss, mae = model.evaluate(X_test, y_test)
print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")


model.save('deprem_modeli.h5')
print("\nBAŞARILI! LSTM modeli 'deprem_modeli.h5' olarak kaydedildi.")
