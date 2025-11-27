import pandas as pd
import numpy as np
import os

dosya_adi = 'data.csv'
print(f"'{dosya_adi}' dosyası okunuyor...")
if not os.path.exists(dosya_adi):
    print(f"HATA: '{dosya_adi}' bulunamadı."); exit()

try:
    df = pd.read_csv(dosya_adi, sep=',')
    if len(df.columns) < 2: df = pd.read_csv(dosya_adi, sep=';')
    df.columns = df.columns.str.strip()
except Exception as e:
    print(f"Dosya okuma hatası: {e}"); exit()

rename_dict = {'Latitude': 'Enlem', 'Longitude': 'Boylam', 'Depth': 'Derinlik', 'Magnitude': 'Buyukluk'}
df.rename(columns=rename_dict, inplace=True)
df = df[['Enlem', 'Boylam', 'Derinlik', 'Buyukluk']]

orjinal_sayi = len(df)
min_buyukluk = 2.0
df = df[df['Buyukluk'] >= min_buyukluk]
print(f"Veri Temizlendi: {min_buyukluk}'den küçük {orjinal_sayi - len(df)} deprem çıkarıldı.")

df = df.iloc[::-1].reset_index(drop=True)

print("\nVeri, LSTM için 'pencereler' halinde hazırlanıyor...")

window_size = 10  
X_data, y_data = [], []
for i in range(len(df) - window_size):
    window = df.iloc[i : i + window_size][['Enlem', 'Boylam', 'Derinlik', 'Buyukluk']].values
    X_data.append(window)

    target = df.iloc[i + window_size]['Buyukluk']
    y_data.append(target)


X = np.array(X_data)
y = np.array(y_data)


print(f"Hazırlanan veri şekli: {X.shape}")


split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Eğitim verisi: {len(X_train)} pencere")
print(f"Test verisi: {len(X_test)} pencere")


np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("\n--- İŞLEM TAMAM ---")
print("LSTM için hazırlanan veriler 'X_train.npy', 'y_train.npy' vb. olarak kaydedildi.")
