# api.py (Final Versiyon - LSTM Uyumlu)

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import requests
import urllib3
from datetime import datetime, timedelta

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)
CORS(app)

model_path = 'deprem_modeli.h5'
try:
    model = load_model(model_path, compile=False)
    print("LSTM Modeli başarıyla yüklendi ve hazır.")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    model = None

def gelismis_canli_veri_getir(max_veri_sayisi=2000, gun_araligi=30):
    print(f"\n AFAD'dan son {gun_araligi} gün için en fazla {max_veri_sayisi} deprem verisi çekiliyor...")
    tum_veriler = []
    limit_her_istek = 500
    offset = 0
    bitis_tarihi = datetime.now()
    baslangic_tarihi = bitis_tarihi - timedelta(days=gun_araligi)
    start_str = baslangic_tarihi.strftime('%Y-%m-%dT%H:%M:%S')
    end_str = bitis_tarihi.strftime('%Y-%m-%dT%H:%M:%S')
    headers = {"User-Agent": "Mozilla/5.0"}
    while len(tum_veriler) < max_veri_sayisi:
        try:
            url = f"https://deprem.afad.gov.tr/apiv2/event/filter?start={start_str}&end={end_str}&orderby=timedesc&limit={limit_her_istek}&offset={offset}"
            r = requests.get(url, headers=headers, timeout=20, verify=False )
            if r.status_code == 200:
                gelen_veri = r.json()
                if not gelen_veri: break
                tum_veriler.extend(gelen_veri)
                offset += limit_her_istek
                if len(gelen_veri) < limit_her_istek: break
            else: break
        except Exception: break
    if not tum_veriler: return pd.DataFrame()
    df = pd.DataFrame(tum_veriler)
    rename_map = {'magnitude': 'Buyukluk', 'depth': 'Derinlik', 'latitude': 'Enlem', 'longitude': 'Boylam', 'location': 'Yer', 'date': 'Tarih'}
    df.rename(columns=rename_map, inplace=True)
    print(f"BAŞARILI! Toplam {len(df)} adet canlı deprem verisi çekildi.")
    return df

# Yedek veri kümesi
try:
    df_yedek = pd.read_csv('egitim_kumesi.csv') 
    print("Yedek veri kümesi (egitim_kumesi.csv) başarıyla yüklendi.")
except:
    df_yedek = pd.DataFrame()
    print("Yedek veri kümesi bulunamadı.")

@app.route('/tahmin-et', methods=['POST'])
def tahmin_et():
    if model is None:
        return jsonify({'error': 'Sunucu tarafında model yüklenemediği için tahmin yapılamıyor.'}), 500

    try:
        data = request.json
        secilen_enlem = float(data['enlem'])
        secilen_boylam = float(data['boylam'])

        df_referans = gelismis_canli_veri_getir(max_veri_sayisi=2000, gun_araligi=60) 
        df_referans = df_referans.iloc[::-1].reset_index(drop=True)

        kaynak_tipi = "YOK"

        if df_referans.empty or len(df_referans) < 10:
             return jsonify({'error': 'Tahmin için yeterli sayıda (en az 10) canlı deprem verisi bulunamadı.'}), 503

        kaynak_tipi = f"CANLI (AFAD - {len(df_referans)} Veri)"
        cols = ['Enlem', 'Boylam', 'Derinlik', 'Buyukluk']
        for col in cols:
            df_referans[col] = pd.to_numeric(df_referans[col], errors='coerce')
        df_referans.dropna(subset=cols, inplace=True)
        
        mesafeler = np.sqrt((df_referans['Enlem'] - secilen_enlem)**2 + (df_referans['Boylam'] - secilen_boylam)**2)
        en_yakin_index = mesafeler.idxmin()

        window_size = 10
        if en_yakin_index < window_size -1:
            start_index = 0
            print(f"Uyarı: En yakın depremin yeterli geçmişi yok. Veri setinin başından {window_size} veri kullanılıyor.")
        else:
            start_index = en_yakin_index - window_size + 1
        
        end_index = start_index + window_size
        
        pencere_df = df_referans.iloc[start_index:end_index]
        giris_verisi_penceresi = pencere_df[['Enlem', 'Boylam', 'Derinlik', 'Buyukluk']].values
        
        giris_verisi = np.expand_dims(giris_verisi_penceresi, axis=0)
        
        print(f"Modele gönderilen veri şekli: {giris_verisi.shape}") # Konsolda (1, 10, 4) görmelisin

        tahmin_sonucu = model.predict(giris_verisi)
        tahmin_buyukluk = float(tahmin_sonucu[0][0])

        en_yakin_deprem = df_referans.loc[en_yakin_index]
        girdi_buyukluk = float(en_yakin_deprem['Buyukluk'])
        girdi_derinlik = float(en_yakin_deprem['Derinlik'])
        referans_yer = str(en_yakin_deprem.get('Yer', 'Bölge Bilinmiyor'))
        referans_tarih = str(en_yakin_deprem.get('Tarih', '-'))

        response = {
            'gelecek_buyukluk': tahmin_buyukluk,
            'gecmis_buyukluk': girdi_buyukluk,
            'gecmis_derinlik': girdi_derinlik,
            'referans_yer': referans_yer,
            'referans_tarih': referans_tarih,
            'kaynak': kaynak_tipi,
        }
        return jsonify(response)

    except Exception as e:
        print(f"❌ Sunucu tarafında bir hata oluştu: {e}")
        import traceback
        hata_detay = traceback.format_exc()
        print(hata_detay)
        return jsonify({'error': f'Beklenmedik bir sunucu hatası: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
