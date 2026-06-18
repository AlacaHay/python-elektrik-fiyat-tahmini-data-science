import pandas as pd
import numpy as np
import joblib
import datetime
from lightgbm import LGBMRegressor
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import requests
import sys

# Terminal çıktılarının kodlamasını zorla UTF-8 yapıyoruz
sys.stdout.reconfigure(encoding='utf-8')

# ---------------- 1. VERİ OKUMA VE DÜZENLEME ----------------
df = pd.read_csv('ptf_25_26.csv', sep=';', encoding='utf-8')
df['PTF'] = df['PTF (TL/MWh)'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
df['Datetime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M')
df = df.sort_values('Datetime').reset_index(drop=True)

# ---------------- 2. DOĞALGAZ (GRF) VERİSİNİ ENTEGRE ETME ----------------
try:
    df_grf = pd.read_csv('grf_25_26.csv', sep=';', encoding='utf-8')
    # Fiyatları formattan kurtarıp sayısal değere çevirme
    df_grf['GRF'] = df_grf['GRF (TL/1000Sm3)'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    df_grf['Date'] = pd.to_datetime(df_grf['Gaz Günü'], format='%d.%m.%Y').dt.date
    
    # Ana tablo (PTF) ile sadece Tarih (Date) bazında eşleştirme
    df['Date'] = df['Datetime'].dt.date
    df = pd.merge(df, df_grf[['Date', 'GRF']], on='Date', how='left')
    
    # Hafta sonu ve tatillerdeki boşlukları piyasa kuralına göre doldurma
    df['GRF'] = df['GRF'].ffill().bfill() 
    df.drop('Date', axis=1, inplace=True)   # Zaten datetime üz. birleştirdik tekrardan grf den gelen Date anlamsiz oldu.
    print("Doğalgaz (GRF) verisi eksiksiz olarak başarıyla eklendi!\n")
except Exception as e:
    print(f"GRF verisi eklenirken hata: {e}")

print("Tatil Etkisi Analizi:\n")

# ---------------- 3. TATİL VE HAFTA SONU ETKİSİ ----------------
manuel_tatiller = [
    '2025-01-01', '2025-03-29', '2025-03-30', '2025-03-31', '2025-04-01', 
    '2025-04-23', '2025-05-01', '2025-05-19', '2025-07-15', '2025-08-30', '2025-10-29',
    '2026-01-01', '2026-03-19', '2026-03-20', '2026-03-21', 
    '2026-04-23', '2026-05-01', '2026-05-19', '2026-05-26', '2026-05-27', 
    '2026-05-28', '2026-05-29', '2026-07-15', '2026-08-30', '2026-10-29'
]

tatil_tarihleri = pd.to_datetime(manuel_tatiller).date
df['Is_Holiday'] = df['Datetime'].dt.date.isin(tatil_tarihleri).astype(int)
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Is_Weekend'] = df['Datetime'].dt.dayofweek.isin([5, 6]).astype(int)

print(df.groupby('Is_Holiday')['PTF'].agg(['mean', 'median', 'count']))

# Hafta içi / Hafta sonu Tatil Boxplot Grafiği
plt.figure(figsize=(10, 6))
sns.boxplot(x='Is_Holiday', y='PTF', data=df, hue="Is_Holiday", palette='Set2', legend=False)
plt.title('Tatil Günleri vs Normal Günlerde PTF Dağılımı')
plt.xlabel('Durum (0: Normal Gün, 1: Resmi Tatil)')
plt.ylabel('PTF (TL/MWh)')
plt.show()

# ---------------- 4. HAVA DURUMU VERİSİNİN ÇEKİLMESİ ----------------
print("Hava durumu verileri indiriliyor, lütfen bekleyin...")
start_date = df['Datetime'].min().strftime('%Y-%m-%d')
end_date = df['Datetime'].max().strftime('%Y-%m-%d')

sehirler = {
    "Istanbul": {"lat": 41.0082, "lon": 28.9784},
    "Ankara": {"lat": 39.9199, "lon": 32.8543},
    "Izmir": {"lat": 38.4127, "lon": 27.1384}
}

hava_durumu_listesi = []
for sehir, kord in sehirler.items():
    url = "https://archive-api.open-meteo.com/v1/archive"
    guncel_bitis = end_date
    
    while True:
        params = {
            "latitude": kord["lat"],
            "longitude": kord["lon"],
            "start_date": start_date,
            "end_date": guncel_bitis,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
            "timezone": "Europe/Istanbul"
        }
        response = requests.get(url, params=params)
        data = response.json()  #openmeteodan gelen verileri json formatında okuyoruz.
        
        if "hourly" in data:
            break    #veri gelirse whiledan cik 
        else:
            bitis_obj = datetime.datetime.strptime(guncel_bitis, '%Y-%m-%d')
            guncel_bitis = (bitis_obj - datetime.timedelta(days=1)).strftime('%Y-%m-%d')    #veri istedigimiz gun icin yoksa onceki gunun verisi cekilir.
            
    temp_df = pd.DataFrame({
        "Datetime": pd.to_datetime(data["hourly"]["time"]),
        f"Temp_{sehir}": data["hourly"]["temperature_2m"],
        f"Hum_{sehir}": data["hourly"]["relative_humidity_2m"],
        f"Wind_{sehir}": data["hourly"]["wind_speed_10m"],
        f"Cloud_{sehir}": data["hourly"]["cloud_cover"]
    })
    hava_durumu_listesi.append(temp_df)

df_weather = hava_durumu_listesi[0]
for d in hava_durumu_listesi[1:]:
    df_weather = pd.merge(df_weather, d, on="Datetime", how="inner")

df_weather['Temperature_2m'] = df_weather[['Temp_Istanbul', 'Temp_Ankara', 'Temp_Izmir']].mean(axis=1)
df_weather['Relative_Humidity'] = df_weather[['Hum_Istanbul', 'Hum_Ankara', 'Hum_Izmir']].mean(axis=1)
df_weather['Cloud_Cover'] = df_weather[['Cloud_Istanbul', 'Cloud_Ankara', 'Cloud_Izmir']].mean(axis=1)

df_weather = df_weather[[
    'Datetime', 'Temperature_2m', 'Relative_Humidity', 'Cloud_Cover',
    'Temp_Istanbul', 'Temp_Ankara', 'Temp_Izmir',
    'Wind_Istanbul', 'Wind_Ankara', 'Wind_Izmir',
    'Cloud_Istanbul', 'Cloud_Ankara', 'Cloud_Izmir',
    'Hum_Istanbul', 'Hum_Ankara', 'Hum_Izmir'
]]

# ---------------- 5. BİRLEŞTİRME VE EKSİK VERİ TAMAMLAMA ----------------
df = pd.merge(df, df_weather, on='Datetime', how='left')
df.ffill(inplace=True)  # True olmasi orijinal tabloyu degistirir demek.

# ---------------- 6. ÖZELLİK MÜHENDİSLİĞİ ----------------
hdd_baz_sicaklik = 14.0   # Hava 14 dereceden soguksa isitici calisir.
cdd_baz_sicaklik = 25.0   # 25 ustu ise klimalar calisir. 
df['HDD'] = df['Temperature_2m'].apply(lambda x: hdd_baz_sicaklik - x if x < hdd_baz_sicaklik else 0) # Isitici calisma durumu.
df['CDD'] = df['Temperature_2m'].apply(lambda x: x - cdd_baz_sicaklik if x > cdd_baz_sicaklik else 0) # Sogutucu calisma durumu.

print("\nZaman ve geçmiş fiyat özellikleri ekleniyor...")
df['Hour'] = df['Datetime'].dt.hour
df['Month'] = df['Datetime'].dt.month
df['DayOfYear'] = df['Datetime'].dt.dayofyear

def saat_grubu(hour):
    if 0 <= hour < 6: return 'Gece_Derin'
    elif 6 <= hour < 9: return 'Sabah_Gecis'
    elif 9 <= hour < 17: return 'Gunduz_Mesai'
    elif 17 <= hour < 22: return 'Pik_Aksam'
    else: return 'Gece_Yatis'

df['Time_Period'] = df['Hour'].apply(saat_grubu).astype('category')
df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)  # Bu sin cos donusumleri 23.00 ile 00.00 arasinda iliski old. anlamassi icin gerekli.
df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

# Lag (Gecikme) Özellikleri
df['PTF_Lag24'] = df['PTF'].shift(24)    # 1 Gun onceki veri. 
df['PTF_Lag168'] = df['PTF'].shift(168)  #1 Hafta onceki veri.
df['PTF_Rolling_24'] = df['PTF'].shift(24).rolling(window=24).mean()  #  1 Gun oncenin verisinin ortalamasini alir.
df['Wind_Weekend'] = df['Wind_Izmir'] * df['Is_Weekend']
df['Temp_Mesai'] = df['Temperature_2m'] * (df['Time_Period'] == 'Gunduz_Mesai').astype(int)

#  TATİL ETKİSİ KODLARI
df['Tatil_Lag_Etkisi'] = df['Is_Holiday'] * df['PTF_Lag24']       # Etkilesim Ozellikleri cikariliyor.
df['Tatil_ve_Haftasonu'] = df['Is_Holiday'] * df['Is_Weekend']

df = df.dropna().reset_index(drop=True)
print("Eksik veriler temizlendi. Veri seti model eğitimine %100 hazır!")

# ---------------- 7. EĞİTİM ÖNCESİ KORELASYON ANALİZİ ----------------
print("\nEğitim öncesi özelliklerin PTF ile ilişkisi (Korelasyon) hesaplanıyor...")
# GRF KORELASYONA EKLENDİ
sayisal_kolonlar = [
    'PTF', 'GRF', 'Temperature_2m', 'Temp_Istanbul', 'Temp_Ankara', 'Temp_Izmir', 
    'HDD', 'CDD', 'Wind_Istanbul', 'Wind_Ankara', 'Wind_Izmir',
    'Cloud_Cover', 'Relative_Humidity', 'Is_Holiday', 'DayOfWeek', 'Is_Weekend',
    'Hour', 'PTF_Lag24', 'Tatil_Lag_Etkisi', 'Tatil_ve_Haftasonu'
]

korelasyon = df[sayisal_kolonlar].corr()  # Ozelliklerin hedef degiskenle olan etkilesimleri cikartilir.
ptf_korelasyon = korelasyon[['PTF']].sort_values(by='PTF', ascending=False)

plt.figure(figsize=(8, 12))
sns.heatmap(ptf_korelasyon, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title("Değişkenlerin PTF Üzerindeki Etkisi (Korelasyon)")
plt.tight_layout()
plt.show()

# ---------------- 8. MODEL EĞİTİMİ İÇİN HAZIRLIK ----------------
print("\n--- Model Eğitim Süreci Başlıyor ---")
tr_harfler = {'ş':'s', 'ç':'c', 'ı':'i', 'ğ':'g', 'ü':'u', 'ö':'o', 'Ş':'S', 'Ç':'C', 'İ':'I', 'Ğ':'G', 'Ü':'U', 'Ö':'O', ' ':'_'}
df.columns = [kolon.translate(str.maketrans(tr_harfler)) for kolon in df.columns]


ozellikler = [
    'GRF',
    'Is_Holiday', 'DayOfWeek', 'Is_Weekend', 
    'Temperature_2m', 'Temp_Istanbul', 'Temp_Ankara', 'Temp_Izmir',
    'Wind_Istanbul', 'Wind_Ankara', 'Wind_Izmir',
    'Cloud_Cover', 'Relative_Humidity',
    
    'HDD', 'CDD', 'Hour', 'Month', 'DayOfYear', 'Time_Period',
    'hour_sin', 'hour_cos', 'PTF_Lag24', 'PTF_Lag168',
    'PTF_Rolling_24', 'Temp_Mesai', 'Wind_Weekend',
    
    'Tatil_Lag_Etkisi', 'Tatil_ve_Haftasonu' 
]

X = df[ozellikler]
y = df['PTF']

# ---------------- 9. TRAIN / TEST AYRIMI ----------------
split_point = int(len(df) * 0.8) 
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]   
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# ---------------- 10. MODEL KURULUMU (GÜNCELLENMİŞ PARAMETRELER) ----------------
print("Yapay zeka piyasa kurallarını öğreniyor...")
y_train_log = np.log1p(y_train)

model = LGBMRegressor(
    n_estimators=500,        # Ağaç sayısı çok fazla ağaç overfittinge sebep olur.
    learning_rate=0.015,     # Öğrenme Süresi.
    max_depth=10,            # Ağaç derinliği ağacın dallanma mik.
    num_leaves=45,           # Modelin ezberlemesini önlemek için düşük bir değer seçilir.
    random_state=42,         # hep aynı sonuçlar üretir
    colsample_bytree=0.6,    # her defada özelliklerin sadece %60' rastgele seçilir.
    subsample=0.8,           # Verinin hep %80 ini inceler verinin farklı yüzleri görülür. Tek yüksek fiyatlara saplanıp kalmaz.
    reg_alpha=0.1,           # L1 Düzenlileştirme (Gereksiz verileri tamamen susturur).
    reg_lambda=0.5,          # L2 Düzenlileştirme tek özelliğin modelde diktatör olmasını engellemek için. Cok yuksek degerlerde etkili.
    verbose=-1               # Arkaplanda dönen işlemleri gizler.
)

model.fit(X_train, y_train_log, categorical_feature=['Time_Period'])

# ---------------- 11. TEST SONUÇLARI VE BAŞARI ÖLÇÜMÜ ----------------
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_pred[y_pred < 0] = 0   #negatif tahminleri 0'a sabitler.

r2 = r2_score(y_test, y_pred)                
mae = mean_absolute_error(y_test, y_pred)    

print(f"\n--- ANALİZ SONUÇLARI ---")
print(f"Test R² Skoru: %{r2*100:.2f} (Piyasa oynaklığını açıklama oranı)")
print(f"Ortalama Hata (MAE): {mae:.2f} TL5/MWh (Tahminlerin ortalama sapması)")

# ---------------- 12. FİNAL GÖRSELLEŞTİRME ----------------
plt.figure(figsize=(14, 7))
plt.plot(y_test.values[-240:], label='Gerçek PTF (EPİAŞ)', color='blue', alpha=0.5, linewidth=2)
plt.plot(y_pred[-240:], label='Yapay Zeka Tahmini', color='red', linestyle='--', linewidth=2)
plt.title('EPİAŞ PTF Tahmin Başarısı (Son 10 Günlük Kesit)')
plt.xlabel('Zaman (Saat)')
plt.ylabel('TL/MWh')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

joblib.dump(model, 'ptf_sampiyon_model.pkl')
print("Model başarıyla 'ptf_sampiyon_model.pkl' olarak kaydedildi!")
