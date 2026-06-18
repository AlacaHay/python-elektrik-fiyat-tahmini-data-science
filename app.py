from matplotlib.font_manager import stretch_dict
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import datetime
import holidays
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. SAYFA AYARLARI VE BAŞLIK
# ==========================================
st.set_page_config(page_title="Enerji PTF Tahmin Paneli", page_icon="⚡", layout="wide")

st.title("🌸 Yapay Zeka Destekli Enerji Fiyat Tahmini ve Üretim Planlama")
st.markdown("Bu panel, makine öğrenmesi modeli kullanarak yarının elektrik fiyatlarını tahmin eder ve fabrika üretimi için en maliyetsiz ardışık saatleri bulur.")


# 2. MODELİ YÜKLEME (Önbelleğe Alma )

@st.cache_resource
def modeli_yukle():
    klasor_yolu = os.path.dirname(os.path.abspath(__file__))
    model_yolu = os.path.join(klasor_yolu, 'ptf_sampiyon_model.pkl')
    csv_yolu = os.path.join(klasor_yolu, 'ptf_25_26.csv')
    grf_yolu = os.path.join(klasor_yolu, 'grf_25_26.csv') # GRF DOSYASI EKLENDİ
    
    try:
        model = joblib.load(model_yolu)
        return model, csv_yolu, grf_yolu
    except Exception as e:
        st.error(f"Model yüklenemedi! Dosya yolu hatası: {e}")
        st.stop()

model, CSV_DOSYASI, GRF_DOSYASI = modeli_yukle()

# ==========================================
# 3. YAN MENÜ (Kullanıcı Girdileri)
# ==========================================
st.sidebar.header("⚙️ Planlama Ayarları")
st.sidebar.markdown("Makinenizin ne kadar süre aralıksız çalışacağını seçin:")

secilen_saat = st.sidebar.slider("Çalışma Süresi (Saat)", min_value=1, max_value=24, value=3, step=1)   #default 3

if st.sidebar.button("🚀 Tahmini Çalıştır"):   #calistirma 
    
    with st.spinner('Yapay zeka piyasa verilerini analiz ediyor, lütfen bekleyin...'):
       
        # 4. TAHMİN KISMI 
     
        bugun = datetime.date.today()
        yarin = bugun + datetime.timedelta(days=1)   #sonraki gunu tahmin ediyor.
        hedef_tarih_str = yarin.strftime('%Y-%m-%d')
        lag24_tarih = bugun.strftime('%d.%m.%Y')
        lag168_tarih = (yarin - datetime.timedelta(days=7)).strftime('%d.%m.%Y')
        
        try:
            df_csv = pd.read_csv(CSV_DOSYASI, sep=';', encoding='utf-8')
            df_csv['PTF'] = df_csv['PTF (TL/MWh)'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
            dunku_veriler = df_csv[df_csv['Tarih'] == lag24_tarih]['PTF'].values   #tahminden onceki gun ve 1 hafta oncesi cekilir
            gecen_hafta_veriler = df_csv[df_csv['Tarih'] == lag168_tarih]['PTF'].values
            
            if len(dunku_veriler) != 24 or len(gecen_hafta_veriler) != 24:
                st.error(f"HATA: CSV dosyasında {lag24_tarih} veya {lag168_tarih} verisi eksik!")
                st.stop()  # GUvenlik onlemi
        except Exception as e:
            st.error(f"CSV okuma hatası! Detay: {e}")
            st.stop()

        # GRF Verisini Çekme
        try:
            df_grf = pd.read_csv(GRF_DOSYASI, sep=';', encoding='utf-8')
            df_grf['GRF'] = df_grf['GRF (TL/1000Sm3)'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
            df_grf['Date'] = pd.to_datetime(df_grf['Gaz Günü'], format='%d.%m.%Y').dt.date
            # Sadece bugünün GRF verisini alıyoruz (Tahmin yarın için ama GRF en son bugün biliniyor)
            bugun_grf_df = df_grf[df_grf['Date'] <= bugun]
            
            if not bugun_grf_df.empty:
               guncel_grf = bugun_grf_df.iloc[-1]['GRF']
            else:
               guncel_grf = 12500.0 # B planı varsayılan değer
        except Exception as e:
            st.warning(f"GRF verisi okunamadı, varsayılan (12500) değer kullanılacak. Hata: {e}")
            guncel_grf = 12500.0

        # Hava Durumu Çekme (B Planlı)
        sehirler = {"Istanbul": {"lat": 41.0082, "lon": 28.9784}, "Ankara": {"lat": 39.9199, "lon": 32.8543}, "Izmir": {"lat": 38.4127, "lon": 27.1384}}
        hava_listesi = []
        api_basarili = True

        for sehir, kord in sehirler.items():
            url = "https://api.open-meteo.com/v1/forecast"
            params = {"latitude": kord["lat"], "longitude": kord["lon"], "start_date": hedef_tarih_str, "end_date": hedef_tarih_str, "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover", "timezone": "Europe/Istanbul"}
            try:
                r = requests.get(url, params=params, timeout=5).json()
                if "error" in r or "hourly" not in r:
                    api_basarili = False; break
                temp_df = pd.DataFrame({"Datetime": pd.to_datetime(r["hourly"]["time"]), f"Temp_{sehir}": r["hourly"]["temperature_2m"], f"Hum_{sehir}": r["hourly"]["relative_humidity_2m"], f"Wind_{sehir}": r["hourly"]["wind_speed_10m"], f"Cloud_{sehir}": r["hourly"]["cloud_cover"]})
                hava_listesi.append(temp_df)
            except:
                api_basarili = False; break

        if not api_basarili or len(hava_listesi) != 3:
            st.warning("⚠️ İnternetten hava durumu çekilemedi. Standart bahar havası baz alınarak hesaplama yapılıyor.")
            saatler = pd.date_range(start=hedef_tarih_str, periods=24, freq='h')
            df_tahmin = pd.DataFrame({"Datetime": saatler})
            for sehir in sehirler.keys():
                df_tahmin[f"Temp_{sehir}"] = 22.0; df_tahmin[f"Hum_{sehir}"] = 50.0; df_tahmin[f"Wind_{sehir}"] = 10.0; df_tahmin[f"Cloud_{sehir}"] = 20.0
        else:
            df_tahmin = hava_listesi[0]
            for d in hava_listesi[1:]: df_tahmin = pd.merge(df_tahmin, d, on="Datetime", how="inner")

        # Özellik Mühendisliği
        df_tahmin['Temperature_2m'] = df_tahmin[['Temp_Istanbul', 'Temp_Ankara', 'Temp_Izmir']].mean(axis=1)
        df_tahmin['Relative_Humidity'] = df_tahmin[['Hum_Istanbul', 'Hum_Ankara', 'Hum_Izmir']].mean(axis=1)
        df_tahmin['Cloud_Cover'] = df_tahmin[['Cloud_Istanbul', 'Cloud_Ankara', 'Cloud_Izmir']].mean(axis=1)
        df_tahmin['HDD'] = df_tahmin['Temperature_2m'].apply(lambda x: 14.0 - x if x < 14.0 else 0)
        df_tahmin['CDD'] = df_tahmin['Temperature_2m'].apply(lambda x: x - 25.0 if x > 25.0 else 0)
        
        # Tatil ve Bayram Kontrolü
        tr_holidays = holidays.Turkey(years=[yarin.year])
        
        # Manuel Tatilleri Uygulama (Uygulama içine de aynı mantığı entegre ediyoruz)
        manuel_tatiller_tarihler = [
            datetime.date(2025, 1, 1), datetime.date(2025, 3, 29), datetime.date(2025, 3, 30), datetime.date(2025, 3, 31), datetime.date(2025, 4, 1),
            datetime.date(2025, 4, 23), datetime.date(2025, 5, 1), datetime.date(2025, 5, 19), datetime.date(2025, 7, 15), datetime.date(2025, 8, 30), datetime.date(2025, 10, 29),
            datetime.date(2026, 1, 1), datetime.date(2026, 3, 19), datetime.date(2026, 3, 20), datetime.date(2026, 3, 21), 
            datetime.date(2026, 4, 23), datetime.date(2026, 5, 1), datetime.date(2026, 5, 19), datetime.date(2026, 5, 26), datetime.date(2026, 5, 27), 
            datetime.date(2026, 5, 28), datetime.date(2026, 5, 29), datetime.date(2026, 7, 15), datetime.date(2026, 8, 30), datetime.date(2026, 10, 29)
        ]

        df_tahmin['Is_Holiday'] = df_tahmin['Datetime'].dt.date.apply(lambda x: 1 if x in tr_holidays or x in manuel_tatiller_tarihler else 0)

        df_tahmin['DayOfWeek'] = df_tahmin['Datetime'].dt.dayofweek 
        df_tahmin['Is_Weekend'] = df_tahmin['DayOfWeek'].isin([5, 6]).astype(int)
        df_tahmin['Hour'] = df_tahmin['Datetime'].dt.hour
        df_tahmin['Month'] = df_tahmin['Datetime'].dt.month
        df_tahmin['DayOfYear'] = df_tahmin['Datetime'].dt.dayofyear
        df_tahmin['hour_sin'] = np.sin(2 * np.pi * df_tahmin['Hour'] / 24)
        df_tahmin['hour_cos'] = np.cos(2 * np.pi * df_tahmin['Hour'] / 24)
        df_tahmin['PTF_Lag24'] = dunku_veriler
        df_tahmin['PTF_Lag168'] = gecen_hafta_veriler
        df_tahmin['PTF_Rolling_24'] = np.mean(dunku_veriler)
        
        # Yeni Eklenen Özellikleri Uygulama:
        df_tahmin['GRF'] = guncel_grf 
        df_tahmin['Tatil_Lag_Etkisi'] = df_tahmin['Is_Holiday'] * df_tahmin['PTF_Lag24']
        df_tahmin['Tatil_ve_Haftasonu'] = df_tahmin['Is_Holiday'] * df_tahmin['Is_Weekend']
        df_tahmin['Wind_Weekend'] = df_tahmin['Wind_Izmir'] * df_tahmin['Is_Weekend']

        def s_grup(h):
            if 0 <= h < 6: return 'Gece_Derin'
            elif 6 <= h < 9: return 'Sabah_Gecis'
            elif 9 <= h < 17: return 'Gunduz_Mesai'
            elif 17 <= h < 22: return 'Pik_Aksam'
            else: return 'Gece_Yatis'
            
        df_tahmin['Time_Period'] = df_tahmin['Hour'].apply(s_grup).astype('category')
        df_tahmin['Temp_Mesai'] = df_tahmin['Temperature_2m'] * (df_tahmin['Time_Period'] == 'Gunduz_Mesai').astype(int)

        #  Liste Model İle Uyumlu Hale Getirildi
        final_features = [
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
        
        # Sütun isimlerini küçük harf/büyük harf sorunu yaratmaması için orjinal isminde tuttum, eğer model küçük harf bekliyorsa hata verir ama yukarıda düzeltmiştik bunu
        
        df_tahmin['tahmin_ptf'] = np.expm1(model.predict(df_tahmin[final_features])).round(2)

        # Dinamik Saat Hesaplama
        df_tahmin['Maliyet_Toplam'] = df_tahmin['tahmin_ptf'].rolling(window=secilen_saat).sum()   #ardışık secilen saat kadar satırları toplar
        en_ucuz_bitis = int(df_tahmin['Maliyet_Toplam'].idxmin())   # bu toplamdan minimum olan deger bulunur ve o 3 saat rsdışık secilir.
        en_ucuz_baslangic = int(en_ucuz_bitis - (secilen_saat - 1))
        ardisik_ucuzlar = df_tahmin.iloc[en_ucuz_baslangic : en_ucuz_bitis + 1]
        toplam_maliyet = ardisik_ucuzlar['tahmin_ptf'].sum()

        # ==========================================
        # 5. EKRANA ÇIKTI VERME (Streamlit Arayüzü)
        # ==========================================
        st.success(f"Tahmin Başarılı! Hedef Tarih: **{hedef_tarih_str}**")

        # Görsel Özet Kutuları (Metrics)
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Önerilen Üretim Bloğu", value=f"{en_ucuz_baslangic:02d}:00 - {en_ucuz_bitis:02d}:59")
        col2.metric(label=f"{secilen_saat} Saatlik Toplam Maliyet", value=f"{toplam_maliyet:.2f} TL")
        col3.metric(label="Günlük Ortalama Fiyat", value=f"{df_tahmin['tahmin_ptf'].mean():.2f} TL")

        st.divider()

        # Grafiği Çizdirme
        st.subheader("📊 Fiyat Tahmini ve Fırsat Analizi Grafiği")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_tahmin['Hour'], df_tahmin['tahmin_ptf'], label='Yapay Zeka Tahmini', color='#FF4B4B', marker='o', linewidth=2)
        ax.scatter(ardisik_ucuzlar['Hour'], ardisik_ucuzlar['tahmin_ptf'], color='#00CC96', s=200, zorder=5, label=f'En Ucuz {secilen_saat} Saat')
        ax.axvspan(en_ucuz_baslangic, en_ucuz_bitis, color='#00CC96', alpha=0.2, label='Önerilen Çalışma Bloğu')
        
        ax.set_title(f'{hedef_tarih_str} - Elektrik Fiyat Tahmini', fontweight='bold')
        ax.set_xlabel('Saat (00:00 - 23:00)')
        ax.set_ylabel('Fiyat (TL/MWh)')
        ax.set_xticks(range(24))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        st.pyplot(fig)

        st.divider()

        # Tabloyu Gösterme
        st.subheader("📋 Detaylı Saatlik Veriler")
        gosterilecek_tablo = df_tahmin[['Hour', 'tahmin_ptf']].rename(columns={'Hour': 'Saat', 'tahmin_ptf': 'Tahmini PTF (TL/MWh)'})
        gosterilecek_tablo['Saat'] = gosterilecek_tablo['Saat'].apply(lambda x: f"{int(x):02d}:00")
        
        st.dataframe(gosterilecek_tablo, use_container_width=True)

else:
    st.info("👆 Sol menüden çalışma saatini seçip 'Tahmini Çalıştır' butonuna basınız.")
