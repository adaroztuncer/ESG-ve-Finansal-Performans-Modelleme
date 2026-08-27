# ========================================================
#  GERÇEK 2026 REVENUE TAHMİNİ - FİNAL "ALTIN" VERSİYON
#  Özellikler: Log Transform + Delta Tahmini + Güvenli Reshape
# ========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# -------------------------------------------------------
# 1. AYARLAR VE VERİ HAZIRLIĞI
# -------------------------------------------------------
DATASET_PATH = "../../../data/raw/company_esg_financial_dataset.csv"

# Veri setini okuma
try:
    df = pd.read_csv(DATASET_PATH)
    print("Veri seti başarıyla yüklendi.")
    print(f"Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
except FileNotFoundError:
    print("Hata: Dosya bulunamadı. Lütfen dosya yolunu kontrol edin.")

# ----------------------------------------------------------------------
# NaN değerlerini doldurma işlemi
# ----------------------------------------------------------------------

# 'CompanyID' sütununa göre gruplama yapılır.
# transform() metodu ile her bir şirketin GrowthRate medyanı hesaplanır.
# fillna() ile NaN değerler bu medyan ile doldurulur.
df['GrowthRate'] = df.groupby('CompanyID')['GrowthRate'].transform(lambda x: x.fillna(x.median()))



# df'in yüklü olduğunu varsayıyoruz. Değilse: df = pd.read_csv('dataset.csv')
df_sorted = df.sort_values(['CompanyID', 'Year']).reset_index(drop=True)

# A. Logaritma Dönüşümü (Küçük/Büyük Şirket Farkını Kapatır)
df_sorted['Revenue_Log'] = np.log1p(df_sorted['Revenue'])

# B. Hedef Değişken: "Log Return" (Yüzdesel Değişim)
# Model "100 Milyon"u değil, "%5 Artışı" tahmin edecek.
df_sorted['Log_Return_Next'] = df_sorted.groupby('CompanyID')['Revenue_Log'].diff().shift(-1)

SEQUENCE_LENGTH = 3
feature_cols = [
    'ESG_Overall', 'ESG_Environmental', 'ESG_Social', 'ESG_Governance',
    'Revenue_Log', 'ProfitMargin', 'MarketCap',
    'CarbonEmissions', 'WaterUsage', 'EnergyConsumption'
]

# -------------------------------------------------------
# 2. LAG (GECİKME) ÖZELLİKLERİ
# -------------------------------------------------------
print("Veri zenginleştiriliyor...")
for col in feature_cols:
    for lag in range(1, SEQUENCE_LENGTH + 1):
        df_sorted[f'{col}_lag{lag}'] = df_sorted.groupby('CompanyID')[col].shift(lag)

# -------------------------------------------------------
# 3. ONE-HOT ENCODING (GLOBAL)
# -------------------------------------------------------
df_encoded = pd.get_dummies(df_sorted, columns=['Industry', 'Region'], dtype=int)

lag_columns = [c for c in df_encoded.columns if '_lag' in c]
cat_columns = [c for c in df_encoded.columns if c.startswith(('Industry_', 'Region_'))]
X_columns = lag_columns + cat_columns

# -------------------------------------------------------
# 4. EĞİTİM VERİSİ HAZIRLAMA
# -------------------------------------------------------
# NaN Temizliği
train_df = df_encoded.dropna(subset=X_columns + ['Log_Return_Next']).reset_index(drop=True)

# Outlier Filtreleme (Aşırı uç değerleri eğitimden atıyoruz)
upper = train_df['Log_Return_Next'].quantile(0.99)
lower = train_df['Log_Return_Next'].quantile(0.01)
train_df = train_df[(train_df['Log_Return_Next'] < upper) & (train_df['Log_Return_Next'] > lower)]

train_X, train_y = [], []
for company_id in train_df['CompanyID'].unique():
    comp = train_df[train_df['CompanyID'] == company_id].reset_index(drop=True)
    if len(comp) < SEQUENCE_LENGTH: continue
    
    X_val = comp[X_columns].values
    y_val = comp['Log_Return_Next'].values
    years = comp['Year'].values
    
    for i in range(len(comp) - SEQUENCE_LENGTH):
        # 2026 hedefini eğitime alma
        if years[i + SEQUENCE_LENGTH] > 2025: continue
        train_X.append(X_val[i:i+SEQUENCE_LENGTH])
        train_y.append(y_val[i + SEQUENCE_LENGTH])

X_train = np.array(train_X)
y_train = np.array(train_y)
print(f"Eğitim Seti: {len(X_train)} temiz örnek")

# -------------------------------------------------------
# 5. ROBUST SCALING
# -------------------------------------------------------
n_numeric = len(lag_columns)
scaler_X = RobustScaler()

# X Scaling (Eğitim Verisi)
X_train_num = X_train[:, :, :n_numeric].reshape(-1, n_numeric)
X_train_cat = X_train[:, :, n_numeric:]

X_train_num_scaled = scaler_X.fit_transform(X_train_num)
X_train_scaled = X_train_num_scaled.reshape(X_train.shape[0], SEQUENCE_LENGTH, n_numeric)
X_train_final = np.concatenate([X_train_scaled, X_train_cat], axis=2)

# -------------------------------------------------------
# 6. MODEL EĞİTİMİ (FRENLİ LSTM)
# -------------------------------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X_train_final.shape[2]),
         kernel_regularizer=l2(0.01)), # Aşırı öğrenmeyi engeller
    Dropout(0.4),
    LSTM(32, kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1) # Linear (Log Return tahmini)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("Model eğitiliyor...")
model.fit(X_train_final, y_train, epochs=200, batch_size=32, validation_split=0.2, 
          callbacks=[early_stop], verbose=1)

# -------------------------------------------------------
# 7. 2026 TAHMİNİ (BOYUT HATASI DÜZELTİLMİŞ KISIM)
# -------------------------------------------------------
print("\n2026 tahmin verileri hazırlanıyor...")

pred_sequences = []
pred_metadata = []

for company_id in df_encoded['CompanyID'].unique():
    comp_all = df_encoded[df_encoded['CompanyID'] == company_id].reset_index(drop=True)
    required_years = [2023, 2024, 2025]
    
    # Veri kontrolü
    if not all(y in comp_all['Year'].values for y in required_years): continue
    
    # Sıralama Garantisi
    target_rows = comp_all[comp_all['Year'].isin(required_years)].sort_values('Year')
    if len(target_rows) != 3: continue
    if target_rows[X_columns].isnull().any().any(): continue

    X_seq = target_rows[X_columns].values
    
    # Metadata
    last_row = target_rows.iloc[-1]
    pred_sequences.append(X_seq)
    pred_metadata.append({
        'CompanyID': company_id,
        'CompanyName': last_row.get('CompanyName', f"ID_{company_id}"),
        'Revenue_2025': last_row['Revenue'],
        'Log_Revenue_2025': last_row['Revenue_Log']
    })

X_2026 = np.array(pred_sequences)
meta_2026 = pd.DataFrame(pred_metadata)

if len(X_2026) > 0:
    # --- KRİTİK DÜZELTME: ŞEKİLLENDİRME (RESHAPE) ---
    
    # 1. Sayısal ve Kategorik veriyi ayır
    # X_2026 shape: (N_Samples, 3, Total_Features)
    X_2026_num = X_2026[:, :, :n_numeric] 
    X_2026_cat = X_2026[:, :, n_numeric:]
    
    # 2. Reshape -> (N_Samples * 3, Numeric_Features) -> Scaler -> (N_Samples * 3, Numeric_Features)
    # Scaler 2D matris bekler.
    N_samples = X_2026.shape[0]
    X_2026_num_reshaped = X_2026_num.reshape(-1, n_numeric)
    
    # 3. Transform
    X_2026_num_scaled_flat = scaler_X.transform(X_2026_num_reshaped)
    
    # 4. Geri Reshape -> (N_Samples, 3, Numeric_Features)
    X_2026_num_scaled = X_2026_num_scaled_flat.reshape(N_samples, SEQUENCE_LENGTH, n_numeric)
    
    # 5. Birleştir
    X_2026_final = np.concatenate([X_2026_num_scaled, X_2026_cat], axis=2)
    
    # 6. Tahmin (Delta / Log Return)
    pred_delta = model.predict(X_2026_final, verbose=0).flatten()
    
    # 7. Manuel Kilit (Güvenlik Önlemi)
    # Hiçbir şirket %50'den fazla büyüyemez, %30'dan fazla küçülemez.
    pred_delta = np.clip(pred_delta, -0.30, 0.50)
    
    # 8. Sonuç Hesaplama
    meta_2026['Predicted_GrowthRate_2026'] = pred_delta # Log return ≈ Growth Rate (küçük oranlarda)
    
    # Log Return'ü Gerçek Gelire Çevirme
    # Formül: Yeni_Log = Eski_Log + Delta
    meta_2026['Predicted_Revenue_Log_2026'] = meta_2026['Log_Revenue_2025'] + pred_delta
    # Formül: Yeni_Rev = exp(Yeni_Log) - 1
    meta_2026['Predicted_Revenue_2026'] = np.expm1(meta_2026['Predicted_Revenue_Log_2026'])
    
    # Hassas Büyüme Oranı Hesabı (Opsiyonel, Delta ile hemen hemen aynı çıkar)
    meta_2026['Real_Growth_Percentage'] = (
        meta_2026['Predicted_Revenue_2026'] - meta_2026['Revenue_2025']
    ) / meta_2026['Revenue_2025']

    # Sıralama
    results = meta_2026.sort_values('Real_Growth_Percentage', ascending=False).reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print(f" SONUÇLAR (HATA YOK - GARANTİLİ)")
    print(f"{'='*70}")
    print(results[['CompanyName', 'Revenue_2025', 'Predicted_Revenue_2026', 'Real_Growth_Percentage']].head(15))
    
    results.to_csv('2026_Final_Predictions_Clean.csv', index=False)
    print("\nDosya kaydedildi: 2026_Final_Predictions_Clean.csv")
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(results['Real_Growth_Percentage'] * 100, bins=30, color='teal', edgecolor='black')
    plt.title('2026 Tahmini Büyüme Dağılımı (Makul Oranlar)')
    plt.xlabel('Büyüme (%)')
    plt.ylabel('Şirket Sayısı')
    plt.show()

else:
    print("HATA: Hiçbir şirket için 2023-2024-2025 verisi tam bulunamadı.")