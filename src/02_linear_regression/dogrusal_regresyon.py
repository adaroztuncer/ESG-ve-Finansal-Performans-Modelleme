import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Veri Yükleme ve Ön İşleme
df = pd.read_csv("../../data/raw/company_esg_financial_dataset.csv")
df_clean = df.drop(columns=['GrowthRate']) # Eksik veri içeren sütunu çıkarıyoruz

# Kategorik dönüşüm
df_encoded = pd.get_dummies(df_clean, columns=['Industry', 'Region'], drop_first=True)

# 2. Değişkenlerin Seçimi
target = 'Revenue'
features = df_encoded.drop(columns=[
    target, 'CompanyID', 'CompanyName', 
    'ESG_Overall', 'CarbonEmissions', 'WaterUsage', 'EnergyConsumption'
])

X = features
y = df_encoded[target]

# 3. Eğitim ve Test Bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modelleme
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Performans Metrikleri ve Tutarlılık Kontrolü
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# MAPE hesaplama (Sıfıra bölüm hatasını önlemek için basit kontrol eklenebilir, burada numpy ile doğrudan hesaplandı)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Çapraz Doğrulama (Cross-Validation) - Tutarlılık için en önemli adım
# Veriyi 5 parçaya bölüp her seferinde farklı bir parçayı test ederek ortalama başarıyı ölçer.
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print("Model Performans Değerleri:")
print(f"R-Kare (Test Seti): {r2:.4f}")
print(f"RMSE (Kök Ortalama Kare Hata): {rmse:.2f}")
print(f"MAE (Ortalama Mutlak Hata): {mae:.2f}")
print(f"MAPE (Ortalama Yüzde Hata): {mape:.2f}%")
print("-" * 30)
print("Tutarlılık Analizi (Cross-Validation):")
print(f"5 Katlı Çapraz Doğrulama Skorları: {cv_scores}")
print(f"Ortalama R-Kare: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 6. Hata Analizi Görselleştirmesi (Residual Analysis)
plt.figure(figsize=(14, 6))

# Tahmin vs Hata Grafiği (Homoscedasticity)
plt.subplot(1, 2, 1)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Tahmin Edilen Gelir')
plt.ylabel('Hata (Residuals)')
plt.title('Tahmin vs Hata Dağılımı (Tutarlılık Kontrolü)')

# Hata Dağılımı Histogramı
plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True, color='green')
plt.title('Hata Terimlerinin Dağılımı (Normallik Kontrolü)')
plt.xlabel('Hata Miktarı')

plt.tight_layout()
plt.show()

# 7. 2026 Tahminlerinin Oluşturulması
# En son yıl verisini (2025) alıp yılı 2026 olarak güncelleyerek tahmin yapıyoruz
df_2025 = df_encoded[df_encoded['Year'] == 2025].copy()
X_2026 = df_2025.drop(columns=[
    target, 'CompanyID', 'CompanyName', 
    'ESG_Overall', 'CarbonEmissions', 'WaterUsage', 'EnergyConsumption'
])
X_2026['Year'] = 2026

predictions_2026 = model.predict(X_2026)

results_df = pd.DataFrame({
    'CompanyID': df_2025['CompanyID'],
    'CompanyName': df_2025['CompanyName'],
    'Predicted_Revenue_2026': predictions_2026
})

# Sonuçları kaydet
results_df.to_csv('predicted_revenue_2026_evaluated.csv', index=False)
print("\n2026 Tahminleri 'predicted_revenue_2026_evaluated.csv' dosyasına kaydedildi.")