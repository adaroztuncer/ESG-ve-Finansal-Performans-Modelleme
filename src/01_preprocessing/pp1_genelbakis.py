import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

DATASET_PATH = "../../data/raw/company_esg_financial_dataset.csv"

# Uyarıları daha temiz bir görüntü için filtreleyelim
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Veri setini okuma
try:
    df = pd.read_csv(DATASET_PATH)
    print("Veri seti başarıyla yüklendi.")
    print(f"Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
except FileNotFoundError:
    print("Hata: Dosya bulunamadı. Lütfen dosya yolunu kontrol edin.")

# İlk 5 satıra hızlı bir bakış
df.head()


print("--- Veri Seti Genel Bilgileri ---")
print("Shape:", df.shape)
print("\nInfo:")
df.info()

print("\nDescribe:")
df.describe(include="all")
