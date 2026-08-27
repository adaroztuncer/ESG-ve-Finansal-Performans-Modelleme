# ========================================================
#  SEKTÖR BAZLI ANALİZLER 
# ========================================================

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# -------------------------------------------------------
# 1. SEKTÖR ORTALAMALARI
# -------------------------------------------------------
print("=== SEKTÖR BAZINDA MEDYAN DEĞERLER ===")

agg_cols = [
    'Revenue', 'ProfitMargin', 'MarketCap', 'GrowthRate',
    'ESG_Overall', 'ESG_Environmental', 'ESG_Social', 'ESG_Governance',
    'CarbonEmissions', 'WaterUsage', 'EnergyConsumption'
]

sector_summary = df.groupby('Industry')[agg_cols].median().round(2)
sector_summary = sector_summary.sort_values('GrowthRate', ascending=False)

print(sector_summary)
sector_summary.to_csv('sector_summary_medians.csv')
print("\n→ 'sector_summary_medians.csv' kaydedildi.")

# Görsel: GrowthRate vs ESG
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(data=sector_summary.reset_index(), x='GrowthRate', y='Industry', ax=ax1, palette='viridis')
ax1.set_title('Sektörlere Göre Medyan GrowthRate (%)', fontweight='bold')
ax1.set_xlabel('Medyan Büyüme Oranı (%)')

sns.barplot(data=sector_summary.reset_index(), x='ESG_Overall', y='Industry', ax=ax2, palette='mako')
ax2.set_title('Sektörlere Göre Medyan ESG Skoru', fontweight='bold')
ax2.set_xlabel('ESG Overall (0-100)')

plt.tight_layout()
plt.savefig('sector_growth_vs_esg.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------
# 2. SEKTÖR BAZLI GROWTHRATE KORELASYONLARI 
# -------------------------------------------------------
print("\n" + "="*70)
print("SEKTÖR BAZINDA GROWTHRATE ↔ DİĞER DEĞİŞKENLER KORELASYONU")
print("="*70)

correlation_by_sector = {}

for industry in df['Industry'].unique():
    subset = df[df['Industry'] == industry]
    
    if len(subset) < 20:  # çok küçük sektörleri atla
        continue
        
    corr = subset[agg_cols].corr()['GrowthRate'].drop('GrowthRate', errors='ignore')
    corr_sorted = corr.sort_values(ascending=False)
    correlation_by_sector[industry] = corr_sorted
    
    print(f"\n{industry.upper()} ({len(subset)} gözlem)")
    print("-" * 50)
    print(corr_sorted.round(3).to_string())

# Genel korelasyonları hesapla ve sırala
all_corrs = df[agg_cols].corr()['GrowthRate'].drop('GrowthRate', errors='ignore')
sorted_growth_corrs = all_corrs.sort_values(ascending=False)
# ------------------------------------------

# DataFrame'e çevir
corr_df = pd.DataFrame(correlation_by_sector).T

# Genel korelasyona göre sütunları sırala 
corr_df = corr_df.reindex(columns=sorted_growth_corrs.index)

# NaN → 0 yap
corr_df = corr_df.fillna(0)

print("\n→ Tüm sektör korelasyon tablosu:")
display(corr_df.round(3))

corr_df.to_csv('growthrate_correlations_by_sector.csv')
print("\n→ 'growthrate_correlations_by_sector.csv' kaydedildi.")

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
            cbar_kws={'label': 'Korelasyon (r)'}, linewidths=0.5)

plt.title('Sektör Bazında GrowthRate ile Korelasyonlar', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Değişkenler')
plt.ylabel('Sektör')
plt.tight_layout()
plt.savefig('growthrate_correlations_by_sector_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTüm analizler tamamlandı! Dosyalar ve grafikler kaydedildi.")