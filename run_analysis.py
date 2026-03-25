"""
Smart Customer Insights — Full Pipeline Script
Generates all outputs: cleaned CSVs, segmented CSVs, and all visualisation PNGs.
Run: python3 run_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings, os

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

PALETTE  = ['#2563EB', '#16A34A', '#DC2626', '#D97706', '#7C3AED', '#0891B2']
BG_COLOR = '#F8FAFC'
plt.rcParams.update({
    'figure.facecolor': BG_COLOR, 'axes.facecolor': BG_COLOR,
    'axes.spines.top': False,     'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 14, 'axes.labelsize': 12,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & INSPECT
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Loading & Inspecting Data")
print("=" * 60)
df_raw = pd.read_csv('customer_data.csv')
df     = df_raw.copy()
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")
missing = df.isnull().sum()
print(f"  Missing values:\n{missing[missing > 0].to_string()}")

num_cols = ['Age', 'PurchaseAmount', 'Frequency']
cat_cols = ['Gender', 'Region', 'ProductCategory']

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — Data Cleaning")
print("=" * 60)

# Convert numeric columns
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print("  ✓ Numeric columns converted")

# Remove invalid rows
before = len(df)
df = df[
    (df['Age'].isna()            | (df['Age'].between(10, 100)))  &
    (df['PurchaseAmount'].isna() | (df['PurchaseAmount'].between(0, 9999))) &
    (df['Frequency'].isna()      | (df['Frequency'] > 0))
]
print(f"  ✓ Removed {before - len(df)} invalid/test rows")

# Cap outliers (IQR × 3)
def cap_outliers(series, factor=3.0):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr    = q3 - q1
    return series.clip(lower=q1 - factor*iqr, upper=q3 + factor*iqr)

for col in num_cols:
    df[col] = cap_outliers(df[col])
print("  ✓ Outliers capped (IQR×3 Winsorisation)")

# Impute missing numerics with median
for col in num_cols:
    n = df[col].isna().sum()
    if n:
        df[col] = df[col].fillna(df[col].median())
        print(f"  ✓ Imputed {n} missing {col} values with median")

# Standardise categoricals
gender_map = {'male':'Male','m':'Male','man':'Male','female':'Female','f':'Female','woman':'Female'}
df['Gender'] = df['Gender'].str.strip().str.lower().map(gender_map).fillna('Unknown')
df['Region']          = df['Region'].str.strip().str.title()
df['ProductCategory'] = df['ProductCategory'].str.strip().str.title()
df['Name']            = df['Name'].fillna('Unknown')
print("  ✓ Categorical values standardised")

# Remove duplicates
before = len(df)
df = df.drop_duplicates(subset=['Name','Age','PurchaseAmount','Frequency'])
print(f"  ✓ Removed {before - len(df)} duplicate rows")
print(f"  Final clean dataset: {df.shape[0]:,} rows")

df.to_csv('output/customer_data_cleaned.csv', index=False)
print("  ✅ Saved → output/customer_data_cleaned.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 3. EDA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Exploratory Data Analysis")
print("=" * 60)

# 3.1 Numeric distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=BG_COLOR)
fig.suptitle('Distribution of Numeric Variables', fontsize=16, fontweight='bold', y=1.01)
titles = ['Customer Age', 'Purchase Amount ($)', 'Purchase Frequency']

for i, (col, title) in enumerate(zip(num_cols, titles)):
    ax_hist = axes[0, i]
    ax_hist.hist(df[col], bins=25, color=PALETTE[i], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_hist.axvline(df[col].mean(),   color='#111827', linestyle='--', linewidth=1.5, label=f'Mean: {df[col].mean():.1f}')
    ax_hist.axvline(df[col].median(), color='#EF4444', linestyle=':',  linewidth=1.5, label=f'Median: {df[col].median():.1f}')
    ax_hist.set_title(f'{title}\nHistogram', fontweight='bold')
    ax_hist.set_xlabel(col); ax_hist.set_ylabel('Count')
    ax_hist.legend(fontsize=9)

    ax_box = axes[1, i]
    ax_box.boxplot(df[col], vert=False, patch_artist=True, widths=0.5,
                   boxprops=dict(facecolor=PALETTE[i], alpha=0.6),
                   medianprops=dict(color='#111827', linewidth=2),
                   flierprops=dict(marker='o', markerfacecolor=PALETTE[i], markersize=4, alpha=0.5))
    ax_box.set_title(f'{title}\nBoxplot', fontweight='bold')
    ax_box.set_xlabel(col); ax_box.set_yticks([])

plt.tight_layout()
plt.savefig('output/01_numeric_distributions.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✓ Saved 01_numeric_distributions.png")

# 3.2 Scatterplots
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG_COLOR)
fig.suptitle('Scatterplots: Relationships Between Numeric Variables', fontsize=15, fontweight='bold')
pairs = [('Age','PurchaseAmount'), ('Age','Frequency'), ('Frequency','PurchaseAmount')]

for ax, (x, y), c in zip(axes, pairs, PALETTE):
    ax.scatter(df[x], df[y], alpha=0.35, s=25, color=c, edgecolors='none')
    m, b = np.polyfit(df[x], df[y], 1)
    xl   = np.linspace(df[x].min(), df[x].max(), 200)
    ax.plot(xl, m*xl + b, color='#111827', linewidth=1.5, linestyle='--')
    corr = df[x].corr(df[y])
    ax.set_title(f'{x} vs {y}\n(r = {corr:.3f})', fontweight='bold')
    ax.set_xlabel(x); ax.set_ylabel(y)

plt.tight_layout()
plt.savefig('output/02_scatterplots.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✓ Saved 02_scatterplots.png")

# 3.3 Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG_COLOR)
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.3f', cmap='Blues',
            vmin=-1, vmax=1, linewidths=0.5, cbar_kws={'shrink':0.8}, ax=ax)
ax.set_title('Correlation Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('output/03_correlation_heatmap.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✓ Saved 03_correlation_heatmap.png")

# 3.4 Categorical distributions
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG_COLOR)
fig.suptitle('Categorical Variable Distributions', fontsize=15, fontweight='bold')

for ax, col in zip(axes, cat_cols):
    counts = df[col].value_counts()
    pct    = counts / counts.sum() * 100
    bars   = ax.bar(counts.index, counts.values, color=PALETTE[:len(counts)], edgecolor='white', linewidth=0.8)
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{p:.1f}%', ha='center', va='bottom', fontsize=9, color='#374151')
    ax.set_title(col, fontweight='bold'); ax.set_xlabel(col); ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('output/04_categorical_distributions.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✓ Saved 04_categorical_distributions.png")

# 3.5 Avg spend by segment
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
fig.suptitle('Average Spend & Frequency by Segment', fontsize=14, fontweight='bold')

for ax, col in zip(axes, ['ProductCategory', 'Region']):
    avg  = df.groupby(col)['PurchaseAmount'].mean().sort_values(ascending=True)
    bars = ax.barh(avg.index, avg.values, color=PALETTE[:len(avg)], edgecolor='white')
    for bar in bars:
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                f'${bar.get_width():.0f}', va='center', fontsize=9)
    ax.set_title(f'Avg Purchase Amount by {col}', fontweight='bold')
    ax.set_xlabel('Avg Purchase Amount ($)')

plt.tight_layout()
plt.savefig('output/05_avg_spend_by_segment.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✓ Saved 05_avg_spend_by_segment.png")

# 3.6 Violin by gender
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
gender_vals  = [g for g in df['Gender'].unique() if g != 'Unknown']
gender_data  = [df[df['Gender'] == g]['PurchaseAmount'].values for g in gender_vals]
parts = ax.violinplot(gender_data, positions=range(len(gender_vals)), showmedians=True)
for i, (pc, c) in enumerate(zip(parts['bodies'], PALETTE)):
    pc.set_facecolor(c); pc.set_alpha(0.6)
parts['cmedians'].set_color('#111827'); parts['cmedians'].set_linewidth(2)
ax.set_xticks(range(len(gender_vals))); ax.set_xticklabels(gender_vals)
ax.set_title('Purchase Amount Distribution by Gender', fontweight='bold')
ax.set_ylabel('Purchase Amount ($)')
plt.tight_layout()
plt.savefig('output/06_spend_by_gender_violin.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✓ Saved 06_spend_by_gender_violin.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — Feature Engineering")
print("=" * 60)

df['CustomerValueScore'] = (
    0.6 * (df['PurchaseAmount'] / df['PurchaseAmount'].max()) +
    0.4 * (df['Frequency']      / df['Frequency'].max())
) * 100

bins   = [0, 25, 35, 45, 55, 100]
labels = ['<25', '25-34', '35-44', '45-54', '55+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
df = pd.get_dummies(df, columns=['Gender'], prefix='Gender', drop_first=False)
print("  ✓ CustomerValueScore, AgeGroup, Gender encoding added")

cluster_features = ['Age', 'PurchaseAmount', 'Frequency']
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[cluster_features])
print("  ✓ Features scaled with StandardScaler")

# ══════════════════════════════════════════════════════════════════════════════
# 5. CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — K-Means Clustering")
print("=" * 60)

# Elbow + Silhouette
inertias   = []
sil_scores = []
k_range    = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

best_k = list(k_range)[np.argmax(sil_scores)]
print(f"  Optimal k = {best_k} (silhouette = {max(sil_scores):.3f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
fig.suptitle('Optimal Number of Clusters', fontsize=15, fontweight='bold')
ax1.plot(list(k_range), inertias,   marker='o', color=PALETTE[0], linewidth=2)
ax1.set_title('Elbow Method (Inertia)', fontweight='bold')
ax1.set_xlabel('Number of Clusters (k)'); ax1.set_ylabel('Inertia')
ax1.set_xticks(list(k_range))
ax2.plot(list(k_range), sil_scores, marker='s', color=PALETTE[1], linewidth=2)
ax2.axvline(best_k, color='#EF4444', linestyle='--', linewidth=1.5, label=f'Best k={best_k}')
ax2.set_title('Silhouette Score', fontweight='bold')
ax2.set_xlabel('Number of Clusters (k)'); ax2.set_ylabel('Silhouette Score')
ax2.set_xticks(list(k_range)); ax2.legend()
plt.tight_layout()
plt.savefig('output/07_elbow_silhouette.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✓ Saved 07_elbow_silhouette.png")

# Final model
final_km     = KMeans(n_clusters=best_k, init='k-means++', n_init=50, random_state=42)
df['Cluster'] = final_km.fit_predict(X_scaled)

cluster_counts = df['Cluster'].value_counts().sort_index()
flat_summary   = df.groupby('Cluster')[cluster_features + ['CustomerValueScore']].mean().round(2)
flat_summary['Size']   = cluster_counts
flat_summary['Size_%'] = (cluster_counts / len(df) * 100).round(1)
flat_summary   = flat_summary.reset_index()

# Auto-assign labels by value score rank
value_rank = flat_summary.set_index('Cluster')['CustomerValueScore'].rank(ascending=False).astype(int)
label_pool = ['Champions','Loyal Customers','Potential Loyalists','At-Risk Customers','Low-Engagement']
cluster_labels = {cid: label_pool[r-1] if r <= len(label_pool) else f'Segment {r}'
                  for cid, r in value_rank.items()}

df['ClusterLabel']    = df['Cluster'].map(cluster_labels)
flat_summary['Label'] = flat_summary['Cluster'].map(cluster_labels)

label_order = [cluster_labels[k] for k in sorted(cluster_labels.keys())]
color_map   = {lbl: PALETTE[i] for i, lbl in enumerate(label_order)}

print("  Cluster → Label mapping:")
for k, v in cluster_labels.items():
    n = (df['Cluster'] == k).sum()
    print(f"    Cluster {k} → {v}  ({n} customers)")

# PCA
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]; df['PCA2'] = X_pca[:, 1]

# Cluster scatter
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_COLOR)
fig.suptitle('Customer Segmentation — K-Means Clusters', fontsize=15, fontweight='bold')

ax1 = axes[0]
for lbl, grp in df.groupby('ClusterLabel'):
    ax1.scatter(grp['PCA1'], grp['PCA2'], label=lbl, color=color_map[lbl], alpha=0.55, s=30, edgecolors='none')
centroids_pca = pca.transform(final_km.cluster_centers_)
ax1.scatter(centroids_pca[:,0], centroids_pca[:,1], marker='*', s=280, c='#111827', zorder=10, label='Centroids')
ax1.set_title(f'PCA Projection ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)', fontweight='bold')
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2'); ax1.legend(fontsize=8)

ax2 = axes[1]
for lbl, grp in df.groupby('ClusterLabel'):
    ax2.scatter(grp['Frequency'], grp['PurchaseAmount'], label=lbl, color=color_map[lbl], alpha=0.55, s=30, edgecolors='none')
ax2.set_title('Frequency vs Purchase Amount', fontweight='bold')
ax2.set_xlabel('Purchase Frequency'); ax2.set_ylabel('Purchase Amount ($)'); ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('output/08_cluster_scatter.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✓ Saved 08_cluster_scatter.png")

# Cluster profiles
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG_COLOR)
fig.suptitle('Cluster Profiles — Feature Means by Segment', fontsize=14, fontweight='bold')
metric_labels = {'Age':'Avg Age','PurchaseAmount':'Avg Spend ($)','Frequency':'Avg Frequency'}

for ax, col in zip(axes, cluster_features):
    cluster_means  = df.groupby('ClusterLabel')[col].mean().sort_values()
    colors_ordered = [color_map[l] for l in cluster_means.index]
    bars = ax.bar(cluster_means.index, cluster_means.values, color=colors_ordered, edgecolor='white', linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_title(metric_labels[col], fontweight='bold')
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('output/09_cluster_profiles.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✓ Saved 09_cluster_profiles.png")

# Doughnut
fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG_COLOR)
seg_counts = df['ClusterLabel'].value_counts()
wedges, texts, autotexts = ax.pie(
    seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%',
    colors=[color_map[l] for l in seg_counts.index],
    startangle=140, pctdistance=0.80,
    wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2)
)
for at in autotexts:
    at.set_fontsize(10); at.set_fontweight('bold')
ax.set_title('Customer Segment Size Distribution', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('output/10_segment_donut.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✓ Saved 10_segment_donut.png")

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10), facecolor='#1E293B')
fig.suptitle('📊  Smart Customer Insights — Executive Dashboard',
             fontsize=18, fontweight='bold', color='white', y=0.98)

kpis = [
    ('Total Customers', f'{len(df):,}'),
    ('Avg Spend',       f'${df["PurchaseAmount"].mean():.0f}'),
    ('Avg Frequency',   f'{df["Frequency"].mean():.1f}x'),
    ('Segments',        f'{best_k}'),
]
for i, (label, value) in enumerate(kpis):
    ax_kpi = fig.add_axes([0.02 + i*0.24, 0.82, 0.21, 0.12])
    ax_kpi.set_facecolor(PALETTE[i])
    ax_kpi.text(0.5, 0.65, value, transform=ax_kpi.transAxes,
                ha='center', va='center', fontsize=22, fontweight='bold', color='white')
    ax_kpi.text(0.5, 0.22, label, transform=ax_kpi.transAxes,
                ha='center', va='center', fontsize=10, color='white', alpha=0.85)
    ax_kpi.set_xticks([]); ax_kpi.set_yticks([])

ax_sc = fig.add_axes([0.02, 0.08, 0.44, 0.68])
ax_sc.set_facecolor('#0F172A')
for lbl, grp in df.groupby('ClusterLabel'):
    ax_sc.scatter(grp['Frequency'], grp['PurchaseAmount'],
                  label=lbl, color=color_map[lbl], alpha=0.55, s=20, edgecolors='none')
ax_sc.set_title('Frequency vs Spend by Segment', color='white', fontweight='bold')
ax_sc.set_xlabel('Frequency', color='#94A3B8'); ax_sc.set_ylabel('Purchase Amount ($)', color='#94A3B8')
ax_sc.tick_params(colors='#94A3B8')
for spine in ['bottom','left']: ax_sc.spines[spine].set_color('#334155')
for spine in ['top','right']:   ax_sc.spines[spine].set_visible(False)
ax_sc.legend(fontsize=8, facecolor='#1E293B', labelcolor='white', framealpha=0.8)

ax_bar = fig.add_axes([0.52, 0.08, 0.46, 0.68])
ax_bar.set_facecolor('#0F172A')
seg_mean_spend = df.groupby('ClusterLabel')['PurchaseAmount'].mean().sort_values(ascending=True)
bars = ax_bar.barh(seg_mean_spend.index, seg_mean_spend.values,
                   color=[color_map[l] for l in seg_mean_spend.index], edgecolor='none', height=0.55)
for bar in bars:
    ax_bar.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'${bar.get_width():.0f}', va='center', color='white', fontsize=10)
ax_bar.set_title('Avg Spend by Segment', color='white', fontweight='bold')
ax_bar.set_xlabel('Avg Purchase Amount ($)', color='#94A3B8')
ax_bar.tick_params(colors='white')
for spine in ['bottom','left']: ax_bar.spines[spine].set_color('#334155')
for spine in ['top','right']:   ax_bar.spines[spine].set_visible(False)

plt.savefig('output/00_executive_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#1E293B')
plt.close()
print("  ✓ Saved 00_executive_dashboard.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6 — Saving Final Outputs")
print("=" * 60)

output_cols = ['CustomerID','Name','Age','AgeGroup','Region','ProductCategory',
               'PurchaseAmount','Frequency','CustomerValueScore','Cluster','ClusterLabel']
output_cols = [c for c in output_cols if c in df.columns]

df[output_cols].to_csv('output/customer_data_segmented.csv', index=False)
flat_summary.to_csv('output/cluster_summary.csv', index=False)
print("  ✅ Saved → output/customer_data_segmented.csv")
print("  ✅ Saved → output/cluster_summary.csv")

print("\n" + "=" * 60)
print("SEGMENT INSIGHTS SUMMARY")
print("=" * 60)
insight_map = {
    'Champions':           '🏆  Highest value. Retain with VIP programs & referral rewards.',
    'Loyal Customers':     '⭐  Regular buyers. Upsell with bundles & premium tiers.',
    'Potential Loyalists': '🌱  Growing engagement. Nurture with personalised recommendations.',
    'At-Risk Customers':   '⚠️   Declining activity. Re-engage with win-back campaigns.',
    'Low-Engagement':      '💤  Occasional buyers. Activate with onboarding incentives.',
}
for lbl in label_order:
    grp = df[df['ClusterLabel'] == lbl]
    print(f"\n  {insight_map.get(lbl, lbl)}")
    print(f"  Size: {len(grp)} ({len(grp)/len(df)*100:.1f}%) | "
          f"Avg Age: {grp['Age'].mean():.1f} | "
          f"Avg Spend: ${grp['PurchaseAmount'].mean():.0f} | "
          f"Avg Freq: {grp['Frequency'].mean():.1f}")

print("\n\n✅ ALL DONE — check the output/ folder for all files!\n")
print("  Files generated:")
for f in sorted(os.listdir('output')):
    size = os.path.getsize(os.path.join('output', f))
    print(f"    {f:<45} {size/1024:.0f} KB")
