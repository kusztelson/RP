import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETTINGS ---
n_clusters = 3
grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts']
weight_col = 'PA_WT2019_PA'
file_prefix = "Bogota_WEIGHTED_PARENTS_V2"

# --- 1. DATA LOADING & FILTERING ---
df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("data_full_data_withoutArts.csv")
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]

# Filter for Site 11.0, Cohort 2.0, and Gender 2.0
df_sub = df[
    (df['ST_SiteID'] == 3.0) & 
    (df['ST_CohortID'] == 2.0) & 
    (df['ST_Gender_Std'] == 2.0)
].copy()

features = [
    "ST_RES_WLE_ADJ", "ST_SEL_WLE_ADJ", "ST_PER_WLE_ADJ", "ST_st_relteach", 
    "ST_st_bully", "ST_st_belong", "ST_st_friends", "ST_st_relpar", 
    "ST_st_wellbeing", "ST_st_anxtest", "ST_SES" ,
    "PA_pa_comm", "PA_pa_wellbeing", "PA_pa_engage", "PA_pa_encour"
]

df_clean = df_sub[features + grade_cols + [weight_col]].dropna().reset_index(drop=True)
print(f"Number of students after clearing dataset: {len(df_clean)}")

# --- 2. PREPARATION & WEIGHT RESCALING ---
n_sample = len(df_clean)
sum_weights = df_clean[weight_col].sum()
# Rescaling weight so sum(weights) == N
df_clean['Rescaled_Weight'] = df_clean[weight_col] * (n_sample / sum_weights)

weights = df_clean['Rescaled_Weight'].values
X_raw = df_clean[features].values

# --- 3. STANDARDIZATION & PCA ---
scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)

# Apply Square Root Weighting for PCA calculation
w_sqrt = np.sqrt(weights).reshape(-1, 1)
X_weighted_std = X_std * w_sqrt 

pca = PCA(n_components=2)
X_pca_data = pca.fit_transform(X_weighted_std)

# Store loadings for the Heatmap
loadings_df = pd.DataFrame(
    pca.components_.T, 
    columns=['PC1', 'PC2'], 
    index=features
)

# --- 4. WEIGHTED K-MEANS ---
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
raw_clusters = kmeans.fit_predict(X_pca_data, sample_weight=weights)

# --- 5. REORDER CLUSTERS BY PCA1 ---
cluster_pca1_means = []
for i in range(n_clusters):
    mask = (raw_clusters == i)
    w_avg_pca1 = np.average(X_pca_data[mask, 0], weights=weights[mask])
    cluster_pca1_means.append((i, w_avg_pca1))

sorted_clusters = sorted(cluster_pca1_means, key=lambda x: x[1])
mapping = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(sorted_clusters)}
final_clusters = np.array([mapping[c] for c in raw_clusters])

df_clean['Cluster'] = final_clusters
df_clean['PCA1'] = X_pca_data[:, 0]
df_clean['PCA2'] = X_pca_data[:, 1]

# --- PLOT 1: PCA LOADINGS HEATMAP ---
plt.figure(figsize=(8, 14))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
plt.title('PCA Loadings - Bogota')
plt.tight_layout()
plt.savefig(f"{file_prefix}_PCA_Loadings_{n_clusters}.jpg", dpi=300)
plt.show()

# --- PLOT 2: SCATTER PLOT (CLUSTERS) ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_clean, palette='viridis', s=60, alpha=0.8)
plt.title('KMeans Clusters')
plt.tight_layout()
plt.savefig(f"{file_prefix}_Clusters_Scatter_{n_clusters}.jpg", dpi=300)
plt.show()

# --- PLOT 3: BOX PLOT (GRADES) ---
plt.figure(figsize=(8, 6))
# Resampling using the rescaled weights for visualization
df_resampled = df_clean.sample(n=len(df_clean), replace=True, weights='Rescaled_Weight', random_state=42)
df_melted_weighted = df_resampled.melt(id_vars=['Cluster'], value_vars=grade_cols, var_name='Subject', value_name='Grade')

sns.boxplot(x='Subject', y='Grade', hue='Cluster', data=df_melted_weighted, palette='viridis', showfliers=False)
plt.title('Weighted Grade Distribution by Ordered Cluster')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{file_prefix}_Grades_Boxplot_{n_clusters}.jpg", dpi=300)
plt.show()

# --- PLOT 4 & 5: FEATURE CONTRIBUTIONS (BAR CHARTS) ---
loadings_melted = loadings_df.reset_index().melt(id_vars='index', var_name='PC', value_name='Loading')
loadings_melted.rename(columns={'index': 'Feature'}, inplace=True)

# PC1 Contributions
plt.figure(figsize=(10, 6))
pca1_data = loadings_melted[loadings_melted['PC'] == 'PC1'].sort_values(by='Loading', ascending=False)
sns.barplot(data=pca1_data, x='Feature', y='Loading', palette='RdBu_r')
plt.title(f"Deconstructing PCA 1: Feature Contributions (Bogota)")
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig(f"{file_prefix}_PCA1_BarChart.jpg", dpi=300)
plt.show()

# PC2 Contributions
plt.figure(figsize=(10, 6))
pca2_data = loadings_melted[loadings_melted['PC'] == 'PC2'].sort_values(by='Loading', ascending=False)
sns.barplot(data=pca2_data, x='Feature', y='Loading', palette='BrBG')
plt.title(f"Deconstructing PCA 2: Feature Contributions (Bogota)")
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig(f"{file_prefix}_PCA2_BarChart.jpg", dpi=300)
plt.show()

# --- FINAL STATS PRINT ---
print("\nWeighted Average Grades by Cluster Bogota")
for cluster_id in range(n_clusters):
    subset = df_clean[df_clean['Cluster'] == cluster_id]
    if len(subset) > 0:
        w_avg_math = np.average(subset['ST_Sgrade_Math'], weights=subset['Rescaled_Weight'])
        w_avg_read = np.average(subset['ST_Sgrade_Read_Lang'], weights=subset['Rescaled_Weight'])
        w_avg_art = np.average(subset['ST_Sgrade_Arts'], weights=subset['Rescaled_Weight'])
        print(f"Cluster {cluster_id}: Math={w_avg_math:.2f}, Read={w_avg_read:.2f}, Art={w_avg_art:.2f}, N={len(subset)}")