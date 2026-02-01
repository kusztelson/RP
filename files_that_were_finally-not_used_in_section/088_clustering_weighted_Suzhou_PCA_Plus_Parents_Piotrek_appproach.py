import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# --- SETTINGS ---
n_clusters = 3
grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts']
weight_col = 'PA_WT2019_PA'
file_prefix = "Suzhou_WEIGHTED_PARENTS_V2"

# --- 1. DATA LOADING & FILTERING ---
df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("data_full_data_withoutArts.csv")
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]

# Filter for Site 11.0, Cohort 2.0, and Gender 2.0
df_sub = df[
    (df['ST_SiteID'] == 11.0) & 
    (df['ST_CohortID'] == 2.0) & 
    (df['ST_Gender_Std'] == 2.0)
].copy()

features = [
    "ST_RES_WLE_ADJ", "ST_SEL_WLE_ADJ", "ST_PER_WLE_ADJ", "ST_st_relteach", 
    "ST_st_bully", "ST_st_belong", "ST_st_friends", "ST_st_relpar", 
    "ST_st_wellbeing", "ST_st_anxtest", "ST_SES" ,
    "PA_pa_comm", "PA_pa_wellbeing", "PA_pa_engage", "PA_pa_encour"
]

# Feature label mapping for explainable names
feature_labels = {
    "ST_RES_WLE_ADJ": "Responsibility",
    "ST_SEL_WLE_ADJ": "Self-control",
    "ST_PER_WLE_ADJ": "Persistence",
    "ST_st_relteach": "Teacher Relations",
    "ST_st_bully": "Bullying",
    "ST_st_belong": "School Belonging",
    "ST_st_friends": "Friend Relations",
    "ST_st_relpar": "Parent Relations",
    "ST_st_wellbeing": "Wellbeing",
    "ST_st_anxtest": "School Anxiety",
    "ST_SES": "Socioeconomic Status",
    "PA_pa_comm": "Parental Community Closeness",
    "PA_pa_wellbeing": "Parental Wellbeing",
    "PA_pa_engage": "Parental Engagement",
    "PA_pa_encour": "Parental Need for Encouragement"
}

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
    index=[feature_labels[f] for f in features]
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
plt.title('PCA Loadings - Suzhou')
plt.tight_layout()
plt.savefig(f"{file_prefix}_PCA_Loadings_{n_clusters}.jpg", dpi=300)
plt.savefig(f"{file_prefix}_PCA_Loadings_{n_clusters}.pdf")
plt.show()

# --- PLOT 2: SCATTER PLOT (CLUSTERS) ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_clean, palette='viridis', s=60, alpha=0.8)
plt.title('KMeans Clusters')
plt.tight_layout()
plt.savefig(f"{file_prefix}_Clusters_Scatter_{n_clusters}.jpg", dpi=300)
plt.savefig(f"{file_prefix}_Clusters_Scatter_{n_clusters}.pdf")
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
plt.savefig(f"{file_prefix}_Grades_Boxplot_{n_clusters}.pdf")
plt.show()

# --- PLOT 4 & 5: FEATURE CONTRIBUTIONS (BAR CHARTS) ---
loadings_melted = loadings_df.reset_index().melt(id_vars='index', var_name='PC', value_name='Loading')
loadings_melted.rename(columns={'index': 'Feature'}, inplace=True)

# PC1 Contributions
plt.figure(figsize=(10, 6))
pca1_data = loadings_melted[loadings_melted['PC'] == 'PC1'].sort_values(by='Loading', ascending=False)
sns.barplot(data=pca1_data, x='Feature', y='Loading', palette='RdBu_r')
plt.title(f"Deconstructing PCA 1: Feature Contributions (Suzhou)")
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig(f"{file_prefix}_PCA1_BarChart.jpg", dpi=300)
plt.savefig(f"{file_prefix}_PCA1_BarChart.pdf")
plt.show()

# PC2 Contributions
plt.figure(figsize=(10, 6))
pca2_data = loadings_melted[loadings_melted['PC'] == 'PC2'].sort_values(by='Loading', ascending=False)
sns.barplot(data=pca2_data, x='Feature', y='Loading', palette='BrBG')
plt.title(f"Deconstructing PCA 2: Feature Contributions (Suzhou)")
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig(f"{file_prefix}_PCA2_BarChart.jpg", dpi=300)
plt.savefig(f"{file_prefix}_PCA2_BarChart.pdf")
plt.show()

# --- FINAL STATS PRINT ---
print("\nWeighted Average Grades by Cluster Suzhou")
for cluster_id in range(n_clusters):
    subset = df_clean[df_clean['Cluster'] == cluster_id]
    if len(subset) > 0:
        w_avg_math = np.average(subset['ST_Sgrade_Math'], weights=subset['Rescaled_Weight'])
        w_avg_read = np.average(subset['ST_Sgrade_Read_Lang'], weights=subset['Rescaled_Weight'])
        w_avg_art = np.average(subset['ST_Sgrade_Arts'], weights=subset['Rescaled_Weight'])
        print(f"Cluster {cluster_id}: Math={w_avg_math:.2f}, Read={w_avg_read:.2f}, Art={w_avg_art:.2f}, N={len(subset)}")
        
        
        
# 1. Calculate the weighted means and store in a DataFrame
cluster_stats = []
for cluster_id in range(n_clusters):
    subset = df_clean[df_clean['Cluster'] == cluster_id]
    if len(subset) > 0:
        row = {'Cluster': cluster_id}
        for col in grade_cols:
            row[col] = np.average(subset[col], weights=subset['Rescaled_Weight'])
        cluster_stats.append(row)

profile_plot_data = pd.DataFrame(cluster_stats).set_index('Cluster').sort_index()

# 2. Clean up column names for the legend (Remove prefix)
# This mimics your c.replace logic to keep the legend tidy
profile_plot_data.columns = [c.replace('ST_Sgrade_', '') for c in profile_plot_data.columns]

# 3. Create the Plot using the requested layout
fig, ax = plt.subplots(figsize=(8, 6))

profile_plot_data.plot(
    kind='bar', 
    ax=ax, 
    colormap='Paired', 
    edgecolor='black', 
    zorder=3
)

# 4. Styling based on your provided layout
ax.set_title(f'Academic Performance by Cluster (Suzhou)')
ax.set_ylabel('Weighted Mean Grade')
ax.set_xlabel('Cluster Group')

# Note: Adjust ylim (e.g., 0, 50 or 0, 5) based on your specific grade scale
ax.set_ylim(0, profile_plot_data.values.max() * 1.2) 

ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.legend(title='Subject', loc='upper right')

# 5. Save and Close
plt.tight_layout()
filename = f"{file_prefix}_Final_Cluster_Performance.png"
plt.savefig(filename, dpi=300)
plt.savefig(f"{file_prefix}_Final_Cluster_Performance.pdf")
plt.show()

print(f"Final chart saved as: {filename}")






# 1. Calculate the weighted means and store in a DataFrame
cluster_stats = []
for cluster_id in range(n_clusters):
    subset = df_clean[df_clean['Cluster'] == cluster_id]
    if len(subset) > 0:
        row = {'Cluster': cluster_id}
        for col in grade_cols:
            row[col] = np.average(subset[col], weights=subset['Rescaled_Weight'])
        cluster_stats.append(row)

profile_plot_data = pd.DataFrame(cluster_stats).set_index('Cluster').sort_index()

# 2. Clean up column names for the legend (Remove prefix)
# This mimics your c.replace logic to keep the legend tidy
profile_plot_data.columns = [c.replace('ST_Sgrade_', '') for c in profile_plot_data.columns]

# 3. Create the Plot using the requested layout
fig, ax = plt.subplots(figsize=(8, 6))

profile_plot_data.plot(
    kind='bar', 
    ax=ax, 
    colormap='Paired', 
    edgecolor='black', 
    zorder=3
)

# 4. Styling based on your provided layout
ax.set_title(f'Academic Performance by Cluster (Suzhou)')
ax.set_ylabel('Weighted Mean Grade')
ax.set_xlabel('Cluster Group')

# Note: Adjust ylim (e.g., 0, 50 or 0, 5) based on your specific grade scale
ax.set_ylim(25, profile_plot_data.values.max() * 1.2) 

ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.legend(title='Subject', loc='upper right')

# 5. Save and Close
plt.tight_layout()
filename = f"{file_prefix}_Final_Cluster_Performance_different_scale.png"
plt.savefig(filename, dpi=300)
plt.savefig(f"{file_prefix}_Final_Cluster_Performance_different_scale.pdf")
plt.show()

print(f"Final chart saved as: {filename}")