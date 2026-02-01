import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETTINGS ---
n_clusters = 3
grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang','ST_Sgrade_Arts']
weight_col = 'PA_WT2019_PA'
file_prefix = "Suzhou_WEIGHTED_PARENTS_V2"

# --- LOADING DATA ---
df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("data_full_data_withoutArts.csv")
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]

# Filtering for Site 3.0
df_fin = df[df['ST_SiteID'] == 11.0]
print(f"Number of students in Site 3.0: {len(df_fin)}")

# Specific filter for Cohort 2.0 and Gender 2.0
df = df_fin[(df_fin['ST_CohortID'] == 2.0) & (df_fin['ST_Gender_Std'] == 2.0)]
df = df.reset_index(drop=True)

# Column selection & Dropna
columns_to_keep = [
    "ST_RES_WLE_ADJ", "ST_SEL_WLE_ADJ", "ST_PER_WLE_ADJ", "ST_st_relteach", "ST_st_bully", 
    "ST_st_belong", "ST_st_friends", "ST_st_relpar", "ST_st_wellbeing", 
    "ST_st_anxtest", "ST_SES" ,
    "PA_pa_comm", "PA_pa_wellbeing", "PA_pa_engage", "PA_pa_encour"
] + grade_cols + [weight_col]

df = df[columns_to_keep].dropna().reset_index(drop=True)
print(f"Number of students after clearing dataset: {len(df)}")

# --- PREPARATION ---
X_df = df.drop(columns=grade_cols + [weight_col]) 
weights = df[weight_col].values 
weights = weights / weights.mean()

# --- CUSTOM WEIGHTED PCA FUNCTION ---
def run_weighted_pca(X, weights, n_components=2):
    weighted_mean = np.average(X, axis=0, weights=weights)
    weighted_var = np.average((X - weighted_mean)**2, axis=0, weights=weights)
    weighted_std = np.sqrt(weighted_var)
    
    X_std = (X - weighted_mean) / weighted_std
    cov_matrix = np.cov(X_std.T, aweights=weights)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    top_eigenvectors = eigenvectors[:, :n_components]
    X_pca = np.dot(X_std, top_eigenvectors)
    
    loadings_df = pd.DataFrame(
        top_eigenvectors, 
        columns=[f'PC{i+1}' for i in range(n_components)], 
        index=X.columns
    )
    return X_pca, loadings_df

print("\nRunning Weighted PCA...")
X_pca_data, loadings_df = run_weighted_pca(X_df, weights, n_components=2)

# --- WEIGHTED K-MEANS ---
print("\nRunning Weighted K-Means...")
weighted_mean = np.average(X_df, axis=0, weights=weights)
weighted_std = np.sqrt(np.average((X_df - weighted_mean)**2, axis=0, weights=weights))
X_scaled_weighted = (X_df - weighted_mean) / weighted_std

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
raw_clusters = kmeans.fit_predict(X_scaled_weighted, sample_weight=weights)

# --- REORDER CLUSTERS BY PCA1 ---
# Calculate weighted average of PCA1 for each raw cluster
cluster_pca1_means = []
for i in range(n_clusters):
    mask = (raw_clusters == i)
    w_avg_pca1 = np.average(X_pca_data[mask, 0], weights=weights[mask])
    cluster_pca1_means.append((i, w_avg_pca1))

# Sort clusters based on PCA1 mean (ascending)
sorted_clusters = sorted(cluster_pca1_means, key=lambda x: x[1])
mapping = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(sorted_clusters)}

# Apply the mapping to the cluster labels
final_clusters = np.array([mapping[c] for c in raw_clusters])

# Add to DataFrame
df_clustered = df.copy()
df_clustered['Cluster'] = final_clusters
df_clustered['PCA1'] = X_pca_data[:, 0]
df_clustered['PCA2'] = X_pca_data[:, 1]

# --- PLOT 1: PCA LOADINGS HEATMAP ---
plt.figure(figsize=(8, 14))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
plt.title('PCA Loadings - Suzhou')
plt.tight_layout()
plt.savefig(f"{file_prefix}_PCA_Loadings_{n_clusters}.jpg", dpi=300)
print(f"Graph saved: {file_prefix}_PCA_Loadings_{n_clusters}.jpg")
plt.show()

# --- PLOT 2: SCATTER PLOT (CLUSTERS) ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_clustered, palette='viridis', s=60, alpha=0.8)
plt.title('KMeans Clusters')
plt.tight_layout()
scatter_filename = f"{file_prefix}_Clusters_Scatter_{n_clusters}.jpg"
plt.savefig(scatter_filename, format='jpg', dpi=300)
print(f"Graph saved: {scatter_filename}")
plt.show()

# --- PLOT 3: BOX PLOT (GRADES) ---
plt.figure(figsize=(8, 6))
# Resampling for weighted visualization
df_resampled = df_clustered.sample(n=len(df_clustered), replace=True, weights=weights, random_state=42)
df_melted_weighted = df_resampled.melt(id_vars=['Cluster'], value_vars=grade_cols, var_name='Subject', value_name='Grade')

sns.boxplot(x='Subject', y='Grade', hue='Cluster', data=df_melted_weighted, palette='viridis', showfliers=False)
plt.title('Weighted Grade Distribution by Ordered Cluster')
plt.xticks(rotation=15)
plt.tight_layout()
grades_filename = f"{file_prefix}_Grades_Boxplot_{n_clusters}.jpg"
plt.savefig(grades_filename, format='jpg', dpi=300)
print(f"Graph saved: {grades_filename}")
plt.show()

# --- WEIGHTED STATISTICS ---
# --- WEIGHTED STATISTICS ---
print("\nWeighted Average Grades by Cluster Suzhou")
for cluster_id in range(n_clusters):
    subset = df_clustered[df_clustered['Cluster'] == cluster_id]
    subset_weights = weights[df_clustered['Cluster'] == cluster_id]
    
    if len(subset) > 0:
        w_avg_math = np.average(subset['ST_Sgrade_Math'], weights=subset_weights)
        w_avg_read = np.average(subset['ST_Sgrade_Read_Lang'], weights=subset_weights)
        w_avg_art = np.average(subset['ST_Sgrade_Arts'], weights=subset_weights)
        print(f"Cluster {cluster_id}: Math={w_avg_math:.2f}, Read={w_avg_read:.2f}, Art={w_avg_art:.2f}, N={len(subset)}")
        
        
loadings_melted = loadings_df.reset_index().melt(id_vars='index', var_name='PC', value_name='Loading')
loadings_melted.rename(columns={'index': 'Feature'}, inplace=True)

# 2. Plot PCA 1 Contributions
plt.figure(figsize=(10, 6))
pca1_data = loadings_melted[loadings_melted['PC'] == 'PC1'].sort_values(by='Loading', ascending=False)
sns.barplot(
    data=pca1_data, 
    x='Feature', 
    y='Loading', 
    palette='RdBu_r'
)
plt.title(f"Deconstructing PCA 1: Feature Contributions (Suzhou)")
plt.xlabel("Feature")
plt.ylabel("Loading (Correlation with PC1)")
plt.axhline(0, color='black', linewidth=1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{file_prefix}_PCA1_BarChart.jpg", dpi=300)
plt.show()

# 3. Plot PCA 2 Contributions
plt.figure(figsize=(10, 6))
pca2_data = loadings_melted[loadings_melted['PC'] == 'PC2'].sort_values(by='Loading', ascending=False)
sns.barplot(
    data=pca2_data, 
    x='Feature', 
    y='Loading', 
    palette='BrBG'
)
plt.title(f"Deconstructing PCA 2: Feature Contributions (Suzhou)")
plt.xlabel("Feature")
plt.ylabel("Loading (Correlation with PC2)")
plt.axhline(0, color='black', linewidth=1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{file_prefix}_PCA2_BarChart.jpg", dpi=300)
plt.show()