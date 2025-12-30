import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETTINGS ---
n_clusters =3 
grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang']
file_prefix = "finn"  # Added prefix variable

# Loading data
df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("data_full_data_withoutArts.csv")
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]

df_fin = df[df['ST_SiteID'] == 6.0]
print("Number of finns in dataset", len(df_fin))
print("Inside finns Site", df_fin.groupby(['ST_CohortID','ST_Gender_Std']).size())

# Specific filter for Cohort 2.0 and Gender 2.0
df = df_fin[(df_fin['ST_CohortID'] == 2.0) & (df_fin['ST_Gender_Std'] == 2.0)]

# Column selection
columns_to_keep = [
    "ST_ASS_WLE_ADJ", "ST_COO_WLE_ADJ", "ST_CRE_WLE_ADJ", "ST_CUR_WLE_ADJ",
    "ST_EFF_WLE_ADJ", "ST_EMO_WLE_ADJ", "ST_EMP_WLE_ADJ", "ST_ENE_WLE_ADJ",
    "ST_MOT_WLE_ADJ", "ST_OPT_WLE_ADJ", "ST_PER_WLE_ADJ", "ST_RES_WLE_ADJ",
    "ST_SEL_WLE_ADJ", "ST_SOC_WLE_ADJ", "ST_STR_WLE_ADJ", "ST_TOL_WLE_ADJ",
    "ST_TRU_WLE_ADJ", "ST_st_relteach", "ST_st_bully", "ST_st_belong", 
    "ST_st_friends", "ST_st_relpar", "ST_st_globalmind", "ST_st_wellbeing", 
    "ST_st_anxtest", "ST_SES"
] + grade_cols

df = df[columns_to_keep]
df1 = df.dropna()

# Prepare data for clustering
X_df = df.drop(columns=grade_cols)
X_df = X_df.select_dtypes(include=[np.number]).dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- PCA LOADINGS HEATMAP ---
loadings_df = pd.DataFrame(
    pca.components_.T, 
    columns=['PC1', 'PC2'], 
    index=X_df.columns
)

plt.figure(figsize=(8, 14))
sns.heatmap(
    loadings_df, 
    annot=True, 
    cmap='coolwarm', 
    center=0, 
    fmt='.2f', 
    linewidths=0.5
)
plt.title('Wpływ zmiennych na składowe PCA (Loadings)')
plt.tight_layout()

# Updated filename with prefix
heatmap_filename = f"{file_prefix}_PCA_Loadings_{n_clusters}.jpg"
plt.savefig(heatmap_filename, format='jpg', dpi=300, bbox_inches='tight')
print(f"Heatmapa ładunków PCA została zapisana jako: {heatmap_filename}")
plt.show()

# Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_clustered = df.loc[X_df.index].copy()
df_clustered['Cluster'] = clusters
df_clustered['PCA1'] = X_pca[:, 0]
df_clustered['PCA2'] = X_pca[:, 1]

# --- CLUSTER CHARTS ---
plt.figure(figsize=(14, 6))

# Plot 1: Scatter plot
plt.subplot(1, 2, 1)
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_clustered, palette='viridis', s=60)
plt.title('Clusters based on Social-Emotional Skills and Socioeconomic')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot 2: Box plot
df_melted = df_clustered.melt(id_vars=['Cluster'], value_vars=grade_cols, var_name='Subject', value_name='Grade')
plt.subplot(1, 2, 2)
sns.boxplot(x='Subject', y='Grade', hue='Cluster', data=df_melted, palette='viridis')
plt.title('Academic Performance Patterns by Cluster')
plt.xticks(rotation=15)

plt.tight_layout()

# Updated filename with prefix
subjects_str = "_".join(grade_cols)
filename = f"{file_prefix}_Klastry_{n_clusters}_Przedmioty_{subjects_str}.jpg"

plt.savefig(filename, format='jpg', dpi=300)
print(f"Wykres został zapisany jako: {filename}")
plt.show()

print("Average Grades by Cluster:")
print(df_clustered.groupby('Cluster')[grade_cols].mean())