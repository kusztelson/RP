import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETTINGS ---
n_clusters = 4
grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang']
max_pca_to_test = 15
features = [
    "ST_ASS_WLE_ADJ", "ST_COO_WLE_ADJ", "ST_CRE_WLE_ADJ", "ST_CUR_WLE_ADJ",
    "ST_EFF_WLE_ADJ", "ST_EMO_WLE_ADJ", "ST_EMP_WLE_ADJ", "ST_ENE_WLE_ADJ",
    "ST_MOT_WLE_ADJ", "ST_OPT_WLE_ADJ", "ST_PER_WLE_ADJ", "ST_RES_WLE_ADJ",
    "ST_SEL_WLE_ADJ", "ST_SOC_WLE_ADJ", "ST_STR_WLE_ADJ", "ST_TOL_WLE_ADJ",
    "ST_TRU_WLE_ADJ", "ST_st_relteach", "ST_st_bully", "ST_st_belong", 
    "ST_st_friends", "ST_st_relpar", "ST_st_globalmind", "ST_st_wellbeing",
    "ST_st_anxtest", "ST_SES"
]

# --- DATA LOADING & PREPROCESSING ---
# (Using your provided logic)
df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("data_full_data_withoutArts.csv")
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]

df_clean = df[features + grade_cols].dropna()
X_df = df_clean[features]
y_df = df_clean[grade_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# --- PCA OPTIMIZATION LOOP ---
pca_metrics = []
ranked_trends = []

print("Starting PCA optimization and performance ranking...")

for n in range(1, max_pca_to_test + 1):
    # 1. PCA & Clustering
    pca_test = PCA(n_components=n)
    X_pca_test = pca_test.fit_transform(X_scaled)
    
    kmeans_test = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(X_pca_test)
    
    # 2. Geometric Metrics
    sil = silhouette_score(X_pca_test, labels)
    explained_var = np.sum(pca_test.explained_variance_ratio_)
    
    # 3. Grade Performance Analysis
    temp_df = y_df.copy()
    temp_df['Cluster'] = labels
    cluster_means = temp_df.groupby('Cluster')[grade_cols].mean()
    
    # Calculate Separation (How different are the clusters' averages?)
    math_sep = cluster_means['ST_Sgrade_Math'].std()
    read_sep = cluster_means['ST_Sgrade_Read_Lang'].std()
    
    # Ranking clusters (0 = Lowest, 4 = Highest)
    math_rank_map = cluster_means['ST_Sgrade_Math'].sort_values().rank(method='first').astype(int) - 1
    read_rank_map = cluster_means['ST_Sgrade_Read_Lang'].sort_values().rank(method='first').astype(int) - 1
    
    # Average intra-cluster standard deviation (Consistency)
    intra_std = np.mean([temp_df[temp_df['Cluster'] == c][grade_cols].std().mean() for c in range(n_clusters)])

    # Store general metrics
    pca_metrics.append({
        'n_components': n,
        'silhouette': sil,
        'explained_var': explained_var,
        'math_separation': math_sep,
        'read_separation': read_sep,
        'avg_intra_cluster_std': intra_std
    })
    
    # Store ranked data for line plots
    for c in range(n_clusters):
        ranked_trends.append({
            'n_pca': n,
            'math_mean': cluster_means.loc[c, 'ST_Sgrade_Math'],
            'math_rank': math_rank_map.loc[c],
            'read_mean': cluster_means.loc[c, 'ST_Sgrade_Read_Lang'],
            'read_rank': read_rank_map.loc[c]
        })

# Prepare DataFrames
res_df = pd.DataFrame(pca_metrics)
trends_df = pd.DataFrame(ranked_trends)

# --- VISUALIZATION ---

# 1. GEOMETRIC QUALITY AND SEPARATION
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Number of PCA Components')
ax1.set_ylabel('Silhouette Score', color='tab:blue')
ax1.plot(res_df['n_components'], res_df['silhouette'], marker='o', color='tab:blue', linewidth=2, label='Silhouette')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Grade Separation (Std of Means)', color='tab:red')
ax2.plot(res_df['n_components'], res_df['math_separation'], marker='s', color='tab:red', linestyle='--', label='Math Separation')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('PCA Optimization: Structure Quality vs Grade Differentiation')
plt.grid(True, alpha=0.3)
plt.savefig("PCA_Clustering_Metrics.jpg", dpi=300)
plt.show()

# 2. RANKED GRADE TRENDS (Math & Reading)
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 14))
performance_palette = sns.color_palette("RdYlGn", n_colors=n_clusters)
rank_labels = ['Lowest Performance', 'Below Average', 'Average', 'Above Average', 'Highest Performance']

# Math Trends
sns.lineplot(data=trends_df, x='n_pca', y='math_mean', hue='math_rank', 
             palette=performance_palette, marker='o', ax=ax3, linewidth=2.5)
ax3.set_title('Cluster Stability: Mathematics Mean Grades', fontsize=14)
ax3.set_ylabel('Mean Grade')
ax3.legend(title='Performance Rank', labels=rank_labels, bbox_to_anchor=(1.02, 1), loc='upper left')
ax3.grid(True, alpha=0.2)

# Reading Trends
sns.lineplot(data=trends_df, x='n_pca', y='read_mean', hue='read_rank', 
             palette=performance_palette, marker='s', ax=ax4, linewidth=2.5)
ax4.set_title('Cluster Stability: Reading Mean Grades', fontsize=14)
ax4.set_ylabel('Mean Grade')
ax4.set_xlabel('Number of PCA Components')
ax4.legend(title='Performance Rank', labels=rank_labels, bbox_to_anchor=(1.02, 1), loc='upper left')
ax4.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("PCA_Ranked_Grade_Trends.jpg", dpi=300)
plt.show()

print("\nBest PCA configurations by Grade Separation (Math):")
print(res_df.sort_values('math_separation', ascending=False)[['n_components', 'silhouette', 'math_separation']].head(5))