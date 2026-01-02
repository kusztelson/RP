# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("../data_full_data.csv")
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]

df = df[
    [
    "ST_CohortID",
    "ST_SiteID",
    "ST_Gender_Std",
    "ST_WT2019",
    "ST_PER_WLE_ADJ",
    "ST_RES_WLE_ADJ",
    "ST_SEL_WLE_ADJ",
    "ST_st_relpar",
    "ST_st_relteach",
    "ST_st_friends",
    "ST_st_belong",
    "ST_st_bully",
    "ST_st_anxtest",
    "ST_SES",
    'ST_Sgrade_Math',
    'ST_Sgrade_Read_Lang',
    'ST_Sgrade_Arts']]

# 1. Clean Data
# We drop NA for the whole dataset to ensure rows align.
# Note: Ensure the grade columns are not causing excessive row drops if they have many NAs.
df_clean = df.dropna().copy()

# 2. Define Features for Clustering
# Grades REMOVED from this list, but kept in df_clean for later analysis.
features = [
    "ST_PER_WLE_ADJ", "ST_RES_WLE_ADJ", "ST_SEL_WLE_ADJ",
    "ST_st_relpar", "ST_st_relteach", "ST_st_friends",
    "ST_st_belong", "ST_st_bully", "ST_st_anxtest",
    "ST_SES"
]

# Grades list for profiling later
grade_columns = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts']

results = {}
unique_sites = df_clean['ST_SiteID'].unique()

for site in unique_sites:
    print(f"--- Processing Site: {site} ---")
    
    # Filter data for site
    site_data = df_clean[df_clean['ST_SiteID'] == site].copy()
    
    # ---------------------------------------------------------
    # A. Rescale Weights (Sum of weights = Sample Size)
    # ---------------------------------------------------------
    n_sample = len(site_data)
    sum_weights = site_data['ST_WT2019'].sum()
    scaling_factor = n_sample / sum_weights
    site_data['Rescaled_Weight'] = site_data['ST_WT2019'] * scaling_factor
    
    # Prepare matrices
    X = site_data[features].values
    w = site_data['Rescaled_Weight'].values
    
    # ---------------------------------------------------------
    # B. Weighted PCA (Square Root Weight Method)
    # ---------------------------------------------------------
    
    # 1. Weighted Mean
    weighted_mean = np.average(X, axis=0, weights=w)
    
    # 2. Center Data
    X_centered = X - weighted_mean
    
    # 3. Apply Square Root Weight for PCA calculation only
    w_sqrt = np.sqrt(w).reshape(-1, 1)
    X_weighted_centered = X_centered * w_sqrt
    
    # 4. Fit PCA
    pca = PCA(n_components=0.95) 
    pca.fit(X_weighted_centered)
    
    print(f"PCA Components: {pca.n_components_}")
    
    # ---------------------------------------------------------
    # C. Weighted K-Means
    # ---------------------------------------------------------
    
    # Project CENTERED data (not weighted data) onto PCA components
    X_pca_for_kmeans = pca.transform(X_centered)
    
    # Run Weighted K-Means
    k = 3 # You can change this or loop to find optimal k
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca_for_kmeans, sample_weight=w)
    
    # Assign labels
    site_data['Cluster_Labels'] = kmeans.labels_
    
    # ---------------------------------------------------------
    # D. Cluster Profiling (Check Grades)
    # ---------------------------------------------------------
    print(f"Cluster Profile for Site {site}:")
    
    # Helper function to calculate weighted mean per group
    def weighted_avg(x):
        # Match weights to the current group's indices
        w_group = site_data.loc[x.index, 'Rescaled_Weight']
        return np.average(x, weights=w_group)

    # Group by Cluster and calculate weighted mean for Grades
    # We use apply(weighted_avg) to ensure the grades are weighted correctly
    profile = site_data.groupby('Cluster_Labels')[grade_columns].apply(
        lambda x: pd.Series({col: weighted_avg(x[col]) for col in grade_columns})
    )
    print(profile)
    print("\n")

    # Store result
    results[site] = site_data

# Combine
final_df = pd.concat(results.values())
# %%
