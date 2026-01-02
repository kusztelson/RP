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

# Assuming 'df' is your initial dataframe loaded with the columns you specified
# df = pd.read_csv("your_data.csv") # Uncomment and adjust if you need to load data

# 1. Drop NA globally first (or you can do it inside the loop if you prefer)
df_clean = df.dropna().copy()

# List of feature columns to be used for clustering/PCA
# (Excluding IDs and Weights)
features = [
    "ST_PER_WLE_ADJ", "ST_RES_WLE_ADJ", "ST_SEL_WLE_ADJ",
    "ST_st_relpar", "ST_st_relteach", "ST_st_friends",
    "ST_st_belong", "ST_st_bully", "ST_st_anxtest",
    "ST_SES", 'ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts'
]

# Dictionary to store results for each site
results = {}

# Get unique sites
unique_sites = df_clean['ST_SiteID'].unique()

for site in unique_sites:
    print(f"Processing Site: {site}")
    
    # 2. Filter data for the specific site
    site_data = df_clean[df_clean['ST_SiteID'] == site].copy()
    
    # 3. Rescale Weights
    # Calculate scaling factor: Sample Size / Sum of Weights
    n_sample = len(site_data)
    sum_weights = site_data['ST_WT2019'].sum()
    scaling_factor = n_sample / sum_weights
    
    # Create the rescaled weight column
    site_data['Rescaled_Weight'] = site_data['ST_WT2019'] * scaling_factor
    
    # Extract feature matrix (X) and weights (w)
    X = site_data[features].values
    w = site_data['Rescaled_Weight'].values
    
    # ---------------------------------------------------------
    # 4. Weighted PCA (Using the Square Root Weight Trick)
    # ---------------------------------------------------------
    
    # A. Calculate Weighted Mean
    # weighted_mean = sum(w * x) / sum(w)
    # Note: sum(w) is approximately n_sample due to rescaling, but best to be precise
    weighted_mean = np.average(X, axis=0, weights=w)
    
    # B. Center the data using the weighted mean
    X_centered = X - weighted_mean
    
    # C. Apply Square Root Weight
    # Shape of w is (n,), we need (n,1) for broadcasting
    w_sqrt = np.sqrt(w).reshape(-1, 1)
    X_weighted_centered = X_centered * w_sqrt
    
    # D. Run Standard PCA
    # We use the weighted-centered data here
    pca = PCA(n_components=0.95) # Keep 95% variance, or pick n_components=3 etc.
    X_pca_transformed = pca.fit_transform(X_weighted_centered)
    
    print(f"  > PCA Components selected: {pca.n_components_}")
    
    # ---------------------------------------------------------
    # 5. Weighted K-Means
    # ---------------------------------------------------------
    
    # IMPORTANT: For K-Means, we project the ORIGINAL centered data 
    # onto the PCA components, NOT the square-root weighted data.
    # If we use X_weighted_centered, we distort distances.
    
    # Transform the unweighted (but centered) data into the PCA space
    X_pca_for_kmeans = pca.transform(X_centered)
    
    # Run K-Means with sample_weights
    # (Adjust n_clusters as needed, e.g., via Elbow method in a separate step)
    k = 3 
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # .fit() allows sample_weight
    kmeans.fit(X_pca_for_kmeans, sample_weight=w)
    
    # 6. Store Results
    site_data['Cluster_Labels'] = kmeans.labels_
    
    # Add PCA coordinates to dataframe for plotting later
    for i in range(min(3, pca.n_components_)): # Store first 3 components
        site_data[f'PCA_{i+1}'] = X_pca_for_kmeans[:, i]

    results[site] = site_data

# 7. Combine all sites back into one dataframe if needed
final_df = pd.concat(results.values())

print("\nProcessing Complete.")
print(final_df[['ST_SiteID', 'Cluster_Labels', 'PCA_1', 'PCA_2']].head())
# %%
