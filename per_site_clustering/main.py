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
pca_models = {}
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

    scaler = StandardScaler()
    
    # Critical: Fit with sample_weight to get Weighted Mean and Weighted Std
    scaler.fit(X, sample_weight=w)
    
    # Transform: This makes Mean=0 and Std=1 for all columns
    X_std = scaler.transform(X)
    
    # 3. Apply Square Root Weight for PCA calculation only
    # We apply this to the STANDARDIZED data now
    w_sqrt = np.sqrt(w).reshape(-1, 1)
    X_weighted_std = X_std * w_sqrt   # <--- Changed X_centered to X_std
    
    # 4. Fit PCA
    pca = PCA(n_components=0.95) 
    pca.fit(X_weighted_std)           # <--- Changed input to X_weighted_std

    pca_models[site] = pca
    
    print(f"PCA Components: {pca.n_components_}")
    
    # ---------------------------------------------------------
    # C. Weighted K-Means
    # ---------------------------------------------------------
    
    # Project STANDARDIZED data (not weighted data) onto PCA components
    # We use X_std here because it represents the physical location in 'Standard Units'
    X_pca_for_kmeans = pca.transform(X_std)  # <--- Changed X_centered to X_std
    
    # Run Weighted K-Means
    k = 3 
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

    # Rename columns to avoid collision with original grades
    # This changes 'ST_Sgrade_Math' to 'Cluster_Mean_ST_Sgrade_Math'
    profile_renamed = profile.add_prefix('Cluster_Mean_')
    
    # Merge the profile data back into site_data
    # We map the cluster averages to each student based on their 'Cluster_Labels'
    site_data = site_data.merge(
        profile_renamed, 
        left_on='Cluster_Labels', 
        right_index=True, 
        how='left'
    )

    print(profile)
    print("\n")

    # Add PCA coordinates to dataframe for plotting later
    for i in range(min(2, pca.n_components_)): # Store first 3 components
        site_data[f'PCA_{i+1}'] = X_pca_for_kmeans[:, i]

    # Store result
    results[site] = site_data

# Combine
final_df = pd.concat(results.values())
# %%

# ---------------------------------------------------------
# Post-Processing: Reorder Clusters based on PCA_1
# ---------------------------------------------------------

def reorder_clusters(site_df):
    """
    Reorders cluster labels within a site such that:
    0 = Lowest average PCA_1
    1 = Middle average PCA_1
    2 = Highest average PCA_1
    """
    # 1. Calculate the mean PCA_1 for each existing cluster label
    cluster_stats = site_df.groupby('Cluster_Labels')['PCA_1'].mean()
    
    # 2. Sort the clusters by their mean PCA_1 value (Ascending)
    # The index will be the Old Label, the position will be the New Label (0, 1, 2)
    sorted_clusters = cluster_stats.sort_values().index
    
    # 3. Create a mapping dictionary: {Old_Label: New_Label}
    # enumerate gives us (0, Old_Label_1), (1, Old_Label_2)...
    # We flip it to create {Old_Label_1: 0, Old_Label_2: 1...}
    mapping = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(sorted_clusters)}
    
    # 4. Apply the mapping
    site_df['Ordered_Cluster'] = site_df['Cluster_Labels'].map(mapping)
    
    return site_df

# Apply the function to each site independently
final_df = final_df.groupby('ST_SiteID', group_keys=False).apply(reorder_clusters)

# Optional: Verify the new order
print("Checking Cluster Order (Expect ascending PCA_1 means):")
print(final_df.groupby('Ordered_Cluster')['PCA_1'].mean())

# 1. Setup Output Directory
output_folder = 'plots'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# 2. Define the pre-calculated profile columns
profile_cols = [
    'Cluster_Mean_ST_Sgrade_Math', 
    'Cluster_Mean_ST_Sgrade_Read_Lang', 
    'Cluster_Mean_ST_Sgrade_Arts'
]

# 3. Visualization Loop
unique_sites = final_df['ST_SiteID'].unique()

print(f"Generating plots for {len(unique_sites)} sites...")

for site in unique_sites:
    # Filter data for the specific site
    site_data = final_df[final_df['ST_SiteID'] == site]
    
    # Create the figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # --- Plot 1: PCA Cluster Map (Scatter) ---
    sns.scatterplot(
        data=site_data, 
        x='PCA_1', 
        y='PCA_2', 
        hue='Ordered_Cluster', 
        palette='viridis', 
        alpha=0.6, 
        s=50, 
        ax=axes[0]
    )
    axes[0].set_title(f'Student Clusters (Site: {site})')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend(title='Cluster')
    
    # --- Plot 2: Academic Profile (Bar Chart) ---
    # Efficiently extract the pre-calculated means
    # We drop duplicates because every student in the same cluster has the same mean value
    profile_plot_data = site_data[['Ordered_Cluster'] + profile_cols].drop_duplicates()
    
    # Set index for plotting and sort
    profile_plot_data = profile_plot_data.set_index('Ordered_Cluster').sort_index()
    
    # Clean up column names for the legend (Remove prefix)
    profile_plot_data.columns = [c.replace('Cluster_Mean_ST_Sgrade_', '') for c in profile_plot_data.columns]
    
    # Plot
    profile_plot_data.plot(kind='bar', ax=axes[1], colormap='Paired', edgecolor='black', zorder=3)
    
    axes[1].set_title(f'Academic Performance by Cluster (Site: {site})')
    axes[1].set_ylabel('Weighted Mean Grade')
    axes[1].set_xlabel('Cluster Group')
    axes[1].set_ylim(0, 50) # Adjust this range based on your data's scale
    axes[1].grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    axes[1].legend(title='Subject', loc='lower right')
    
    # --- Save and Close ---
    plt.tight_layout()
    
    # Save file using the site ID as the name
    filename = os.path.join(output_folder, f"{site}.png")
    plt.savefig(filename, dpi=100)
    plt.close(fig) # Close figure to free memory
    
    print(f"Saved: {filename}")

# %%
# Container for all site loadings
all_loadings = []

for site, pca_model in pca_models.items():
    # pca.components_ has shape [n_components, n_features]
    # We take the first row (PCA_1)
    loadings = pd.DataFrame(
        pca_model.components_, 
        columns=features, 
        index=[f'PCA_{i+1}' for i in range(pca_model.n_components_)]
    )
    
    # Extract just PCA_1 for analysis
    pca1_loadings = loadings.loc['PCA_1'].to_frame(name='Loading').reset_index()
    pca1_loadings['SiteID'] = site
    all_loadings.append(pca1_loadings)

# Combine into one dataframe
df_loadings = pd.concat(all_loadings)

# -------------------------------------------------------
# Visualization: What makes up PCA_1 across all sites?
# -------------------------------------------------------

plt.figure(figsize=(12, 6))

# We plot the average loading of each feature on PCA_1 across all sites
sns.barplot(
    data=df_loadings, 
    x='index', 
    y='Loading', 
    errorbar='sd', # Shows the standard deviation across sites
    palette='RdBu'
)

plt.title("Deconstructing PCA_1: Feature Contributions (Average across Sites)")
plt.xlabel("Feature")
plt.ylabel("Loading on PCA_1 (Correlation)")
plt.axhline(0, color='black', linewidth=1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Interpretation Helper
# -------------------------------------------------------
print("\n--- Interpretation of PCA_1 ---")
mean_loadings = df_loadings.groupby('index')['Loading'].mean().sort_values(ascending=False)
print(mean_loadings)