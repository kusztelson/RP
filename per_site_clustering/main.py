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
    # B. Stratified Standardization (The Modification)
    # ---------------------------------------------------------
    
    # 1. Create new columns for Z-scores in the dataframe
    z_features = [f"Z_{col}" for col in features]
    # Initialize with NaN
    for z_col in z_features:
        site_data[z_col] = np.nan
    
    # 2. Iterate through each subgroup (Cohort x Gender)
    groups = site_data.groupby(['ST_CohortID', 'ST_Gender_Std'])
    
    for (cohort, gender), group_indices in groups.groups.items():
        # Boolean mask for this group
        mask = (site_data['ST_CohortID'] == cohort) & (site_data['ST_Gender_Std'] == gender)
        
        # Check if group has data (safety check)
        if mask.sum() > 1:
            # Extract raw data and weights for this group
            X_sub = site_data.loc[mask, features].values
            w_sub = site_data.loc[mask, 'Rescaled_Weight'].values
            
            # Fit Scaler specific to this Age/Gender group
            scaler = StandardScaler()
            scaler.fit(X_sub, sample_weight=w_sub)
            
            # Transform and save DIRECTLY to the dataframe
            site_data.loc[mask, z_features] = scaler.transform(X_sub)
        else:
            # Handle rare edge case of groups with 0 or 1 student
            site_data.loc[mask, z_features] = 0 # Fallback to mean (0)

    # 3. Create the X_std matrix from the dataframe for PCA use
    # We fill NaNs with 0 just in case a group was skipped, though unlikely
    X_std = site_data[z_features].fillna(0).values

    # ---------------------------------------------------------
    # Resume existing flow
    # ---------------------------------------------------------
    
    # 3. Apply Square Root Weight for PCA calculation only
    # We apply this to the now STRATIFIED standardized data
    w_sqrt = np.sqrt(w).reshape(-1, 1)
    X_weighted_std = X_std * w_sqrt 
    
    # 4. Fit PCA
    pca = PCA(n_components=2) 
    pca.fit(X_weighted_std)

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

# %%
# Container for PCA_2 loadings
all_loadings_pca2 = []

for site, pca_model in pca_models.items():
    # Create a DataFrame for the components
    loadings = pd.DataFrame(
        pca_model.components_, 
        columns=features, 
        index=[f'PCA_{i+1}' for i in range(pca_model.n_components_)]
    )
    
    # Check if PCA_2 exists for this site
    if 'PCA_2' in loadings.index:
        # Extract just PCA_2
        pca2_loadings = loadings.loc['PCA_2'].to_frame(name='Loading').reset_index()
        pca2_loadings['SiteID'] = site
        all_loadings_pca2.append(pca2_loadings)

# Combine into one dataframe
df_loadings_pca2 = pd.concat(all_loadings_pca2)

# -------------------------------------------------------
# Visualization: What makes up PCA_2?
# -------------------------------------------------------

plt.figure(figsize=(12, 6))

# Plotting loadings for PCA_2
sns.barplot(
    data=df_loadings_pca2, 
    x='index', 
    y='Loading', 
    errorbar='sd', # Standard deviation shows how consistent this pattern is across sites
    palette='BrBG' # Using a different palette (Brown-Green) to distinguish from PCA 1
)

plt.title("Deconstructing PCA_2: Feature Contributions (Average across Sites)")
plt.xlabel("Feature")
plt.ylabel("Loading on PCA_2")
plt.axhline(0, color='black', linewidth=1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Assuming 'final_df' is your result from the previous step
unique_sites = final_df['ST_SiteID'].unique()

for site in unique_sites:
    print(f"\n{'='*40}")
    print(f"DEMOGRAPHIC CHECK FOR SITE: {site}")
    print(f"{'='*40}")
    
    site_data = final_df[final_df['ST_SiteID'] == site]
    
    # 1. Check Cohort Balance
    # normalize='index' shows the percentage within each cluster
    cohort_tab = pd.crosstab(
        site_data['Cluster_Labels'], 
        site_data['ST_CohortID'], 
        normalize='index'
    ) * 100
    
    print("\n--- Cohort Distribution (Percentage per Cluster) ---")
    print(cohort_tab.round(1).astype(str) + '%')
    
    # 2. Check Gender Balance
    gender_tab = pd.crosstab(
        site_data['Cluster_Labels'], 
        site_data['ST_Gender_Std'], 
        normalize='index'
    ) * 100
    
    print("\n--- Gender Distribution (Percentage per Cluster) ---")
    print(gender_tab.round(1).astype(str) + '%')
    
    # 3. Check Raw Counts (to ensure no tiny clusters)
    counts = site_data['Cluster_Labels'].value_counts().sort_index()
    print("\n--- Total Students per Cluster ---")
    print(counts)
# %%
import numpy as np
import pandas as pd

# The grades we are checking
grade_columns = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts']
unique_sites = final_df['ST_SiteID'].unique()

print(f"{'Site':<10} | {'Math Rising?':<12} | {'Read Rising?':<12} | {'Arts Rising?':<12} | {'ALL RISING?'}")
print("-" * 70)

for site in unique_sites:
    site_data = final_df[final_df['ST_SiteID'] == site]
    
    # 1. Calculate Weighted Means for each Cluster
    # We group by label (0, 1, 2) and compute the weighted avg for each grade
    means = site_data.groupby('Ordered_Cluster')[grade_columns].apply(
        lambda x: pd.Series({
            col: np.average(x[col], weights=site_data.loc[x.index, 'Rescaled_Weight']) 
            for col in grade_columns
        })
    )
    
    # 2. Check the "0 < 1 < 2" condition for each subject
    # We use .get() to handle cases where a cluster might be missing (rare)
    try:
        math_check = means.loc[0, 'ST_Sgrade_Math'] < means.loc[1, 'ST_Sgrade_Math'] < means.loc[2, 'ST_Sgrade_Math']
        read_check = means.loc[0, 'ST_Sgrade_Read_Lang'] < means.loc[1, 'ST_Sgrade_Read_Lang'] < means.loc[2, 'ST_Sgrade_Read_Lang']
        arts_check = means.loc[0, 'ST_Sgrade_Arts'] < means.loc[1, 'ST_Sgrade_Arts'] < means.loc[2, 'ST_Sgrade_Arts']
        
        all_check = math_check and read_check and arts_check
        
        print(f"{site:<10.0f} | {str(math_check):<12} | {str(read_check):<12} | {str(arts_check):<12} | {str(all_check)}")
        
    except KeyError:
        print(f"{site:<10.0f} | {'ERROR: Missing Cluster (Empty Group)':<40}")
# %%
# ---------------------------------------------------------
# 1. Setup Columns and Labels
# ---------------------------------------------------------
# The list of Z-score columns we created in the loop
z_features = [f"Z_{col}" for col in features]

# Mapping for the legend (Adjust if your clusters 0/1/2 mean something else)
# Viridis colors: 0=Purple (Dark), 1=Teal, 2=Yellow (Bright)
cluster_names = {
    0: '0: Vulnerable / Developing',
    1: '1: Moderate / Average',
    2: '2: High-Mastery / Thriving'
}

# ---------------------------------------------------------
# 2. Calculate Weighted Means per Cluster
# ---------------------------------------------------------
cluster_profile_data = []

# We group by the Ordered_Cluster to ensure the hierarchy (0 -> 1 -> 2)
sorted_clusters = sorted(final_df['Ordered_Cluster'].unique())

for cluster in sorted_clusters:
    # Filter for this cluster
    mask = final_df['Ordered_Cluster'] == cluster
    
    # Check if cluster exists (safety)
    if mask.sum() == 0: continue

    # Dictionary to store this cluster's averages
    cluster_stats = {'Ordered_Cluster': cluster}
    
    # Calculate weighted mean for each feature
    current_weights = final_df.loc[mask, 'Rescaled_Weight']
    
    for raw_feat, z_feat in zip(features, z_features):
        # We grab the Z-score column
        current_values = final_df.loc[mask, z_feat]
        
        # Compute Weighted Average
        w_avg = np.average(current_values, weights=current_weights)
        cluster_stats[raw_feat] = w_avg 
        
    cluster_profile_data.append(cluster_stats)

# Create summary dataframe
df_profile = pd.DataFrame(cluster_profile_data)

# ---------------------------------------------------------
# 3. Reshape for Seaborn ("Long" Format)
# ---------------------------------------------------------
df_long = df_profile.melt(
    id_vars='Ordered_Cluster', 
    var_name='Skill', 
    value_name='Mean_Z_Score'
)

# Add readable names
df_long['Cluster_Label'] = df_long['Ordered_Cluster'].map(cluster_names)

# ---------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------
plt.figure(figsize=(14, 8))

sns.pointplot(
    data=df_long, 
    x='Skill', 
    y='Mean_Z_Score', 
    hue='Cluster_Label',
    palette='viridis',      # Matches your scatterplot
    markers=['o', 's', '^'], # Circle, Square, Triangle for accessibility
    scale=1.0,               # Size of markers
    linestyles=['-', '--', '-.'] # Solid, Dashed, Dash-dot
)

# Visual polish
plt.axhline(0, color='black', linewidth=1.5, linestyle=':', alpha=0.6, label='Global Average (0)')
plt.title('Standardized Profiles of Student Clusters (Weighted Means)', fontsize=16)
plt.ylabel('Score Relative to Peer Group (Z-Score)', fontsize=12)
plt.xlabel('Social & Emotional Skill Domain', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Student Typology', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
# %%
