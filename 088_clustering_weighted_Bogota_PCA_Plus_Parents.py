import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETTINGS ---
n_clusters = 3
grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang']
weight_col = 'PA_WT2019_PA'
file_prefix = "Bogota_WEIGHTED_PARENTS_V2" # Nowa wersja pliku



# --- LOADING DATA ---
# (Tutaj wklej swoją część wczytywania danych, zakładam, że masz już zmienną df)
# Dla pewności upewnij się, że df ma zresetowany index, żeby uniknąć problemów przy łączeniu

# Loading data
df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("data_full_data_withoutArts.csv")
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]
print(df['ST_SiteID'].unique)
df_fin = df[df['ST_SiteID'] == 3.0]
print("Number of finns in dataset", len(df_fin))
print("Inside finns Site", df_fin.groupby(['ST_CohortID','ST_Gender_Std']).size())

# Specific filter for Cohort 2.0 and Gender 2.0
df = df_fin[(df_fin['ST_CohortID'] == 2.0) & (df_fin['ST_Gender_Std'] == 2.0)]



df = df.reset_index(drop=True)


# Column selection & Dropna
columns_to_keep = [
    "ST_RES_WLE_ADJ", "ST_SEL_WLE_ADJ", "ST_PER_WLE_ADJ","ST_st_relteach", "ST_st_bully", 
    "ST_st_belong", "ST_st_friends", "ST_st_relpar", "ST_st_wellbeing", 
    "ST_st_anxtest", "ST_SES" ,
    "PA_pa_comm","PA_pa_wellbeing" , "PA_pa_engage","PA_pa_encour"
    
] + grade_cols + [weight_col]

df = df[columns_to_keep].dropna().reset_index(drop=True)
print("Number of students after clearing dataset",len(df))
# --- PREPARATION ---
X_df = df.drop(columns=grade_cols + [weight_col]) 
weights = df[weight_col].values # Pobieramy jako numpy array

# Normalizacja wag (dla bezpieczeństwa obliczeń numerycznych warto, by sumowały się do N lub 1)
weights = weights / weights.mean()

# --- CUSTOM WEIGHTED PCA FUNCTION ---
def run_weighted_pca(X, weights, n_components=2):
    """
    Oblicza PCA uwzględniając wagi obserwacji (studentów).
    1. Ważona standaryzacja.
    2. Ważona macierz kowariancji.
    3. Rozkład na wartości własne.
    """
    # 1. Obliczamy ważoną średnią dla każdej kolumny
    weighted_mean = np.average(X, axis=0, weights=weights)
    
    # 2. Obliczamy ważoną wariancję (i odchylenie standardowe)
    # Wariancja = średnia ważona z kwadratów odchyleń od średniej
    weighted_var = np.average((X - weighted_mean)**2, axis=0, weights=weights)
    weighted_std = np.sqrt(weighted_var)
    
    # 3. Standaryzujemy dane (Z-score) używając ważonych parametrów
    # To kluczowe: zwykły StandardScaler tutaj by przekłamał
    X_std = (X - weighted_mean) / weighted_std
    
    # 4. Obliczamy Macierz Kowariancji (na danych standaryzowanych = Macierz Korelacji)
    # aweights w numpy.cov służy właśnie do wag obserwacji (reliability weights)
    cov_matrix = np.cov(X_std.T, aweights=weights)
    
    # 5. Rozkład na wartości własne (Eigendecomposition)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sortujemy wyniki malejąco (bo linalg.eigh zwraca rosnąco)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # Wybieramy top N komponentów
    top_eigenvectors = eigenvectors[:, :n_components]
    
    # 6. Transformacja danych (rzutowanie na nowe osie)
    X_pca = np.dot(X_std, top_eigenvectors)
    
    # Tworzymy DataFrame z ładunkami (Loadings) dla łatwego rysowania
    loadings_df = pd.DataFrame(
        top_eigenvectors, 
        columns=[f'PC{i+1}' for i in range(n_components)], 
        index=X.columns
    )
    
    return X_pca, loadings_df

print("\nLiczenie Ważonego PCA (metoda NumPy)...")
X_pca_data, loadings_df = run_weighted_pca(X_df, weights, n_components=2)

# --- PCA LOADINGS HEATMAP ---
plt.figure(figsize=(8, 14))
sns.heatmap(
    loadings_df, 
    annot=True, 
    cmap='coolwarm', 
    center=0, 
    fmt='.2f', 
    linewidths=0.5
)
plt.title(f'Ładunki czynnikowe Ważonego PCA (N={len(df)})')
plt.tight_layout()

heatmap_filename = f"{file_prefix}_PCA_Loadings_{n_clusters}.jpg"
plt.savefig(heatmap_filename, format='jpg', dpi=300, bbox_inches='tight')
print(f"Heatmapa zapisana: {heatmap_filename}")
plt.show()

# --- WEIGHTED K-MEANS ---
print("\nLiczenie Ważonego K-Means...")
# Do K-Means też musimy podać dane zestandaryzowane. 
# Żeby być spójnym z PCA, użyjmy tych samych danych co w funkcji PCA:
weighted_mean = np.average(X_df, axis=0, weights=weights)
weighted_std = np.sqrt(np.average((X_df - weighted_mean)**2, axis=0, weights=weights))
X_scaled_weighted = (X_df - weighted_mean) / weighted_std

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled_weighted, sample_weight=weights)

# --- PLOTTING CLUSTERS ---
df_clustered = df.copy()
df_clustered['Cluster'] = clusters
df_clustered['PCA1'] = X_pca_data[:, 0]
df_clustered['PCA2'] = X_pca_data[:, 1]

plt.figure(figsize=(14, 6))

# Plot 1: Scatter plot
plt.subplot(1, 2, 1)
sns.scatterplot(
    x='PCA1', y='PCA2', 
    hue='Cluster', 
    data=df_clustered, 
    palette='viridis', 
    s=60, alpha=0.8
)
plt.title('Klastry na tle Ważonego PCA')
plt.xlabel('PC1 (Weighted)')
plt.ylabel('PC2 (Weighted)')

# Plot 2: Box plot
df_melted = df_clustered.melt(id_vars=['Cluster'], value_vars=grade_cols, var_name='Subject', value_name='Grade')
plt.subplot(1, 2, 2)
sns.boxplot(x='Subject', y='Grade', hue='Cluster', data=df_melted, palette='viridis')
plt.title('Wyniki w nauce wg klastrów')
plt.xticks(rotation=15)

plt.tight_layout()
subjects_str = "_".join(grade_cols)
filename = f"{file_prefix}_Klastry_{n_clusters}.jpg"
plt.savefig(filename, format='jpg', dpi=300)
print(f"Wykres zapisany: {filename}")
plt.show()

# --- WEIGHTED STATISTICS ---
print("\nWeighted Average Grades by Cluster:")
for cluster_id in range(n_clusters):
    subset = df_clustered[df_clustered['Cluster'] == cluster_id]
    subset_weights = weights[df_clustered['Cluster'] == cluster_id] # Filtrujemy też wagi!
    
    if len(subset) > 0:
        w_avg_math = np.average(subset['ST_Sgrade_Math'], weights=subset_weights)
        w_avg_read = np.average(subset['ST_Sgrade_Read_Lang'], weights=subset_weights)
        print(f"Cluster {cluster_id}: Math={w_avg_math:.2f}, Read={w_avg_read:.2f}, N={len(subset)}")