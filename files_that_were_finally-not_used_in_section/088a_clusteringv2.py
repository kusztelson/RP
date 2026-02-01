import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienia
n_clusters = 3  # Liczba klastrów wyciągnięta do zmiennej
grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang']

# Wczytywanie danych
df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("data_full_data_withoutArts.csv")
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]

# Wybór kolumn
df = df[
    ["ST_ASS_WLE_ADJ",
    "ST_COO_WLE_ADJ",
    "ST_CRE_WLE_ADJ",
    "ST_CUR_WLE_ADJ",
    "ST_EFF_WLE_ADJ",
    "ST_EMO_WLE_ADJ",
    "ST_EMP_WLE_ADJ",
    "ST_ENE_WLE_ADJ",
    "ST_MOT_WLE_ADJ",
    "ST_OPT_WLE_ADJ",
    "ST_PER_WLE_ADJ",
    "ST_RES_WLE_ADJ",
    "ST_SEL_WLE_ADJ",
    "ST_SOC_WLE_ADJ",
    "ST_STR_WLE_ADJ",
    "ST_TOL_WLE_ADJ",
    "ST_TRU_WLE_ADJ",
    #"ST_ARS",
    #"ST_ARS_PAIRS",
    #"IMMBACK",
    "ST_st_relteach",   # bully reltech and anxtest together rmsea - 0.035 (Cohort) , 0.056 (site) , 0.035 (gender)
    "ST_st_bully",      # bully reltech and anxtest together rmsea - 0.035 (Cohort) , 0.056 (site) , 0.035 (gender)
    "ST_st_belong",    # rmsea - 0.064 (Cohort) , 0.075 (site) , 0.044 (gender)
    "ST_st_friends",   # friends and parents together # rmsea - 0.030 (Cohort) , 0.065 (site) , 0.028 (gender)
    "ST_st_relpar",    # friends and parents together # rmsea - 0.030 (Cohort) , 0.065 (site) , 0.028 (gender)
    "ST_st_globalmind",   # rmsea - 0.099 (Cohort) , 0.134 (site) , 0.086 (gender) # aboce 0.1 = pproblem withh generalization across sites
    "ST_st_wellbeing",   # rmsea - 0.066 (Cohort) , 0.075 (site) , 0.070 (gender)  
    "ST_st_anxtest",        # bully reltech and anxtest together rmsea - 0.035 (Cohort) , 0.056 (site) , 0.035 (gender)
    "ST_SES",
    'ST_Sgrade_Math',
    'ST_Sgrade_Read_Lang'
    #'ST_Sgrade_Arts'
    ]]

df1 = df.dropna()

# Przygotowanie danych do klastrowania
X_df = df.drop(columns=grade_cols)
X_df = X_df.select_dtypes(include=[np.number]).dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# część z analizą pca

# --- NOWA CZĘŚĆ: ANALIZA ŁADUNKÓW PCA (HEATMAPA) ---

# 1. Tworzenie DataFrame z wagami (transponujemy macierz, żeby cechy były wierszami)
# pca.components_ ma wymiar [n_components, n_features], my chcemy odwrotnie.
loadings_df = pd.DataFrame(
    pca.components_.T, 
    columns=['PC1', 'PC2'], 
    index=X_df.columns
)

# 2. Rysowanie Heatmapy
# Ustawiamy dużą wysokość (figsize=(8, 12)), bo masz dużo zmiennych (ST_...)
plt.figure(figsize=(8, 14))

sns.heatmap(
    loadings_df, 
    annot=True,       # Pokaż liczby
    cmap='coolwarm',  # Czerwony = dodatnie, Niebieski = ujemne
    center=0,         # Zero na biało
    fmt='.2f',        # Formatowanie liczb do 2 miejsc po przecinku
    linewidths=0.5    # Linie oddzielające kratki
)

plt.title('Wpływ zmiennych na składowe PCA (Loadings)')
plt.tight_layout() # Ważne, żeby nie ucięło etykiet przy zapisie

# 3. Zapisywanie Heatmapy do pliku
heatmap_filename = f"PCA_Loadings_{n_clusters}.jpg"
plt.savefig(heatmap_filename, format='jpg', dpi=300, bbox_inches='tight')

print(f"Heatmapa ładunków PCA została zapisana jako: {heatmap_filename}")
plt.show()

# Opcjonalnie: Wyświetlenie tekstowe najważniejszych cech
print("\n--- Najważniejsze cechy dla PC1 (Top 5) ---")
print(loadings_df['PC1'].abs().sort_values(ascending=False).head(5))

print("\n--- Najważniejsze cechy dla PC2 (Top 5) ---")
print(loadings_df['PC2'].abs().sort_values(ascending=False).head(5))







# Klastrowanie z użyciem zmiennej n_clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_clustered = df.loc[X_df.index].copy()
df_clustered['Cluster'] = clusters
df_clustered['PCA1'] = X_pca[:, 0]
df_clustered['PCA2'] = X_pca[:, 1]

# Rysowanie wykresów
plt.figure(figsize=(14, 6))

# Wykres 1: Scatter plot
plt.subplot(1, 2, 1)
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_clustered, palette='viridis', s=60)
plt.title('Clusters based on Social-Emotional Skills and Socioeconomic (Grades Excluded)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Wykres 2: Box plot
df_melted = df_clustered.melt(id_vars=['Cluster'], value_vars=grade_cols, var_name='Subject', value_name='Grade')

plt.subplot(1, 2, 2)
sns.boxplot(x='Subject', y='Grade', hue='Cluster', data=df_melted, palette='viridis')
plt.title('Academic Performance Patterns by Cluster')
plt.xticks(rotation=15)

plt.tight_layout()

# --- ZAPISYWANIE PLIKU ---
# Tworzenie nazwy pliku na podstawie parametrów
subjects_str = "_".join(grade_cols)  # Łączy nazwy przedmiotów podkreślnikiem
filename = f"Klastry_{n_clusters}_Przedmioty_{subjects_str}.jpg"

# Zapis (dpi=300 dla lepszej jakości)
plt.savefig(filename, format='jpg', dpi=300)
print(f"Wykres został zapisany jako: {filename}")

plt.show()

print("Average Grades by Cluster:")
print(df_clustered.groupby('Cluster')[grade_cols].mean())



