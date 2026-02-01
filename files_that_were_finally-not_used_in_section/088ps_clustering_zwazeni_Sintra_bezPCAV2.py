import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# ==========================================
# 1. KONFIGURACJA I ŚCIEŻKI
# ==========================================
n_clusters = 3
grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang']
# Lista cech psychologicznych do klastrowania
psych_traits = [
    "ST_RES_WLE_ADJ", "ST_SEL_WLE_ADJ", "ST_st_relteach", "ST_st_bully", 
    "ST_st_belong", "ST_st_friends", "ST_st_relpar", "ST_st_wellbeing", 
    "ST_st_anxtest", "ST_SES"
]
weight_col = 'ST_WT2019'

# Nazwy plików wyjściowych
file_prefix = "noPCAv2_Sintra_COMPLETE_ANALYSIS"
plot_filename = f"{file_prefix}_Charts.jpg"
report_filename = f"{file_prefix}_Report.txt"

# ==========================================
# 2. WCZYTANIE I FILTROWANIE DANYCH
# ==========================================
print(">>> 1/6 Wczytywanie i filtrowanie danych (Finlandia, Cohort 2)...")
try:
    df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
    df_st_filter = pd.read_csv("data_full_data_withoutArts.csv")
except FileNotFoundError as e:
    print(f"BŁĄD: Nie znaleziono plików z danymi. Sprawdź ścieżki.\nSzczegóły: {e}")
    sys.exit(1)

# Filtrowanie po ID studenta
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]

# Filtry demograficzne: Finowie (6.0) + Kohorta 2 + Płeć 2
df = df[
    (df['ST_SiteID'] == 10.0) & 
    (df['ST_CohortID'] == 2.0) & 
    (df['ST_Gender_Std'] == 2.0)
]

# Wybór kolumn i usuwanie braków
cols_to_keep = psych_traits + grade_cols + [weight_col]
df = df[cols_to_keep].dropna().reset_index(drop=True)
print(f"Liczba studentów do analizy (N): {len(df)}")

# ==========================================
# 3. PRZYGOTOWANIE DANYCH (WAŻONA STANDARYZACJA)
# ==========================================
print(">>> 2/6 Obliczanie globalnych statystyk ważonych i standaryzacja...")

X_raw = df[psych_traits]        # Surowe cechy psychologiczne
weights = df[weight_col].values # Wagi
weights = weights / weights.mean() # Normalizacja wag dla stabilności

# --- KLUCZOWE: Obliczamy WAŻONĄ średnią i odchylenie globalne ---
# Potrzebujemy tego do standaryzacji danych ORAZ do raportu (jako punkt odniesienia)
global_w_mean = np.average(X_raw, axis=0, weights=weights)
global_w_var = np.average((X_raw - global_w_mean)**2, axis=0, weights=weights)
global_w_std = np.sqrt(global_w_var)

# Tworzymy słowniki do szybkiego dostępu w raporcie później
global_stats_map = {col: {'mean': m, 'std': s} for col, m, s in zip(psych_traits, global_w_mean, global_w_std)}

# Standaryzacja danych wejściowych do K-Means (Z-score ważony)
X_std = (X_raw - global_w_mean) / global_w_std

# ==========================================
# 4. KLASTROWANIE (WAŻONE K-MEANS)
# ==========================================
print(f">>> 3/6 Uruchamianie Ważonego K-Means (k={n_clusters})...")

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# sample_weight = wagi uczniów wpływają na położenie centroidów
clusters = kmeans.fit_predict(X_std, sample_weight=weights)

# Dodajemy wyniki do głównego DataFrame
df_final = df.copy()
df_final['Cluster'] = clusters

# ==========================================
# 5. WAŻONE PCA (TYLKO DO WIZUALIZACJI)
# ==========================================
print(">>> 4/6 Obliczanie PCA do tła wykresu...")
# Liczymy macierz kowariancji z wagami (aweights)
cov_matrix = np.cov(X_std.T, aweights=weights)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
# Sortujemy i bierzemy 2 główne komponenty
sorted_idx = np.argsort(eigenvalues)[::-1]
top_eigenvectors = eigenvectors[:, sorted_idx][:, :2]
# Rzutujemy dane
X_pca_2d = np.dot(X_std, top_eigenvectors)

df_final['PCA1'] = X_pca_2d[:, 0]
df_final['PCA2'] = X_pca_2d[:, 1]

# ==========================================
# 6. GENEROWANIE WYKRESÓW (ZAPIS DO JPG)
# ==========================================
print(f">>> 5/6 Generowanie wykresów do pliku: {plot_filename} ...")

plt.figure(figsize=(16, 7))

# --- LEWY WYKRES: Mapa Klastrów (na tle PCA) ---
plt.subplot(1, 2, 1)
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Cluster', 
    data=df_final, palette='viridis', s=60, alpha=0.8, edgecolor='k'
)
plt.title('Mapa Uczniów (Klastry na tle PCA cech psychologicznych)')
plt.xlabel('Wymiar PCA 1')
plt.ylabel('Wymiar PCA 2')
plt.legend(title='Klaster')

# --- PRAWY WYKRES: Wyniki w Nauce (Boxploty) ---
plt.subplot(1, 2, 2)
# Konwersja do formatu długiego dla seaborn
df_grades_melted = df_final.melt(
    id_vars=['Cluster'], value_vars=grade_cols, 
    var_name='Przedmiot', value_name='Ocena'
)
# Ładniejsze nazwy na osi
df_grades_melted['Przedmiot'] = df_grades_melted['Przedmiot'].replace({
    'ST_Sgrade_Math': 'Matematyka', 'ST_Sgrade_Read_Lang': 'Język/Czytanie'
})

sns.boxplot(x='Przedmiot', y='Ocena', hue='Cluster', data=df_grades_melted, palette='viridis')
plt.title('Rozkład Ocen w Klastrach Psychologicznych')
plt.ylim(0, 51) # Zakładam skalę 1-10
plt.ylabel('Ocena (Surowa)')

plt.tight_layout()
plt.savefig(plot_filename, format='jpg', dpi=300)
plt.close() # Zamykamy wykres, żeby nie wisiał w pamięci

# ==========================================
# 7. GENEROWANIE RAPORTU (ZAPIS DO TXT)
# ==========================================
print(f">>> 6/6 Generowanie szczegółowego raportu do pliku: {report_filename} ...")

# Funkcja pomocnicza
def weighted_avg(values, weights):
    return np.average(values, weights=weights)

with open(report_filename, "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write(f"RAPORT ANALIZY KLASTRÓW SSES (FINLANDIA, N={len(df_final)})\n")
    f.write("="*80 + "\n\n")
    f.write("LEGENDA KOLUMN:\n")
    f.write(" - Raw Weighted Mean: Średnia ważona w klastrze w oryginalnej skali punktowej.\n")
    f.write(" - STD VAL (Z-Score): Wartość zestandaryzowana.\n")
    f.write("   (Ile odchyleń standardowych średnia klastra różni się od średniej krajowej).\n")
    f.write("   Wartości > +0.5 lub < -0.5 oznaczają silne natężenie cechy.\n")
    f.write("-" * 80 + "\n\n")

    # Pętla po klastrach
    for c in range(n_clusters):
        # Filtrujemy dane dla klastra
        sub_df = df_final[df_final['Cluster'] == c]
        sub_weights = weights[df_final['Cluster'] == c]
        n_sub = len(sub_df)
        pct_sub = (n_sub / len(df_final)) * 100
        
        f.write(f" KLASTER {c}  [Liczebność: {n_sub} ({pct_sub:.1f}% populacji)]\n")
        f.write("=" * 80 + "\n")
        
        # --- SEKCJJA A: OCENY (Tylko średnie surowe) ---
        f.write(" A. WYNIKI W NAUCE (Oceny)\n")
        f.write(f" {'PRZEDMIOT':<25} | {'Raw Weighted Mean':>18}\n")
        f.write("-" * 46 + "\n")
        for g_col in grade_cols:
            w_mean_grade = weighted_avg(sub_df[g_col], sub_weights)
            f.write(f" {g_col:<25} | {w_mean_grade:>18.2f}\n")
        f.write("\n")

        # --- SEKCJA B: CECHY PSYCHOLOGICZNE (Surowe + Standaryzowane) ---
        f.write(" B. PROFIL PSYCHOLOGICZNY I SPOŁECZNY\n")
        
        # Zbieramy dane do posortowania
        trait_results = []
        for trait in psych_traits:
            # 1. Średnia ważona w klastrze (surowa skale)
            raw_cluster_mean = weighted_avg(sub_df[trait], sub_weights)
            
            # 2. Pobieramy globalne statystyki (policzone w kroku 3)
            g_mean = global_stats_map[trait]['mean']
            g_std = global_stats_map[trait]['std']
            
            # 3. Obliczamy Z-Score klastra (Standaryzowana wartość)
            # (Średnia klastra - Średnia globalna) / Odchylenie globalne
            z_score_cluster = (raw_cluster_mean - g_mean) / g_std
            
            trait_results.append((trait, raw_cluster_mean, z_score_cluster))
            
        # Sortujemy od najwyższego Z-Score (najsilniejsze cechy na górze)
        trait_results.sort(key=lambda x: x[2], reverse=True)
        
        # Zapisujemy tabelę
        # Nagłówek tabeli z formatowaniem
        header = f" {'CECHA (TRAIT)':<25} | {'Raw Weighted Mean':>18} | {'STD VAL (Z-Score)':>18}"
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        
        for trait, raw_mean, z_score in trait_results:
            # Dodajemy gwiazdkę dla silnych odchyleń (>0.5 std)
            marker = "(*)" if abs(z_score) > 0.5 else "   "
            trait_display = f"{trait} {marker}"
            
            # Formatowanie wiersza
            row_str = f" {trait_display:<25} | {raw_mean:>18.2f} | {z_score:>18.2f}"
            f.write(row_str + "\n")
            
        f.write("\n" + "="*80 + "\n\n\n")

print(f"\n>>> ZAKOŃCZONO SUKCESEM.")
print(f"Wykresy zapisano w: {plot_filename}")
print(f"Raport zapisano w: {report_filename}")