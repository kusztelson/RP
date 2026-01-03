import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURACJA
# ==========================================
n_clusters = 3
grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang']
weight_col = 'ST_WT2019'
file_prefix = "noPCA_finn_FINAL"
report_filename = f"{file_prefix}_RAPORT_KLASTRY.txt"

# ==========================================
# 2. WCZYTANIE I CZYSZCZENIE DANYCH
# ==========================================
print(">>> Wczytywanie danych...")
df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("data_full_data_withoutArts.csv")

# Filtry
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]
df = df[(df['ST_SiteID'] == 6.0) & (df['ST_CohortID'] == 2.0) & (df['ST_Gender_Std'] == 2.0)]

# Wybór kolumn
psych_traits = [
    "ST_RES_WLE_ADJ", "ST_SEL_WLE_ADJ", "ST_st_relteach", "ST_st_bully", 
    "ST_st_belong", "ST_st_friends", "ST_st_relpar", "ST_st_wellbeing", 
    "ST_st_anxtest", "ST_SES"
]
all_cols = psych_traits + grade_cols + [weight_col]
df = df[all_cols].dropna().reset_index(drop=True)

# ==========================================
# 3. PRZYGOTOWANIE I KLASTROWANIE
# ==========================================
print(">>> Przetwarzanie (Ważona standaryzacja i K-Means)...")

X_traits = df[psych_traits]
weights = df[weight_col].values 
weights = weights / weights.mean() # Normalizacja wag

# Ważona średnia i odchylenie (do standaryzacji)
w_mean = np.average(X_traits, axis=0, weights=weights)
w_var = np.average((X_traits - w_mean)**2, axis=0, weights=weights)
w_std = np.sqrt(w_var)

# Klastrowanie na danych zestandaryzowanych
X_clustering = (X_traits - w_mean) / w_std
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_clustering, sample_weight=weights)
df['Cluster'] = clusters

# ==========================================
# 4. GENEROWANIE RAPORTU DO PLIKU .TXT
# ==========================================
print(f">>> Generowanie raportu do pliku: {report_filename} ...")

# Funkcja pomocnicza do liczenia średniej ważonej
def get_weighted_mean(series, w):
    return np.average(series, weights=w)

# Obliczamy średnie globalne dla całej populacji (do porównań)
global_stats = {}
for col in psych_traits + grade_cols:
    global_stats[col] = get_weighted_mean(df[col], weights)

with open(report_filename, "w", encoding="utf-8") as f:
    # --- NAGŁÓWEK ---
    f.write("="*60 + "\n")
    f.write(f"RAPORT KLASTRÓW SSES (FINLANDIA, COHORT 2)\n")
    f.write(f"Liczba studentów (N): {len(df)}\n")
    f.write("="*60 + "\n\n")
    
    f.write("LEGENDA:\n")
    f.write("  - Mean: Średnia ważona w klastrze\n")
    f.write("  - Diff: Różnica względem średniej globalnej (Mean - Global)\n")
    f.write("-" * 60 + "\n\n")

    # --- PĘTLA PO KLASTRACH ---
    for c in range(n_clusters):
        # Filtrujemy dane dla konkretnego klastra
        sub_df = df[df['Cluster'] == c]
        sub_weights = weights[df['Cluster'] == c]
        n_students = len(sub_df)
        pct_students = (n_students / len(df)) * 100
        
        f.write(f"KLASTER {c}  (N = {n_students} | {pct_students:.1f}% populacji)\n")
        f.write("-" * 60 + "\n")
        
        # Formatowanie tabeli
        # Kolumny: Nazwa Cechy | Średnia Ważona | Różnica od Globalnej
        header = f"{'CECHA / ZMIENNA':<25} | {'MEAN':>8} | {'DIFF':>8}"
        f.write(header + "\n")
        f.write("-" * 60 + "\n")
        
        # 1. Wyniki w nauce (Grades)
        f.write(" [ WYNIKI W NAUCE ]\n")
        for col in grade_cols:
            w_avg = get_weighted_mean(sub_df[col], sub_weights)
            diff = w_avg - global_stats[col]
            f.write(f"{col:<25} | {w_avg:8.2f} | {diff:+8.2f}\n")
            
        f.write("\n [ CECHY PSYCHOLOGICZNE ]\n")
        # Sortujemy cechy psychologiczne wg "siły" odchylenia w tym klastrze
        # Dzięki temu na górze listy będą cechy najbardziej charakterystyczne dla grupy
        psych_results = []
        for col in psych_traits:
            w_avg = get_weighted_mean(sub_df[col], sub_weights)
            diff = w_avg - global_stats[col]
            # Normalizujemy różnicę przez odchylenie std (z-score), żeby sortowanie miało sens
            # (bo SES ma inną skalę niż WLE)
            col_idx = psych_traits.index(col)
            z_score = diff / w_std[col_idx]
            psych_results.append((col, w_avg, diff, z_score))
        
        # Sortowanie: od najbardziej pozytywnych do najbardziej negatywnych cech
        psych_results.sort(key=lambda x: x[3], reverse=True)
        
        for col, w_avg, diff, z_score in psych_results:
            # Dodajemy gwiazdkę (*) jeśli cecha jest bardzo silna (> 0.5 std dev)
            marker = "(*)" if abs(z_score) > 0.5 else ""
            name_display = f"{col} {marker}"
            f.write(f"{name_display:<25} | {w_avg:8.2f} | {diff:+8.2f}\n")
            
        f.write("\n" + "="*60 + "\n\n")

print(f"Gotowe! Raport został zapisany w pliku: {report_filename}")