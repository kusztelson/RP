import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("data_full_data.csv")
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]



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
    "ST_ARS",
    "ST_ARS_PAIRS",
    #"IMMBACK",
    "ST_st_relteach",
    "ST_st_bully",
    "ST_st_belong",
    "ST_st_friends",
    "ST_st_relpar",
    "ST_st_globalmind",
    "ST_st_wellbeing",
    "ST_st_anxtest",
    "ST_SES",
    'ST_Sgrade_Math',
    'ST_Sgrade_Read_Lang',
    'ST_Sgrade_Arts']]


df1 = df.dropna()


grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts']
X_df = df.drop(columns=grade_cols)

X_df = X_df.select_dtypes(include=[np.number]).dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)


df_clustered = df.loc[X_df.index].copy()
df_clustered['Cluster'] = clusters
df_clustered['PCA1'] = X_pca[:, 0]
df_clustered['PCA2'] = X_pca[:, 1]


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_clustered, palette='viridis', s=60)
plt.title('Clusters based on Social-Emotional Skills and Socioeconomic (Grades Excluded)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')


df_melted = df_clustered.melt(id_vars=['Cluster'], value_vars=grade_cols, var_name='Subject', value_name='Grade')

plt.subplot(1, 2, 2)
sns.boxplot(x='Subject', y='Grade', hue='Cluster', data=df_melted, palette='viridis')
plt.title('Academic Performance Patterns by Cluster')
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

print("Average Grades by Cluster:")
print(df_clustered.groupby('Cluster')[grade_cols].mean())