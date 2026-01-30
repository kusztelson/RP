import time
from collections import defaultdict, Counter
from urllib.error import HTTPError

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns

SAVE_DIR = "plots"


def generate_tree(X, y, standardize=True, sample_weight=None, **kwargs):
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if sample_weight is None:
        sample_weight = np.ones(shape=X.shape[0])

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
X, y, sample_weight, test_size=0.1, random_state=42)
    clf = tree.DecisionTreeRegressor(**kwargs)

    clf = clf.fit(X_train, y_train, sample_weight=w_train)

    return clf, (X_train, X_test, y_train, y_test, w_train, w_test)


def plot_tree(tree_model: tree.DecisionTreeRegressor, plot_suffix: str, feature_names, depth=3, site=""):
    plt.figure(figsize=((depth ** 2), 2 * depth))
    tree.plot_tree(tree_model, feature_names=feature_names,
                   filled=True, fontsize=8)
    name = "Decision tree - " + plot_suffix
    plt.title(name)
    plt.tight_layout()
    filename = plot_suffix.replace(": ", " ")
    filename_parts = filename.split(", ")
    filename = " ".join(filename_parts[-2:])

    dir_path = SAVE_DIR
    if site != "":
        dir_path = os.path.join(SAVE_DIR, site)

    plt.savefig(os.path.join(dir_path, filename),
                dpi=300
                )
    # plt.show()


def plot_true_vs_pred(x_true_train, x_pred_train, x_true_test, x_pred_test, title):
    plt.figure(figsize=(8, 6))

    plt.scatter(x_true_train, x_pred_train, color='green', label="Train")
    plt.scatter(x_true_test, x_pred_test, color='purple', label="Test")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.legend(loc="upper left", fontsize=12)

    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# --- SETTINGS ---
grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts', 'ST_Grade_Mean']

skills = [
    "ST_PER_WLE_ADJ", "ST_RES_WLE_ADJ",
    "ST_SEL_WLE_ADJ",
]

# skills = [
#     "ST_ASS_WLE_ADJ", "ST_COO_WLE_ADJ", "ST_CRE_WLE_ADJ", "ST_CUR_WLE_ADJ",
#     "ST_EFF_WLE_ADJ", "ST_EMO_WLE_ADJ", "ST_EMP_WLE_ADJ", "ST_ENE_WLE_ADJ",
#     "ST_MOT_WLE_ADJ", "ST_OPT_WLE_ADJ", "ST_PER_WLE_ADJ", "ST_RES_WLE_ADJ",
#     "ST_SEL_WLE_ADJ", "ST_SOC_WLE_ADJ", "ST_STR_WLE_ADJ", "ST_TOL_WLE_ADJ",
#     "ST_TRU_WLE_ADJ"
# ]

control_var = [
    "ST_st_relteach", "ST_st_bully",
    "ST_st_belong", "ST_st_anxtest",
]

# control_var = [
#     "ST_WT2019",
#     "ST_st_relteach", "ST_st_bully", "ST_st_belong",
#     "ST_st_friends", "ST_st_relpar", "ST_st_globalmind", "ST_st_wellbeing",
#     "ST_st_anxtest"
# ]

SES = [
    # "ST_WT2019",
    "ST_SES"
]

# --- DATA LOADING & PREPROCESSING ---
# (Using your provided logic)
df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("data_full_data.csv")
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]

df['ST_Grade_Mean'] = df[['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts']].mean(axis=1)

df_clean = df[["ST_SiteID", 'ST_WT2019'] + skills + control_var + SES + grade_cols].dropna()

ignore_features = 2
scaler = StandardScaler()
df_clean = df_clean.reset_index(drop=True)
X_scaled = scaler.fit_transform(df_clean.iloc[:, ignore_features:])

df_scaled = pd.DataFrame(X_scaled, columns=df_clean.columns[ignore_features:])

for col in df_scaled.columns:
    df_clean[col] = df_scaled[col]

# df_clean.update(df_scaled)



site_codes = {
    1.0: "Ottawa",
    2.0: "Houston",
    3.0: "Bogota",
    4.0: "Manizales",
    6.0: "Helsinki",
    7.0: "Moscow",
    8.0: "Istanbul",
    9.0: "Daegu",
    10.0: "Sintra",
    11.0: "Suzhou"
}

model_x_set = [
    SES + control_var,
    skills + control_var,
    SES + skills + control_var
]

model_x_names = [
    "SES + controls",
    "skills + controls",
    "SES + skills + controls"
]

max_depth = 6
tree_params_config = {
    "max_depth": max_depth,
    "min_samples_leaf": 250,
    "min_samples_split": 500,
    "random_state": 42
}

top_features_per_site = defaultdict(lambda: defaultdict(list))
score_per_site = defaultdict(lambda: defaultdict(list))
score_set = {
    "site": [],
    "model_type": [],
    "Math": [],
    "Reading": [],
    "Art": [],
    "Mean": [],
}

ignore_index = 0
selected_site = 5
for site in df["ST_SiteID"].unique().tolist():
    # if ignore_index != selected_site:
    #     ignore_index += 1
    #     continue

    ignore_index += 1
    site_mask = df_clean["ST_SiteID"] == site
    df_site = df_clean[site_mask]
    # df_site = df_clean

    tree_params = tree_params_config.copy()
    if 'min_samples_leaf' in tree_params_config:
        tree_params['min_samples_leaf'] = int(np.round(df_site.shape[0] * 0.05))

    if 'min_samples_split' in tree_params_config:
        tree_params['min_samples_split'] = int(np.round(df_site.shape[0] * 0.1))

    for model_id, model_type in enumerate(model_x_set):
        score_set['site'].append(site)
        score_set['model_type'].append(model_id)

        for y_col in grade_cols:
            X_df = df_site[model_type]
            y_df = df_site[y_col]

            tree_model, data = generate_tree(X_df, y_df,
                                             # sample_weight=df_site['ST_WT2019'],
                                             **tree_params)

            name = f"site: {site_codes[site]}, model: {model_x_names[model_id]}, predicting: {y_col}"
            # while True:
            #     try:
            #         plot_tree(tree_model, name, list(X_df.columns), max_depth, site=site_codes[site])
            #         break
            #     except HTTPError:
            #         time.sleep(10)

            y_pred_train = tree_model.predict(data[0])
            r2_train = r2_score(data[2], y_pred_train)

            y_pred = tree_model.predict(data[1])
            r2_test = r2_score(data[3], y_pred)

            plot_true_vs_pred(data[2], y_pred_train, data[3], y_pred,
                              f"True vs Predicted: ({site_codes[site]}, {model_x_names[model_id]}, {y_col})")

            # grade_cols = ['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts', 'ST_Grade_Mean']

            if y_col == 'ST_Sgrade_Math':
                score_set['Math'].append(r2_test)
            elif y_col == 'ST_Sgrade_Read_Lang':
                score_set['Reading'].append(r2_test)
            elif y_col == 'ST_Sgrade_Arts':
                score_set['Art'].append(r2_test)
            elif y_col == 'ST_Grade_Mean':
                score_set['Mean'].append(r2_test)

            score_per_site[site][y_col].extend([r2_test])
            # print("=" * 15)
            # print(name)
            # print(f"R squared: \ntrain: {r2_train}\ntest: {r2_test}\n")

            fi = pd.Series(tree_model.feature_importances_, index=X_df.columns)
            # print("Most important features:")
            # print(fi.sort_values(ascending=False))

            top_features = fi.sort_values(ascending=False).head(3).index.tolist()
            top_features_per_site[site][y_col].extend(top_features)

            # y_pred = tree_model.predict(data[1])
            # residuals = data[3] - y_pred
            # plt.scatter(y_pred, residuals)
            # plt.axhline(0, color='red', linestyle='--')
            # plt.xlabel("Predicted")
            # plt.ylabel("Residuals")
            # plt.show()

# exit()
df = pd.DataFrame(
    score_set
)

df_long_org = df.melt(
    id_vars=["site", "model_type"],
    value_vars=["Math", "Reading", "Art", "Mean"],
    var_name="data_type",
    value_name="score"
)

df_long = df_long_org.copy()
df_long['site'] = df_long['site'].map(site_codes)

print(df_long)

models = df_long["model_type"].unique()

# fig, axes = plt.subplots(len(models), 1, figsize=(7, 6 * len(models)), sharex=True)
#
# if len(models) == 1:
#     axes = [axes]  # handle edge case
#
# for ax, current_y in zip(axes, models):
#     sub = df_long[df_long["model_type"] == current_y]
#
#     # Create matrix: rows = data_type, columns = site
#     pivot = sub.pivot(
#         index="data_type",
#         columns="site",
#         values="score"
#     )
#
#     im = ax.imshow(pivot.values, aspect="auto")
#
#     # Ticks & labels
#     ax.set_xticks(np.arange(len(pivot.columns)))
#     ax.set_yticks(np.arange(len(pivot.index)))
#
#     ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
#     ax.set_yticklabels(pivot.index)
#     if current_y == 1:
#         ax.set_ylabel("Data Type")
#         cbar_ax = fig.add_axes([0.90, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
#         fig.colorbar(im, cax=cbar_ax)
#
#     ax.set_title(f"{model_x_names[current_y]}")
#     # ax.set_xlabel("Site")
#
#     # if model == 2:
#     #     cax = ax.inset_axes([0.3, 0.7, 0.4, 0.04])
#     #     fig.colorbar(im, cax=cax, orientation='vertical')
#
#     # Write numbers in cells (optional but very useful)
#     for i in range(pivot.shape[0]):
#         for j in range(pivot.shape[1]):
#             ax.text(j, i, f"{pivot.values[i, j]:.2f}",
#                     ha="center", va="center", fontsize=9)
#
# # Shared colorbar
# # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# # fig.colorbar(im, cax=cbar_ax)
# # plt.tight_layout()
# plt.subplots_adjust(
#     left=0.1,   # space on the left
#     right=0.8, # space on the right (leave room for colorbar)
#     top=0.95,   # top margin
#     bottom=0.05, # bottom margin
#     wspace=0.4, # horizontal space between subplots
#     hspace=0.1  # vertical space between subplots
# )
# plt.show()


y_types = df_long_org["data_type"].unique()

present_sites = df_long_org["site"].unique()
new_sites_dict = {i: site_codes[site] for i, site in enumerate(present_sites)}
new_sites_codes = {site: i for i, site in enumerate(present_sites)}
df_long_org["site"] = df_long_org["site"].map(new_sites_codes)
present_sites = df_long_org["site"].unique()

fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

colors = ["#1B9E77",  # Teal / green
          "#F0E442",  # Yellow
          "#D7191C"]

axes = axes.flatten()

if len(models) == 1:
    axes = [axes]  # handle edge case

for ax, current_y in zip(axes, y_types):
    sub = df_long_org[df_long_org["data_type"] == current_y]

    # Create matrix: rows = data_type, columns = site
    pivot = sub.pivot(
        index="model_type",
        columns="site",
        values="score"
    )

    for i, y_type in enumerate(models):
        y_mask = sub["model_type"] == y_type
        # print(sub[model_mask]["site"])
        ax.plot(sub[y_mask]["site"], sub[y_mask]["score"], label=f"{model_x_names[y_type]}", color=colors[i])

    global_mean = sub["score"].mean()
    ax.plot(present_sites, np.repeat(global_mean, present_sites.shape[0]),
            linestyle="--", label="Ensemble mean", color="black")

    ax.legend(loc="lower right", fontsize=12)

    ax.annotate(f'{np.round(global_mean, 3)}', xy=(1, global_mean), xycoords='data',
                xytext=(0, global_mean + 0.1), textcoords='data',
                va='top', ha='left',
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    ax.set_ylim(-0.2, 0.2)
    ax.set_yticks(np.linspace(-0.2, 0.2, 5))

    ax.set_xticks(list(present_sites))
    ax.set_xticklabels([new_sites_dict[s] for s in present_sites], rotation=45, ha="right")  # map codes to names

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    metric_name = current_y
    if metric_name == "Mean":
        metric_name = "grades mean"
    else:
        metric_name = str.lower(current_y) + " grade"
    ax.set_title(f"R² scores per site predicting {metric_name}", fontsize=15)

# Shared colorbar
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# fig.colorbar(im, cax=cbar_ax)
plt.tight_layout()
plt.ylim(-0.2, 0.2)
# plt.subplots_adjust(
#     left=0.1,   # space on the left
#     right=0.8, # space on the right (leave room for colorbar)
#     top=0.95,   # top margin
#     bottom=0.05, # bottom margin
#     wspace=0.4, # horizontal space between subplots
#     hspace=0.1  # vertical space between subplots
# )
plt.savefig("R2 comparison between models.png")
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(6, 15), sharex=True, sharey=True)

colors = ["#1B9E77",  # Teal / green
          "#F0E442",  # Yellow
          "#D7191C",
          "#8119D7"]

axes = axes.flatten()

if len(models) == 1:
    axes = [axes]  # handle edge case

for ax, current_y in zip(axes, models):
    sub = df_long[df_long["model_type"] == current_y]

    # Create matrix: rows = data_type, columns = site

    for i, y_type in enumerate(y_types):
        y_mask = sub["data_type"] == y_type
        # print(sub[model_mask]["site"])
        ax.plot(sub[y_mask]["site"], sub[y_mask]["score"], label=f"{y_type}", color=colors[i])

    global_mean = sub["score"].mean()
    ax.plot(present_sites, np.repeat(global_mean, present_sites.shape[0]),
            linestyle="--", label="Ensemble mean", color="black")

    ax.legend(loc="lower right", fontsize=12)

    ax.annotate(f'{np.round(global_mean, 3)}', xy=(1, global_mean), xycoords='data',
                xytext=(0, global_mean + 0.1), textcoords='data',
                va='top', ha='left',
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    ax.set_ylim(-0.2, 0.2)
    ax.set_yticks(np.linspace(-0.2, 0.2, 5))

    ax.set_xticks(list(present_sites))
    ax.set_xticklabels([new_sites_dict[s] for s in present_sites], rotation=45, ha="right")

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    ax.set_title(f"R² scores per site predicted by {model_x_names[current_y]} model", fontsize=12)

# Shared colorbar
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# fig.colorbar(im, cax=cbar_ax)
plt.tight_layout()
plt.ylim(-0.2, 0.2)
# plt.subplots_adjust(
#     left=0.1,   # space on the left
#     right=0.8, # space on the right (leave room for colorbar)
#     top=0.95,   # top margin
#     bottom=0.05, # bottom margin
#     wspace=0.4, # horizontal space between subplots
#     hspace=0.1  # vertical space between subplots
# )
plt.savefig("R2 comparison between ys.png")
plt.show()

exit()
for site, y_dict in top_features_per_site.items():
    print(f"\nSite: {site_codes[site]}")
    for y_col, features in y_dict.items():
        # Count occurrence of each feature
        feature_counts = Counter(features)
        # Sort by most common
        most_common = feature_counts.most_common()
        print(f"  Target: {y_col}")
        print(f"    Most common features (with counts): {most_common}")

for site, y_dict in score_per_site.items():
    print(f"\nSite: {site_codes[site]}")
    for y_col, scores in y_dict.items():
        # Count occurrence of each feature
        score_sum = np.median(scores)
        # Sort by most common
        print(f"  Target: {y_col}")
        print(f"    R2 score: {score_sum}")

for site, y_dict in score_per_site.items():
    row = f"{site_codes[site]}"
    for y_col, scores in y_dict.items():
        # Count occurrence of each feature
        score_sum = np.max(scores)
        # Sort by most common
        row += f";{score_sum}"
    print(row)
