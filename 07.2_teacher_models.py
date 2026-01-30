import pandas
import pandas as pd
import numpy as np
from matplotlib.ticker import PercentFormatter
import pyreadstat
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
import matplotlib.pyplot as plt
# import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import Utils

# assumption that files generated in step 5 are


df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_teacher = pd.read_csv(
    "C:\\Users\\ppp\\Documents\\Witek\\Studia\\Laby\\Research Project\\main project\\Post CFA\\Post_CFA_INT_04_TCQ_(2021.04.14)_Public.csv")
df = pd.merge(df, df_teacher, left_on='ST_Username_TC', right_on='TCQ_Username_TC', how='left')

df['y'] = df[['ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts']].mean(axis=1)

# to make case as simple as possible for start we will remove rows containing any na in newly craftet features columns

df['MAX_PARENT_Occupation'] = df[['PA_MISEI_PA', 'PA_FISEI_PA']].max(axis=1)  # max from quality score of pparent occupation
df['MAX_PARENT_Education'] = df[['PA_PAQM00601', 'PA_PAQM00602']].max(axis = 1)  # max from education of parent


data_together = df[['MAX_PARENT_Occupation', 'MAX_PARENT_Education',
                    'ST_PER_WLE_ADJ', 'ST_CUR_WLE_ADJ',
                    # Teacher
                    'ST_LANG', 'TCQ_TCQM02801_x', 'TCQ_Experience',
                    'TCQ_Taught_Subject', 'TCQ_Teaching_Methods', 'TCQ_Student_Help',
                    'TCQ_Child_Development', 'TCQ_Educational_Risks', 'TCQ_Bullying',
                    # Y
                    'y']]

data_together = df

ignore = ['ST_LANG', 'ST_Start_date', 'ST_Start_time',
          'ST_End_date', 'ST_End_time', 'PA_LANG_PA',
          'PA_Start_date_PA', 'PA_Start_time_PA',
          'PA_End_date_PA', 'PA_End_time_PA',
          'TC_Start_date_TCA', 'TC_Start_time_TCA', 'TC_LANG_TCA',
          'TC_End_date_TCA', 'TC_End_time_TCA',
          'y',
          ]
for col in data_together.columns:
    if col in ignore:
        continue
    data_together[col] = data_together[col].fillna(data_together[col].median())

ignore += ['ST_FullID', 'Username_Std', 'ST_Username_TC',
           'PA_FullID_PA', 'PA_Username_PA',
           'TC_FullID_TC', 'TC_Username_TC',
           'TCQ_FullID_TC', 'TCQ_Username_TC',
           'TCQ_Username_TC',
           # Ignore all y values
           'ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts']

data_together = data_together.dropna()

print(data_together.shape)

X_tc_w = data_together[[column_name for column_name in data_together.columns if column_name not in ignore]]
# y_grade_avg = data_together['y']

print("Standardizing...")
scaler_tc_w = StandardScaler()
X_tc_std = scaler_tc_w.fit_transform(X_tc_w)
X_tc_std = pandas.DataFrame(np.column_stack((data_together[['ST_LANG']].to_numpy(), X_tc_std)),
                            columns=['ST_LANG'] + list(X_tc_w.columns))
print("Standardizing finished!")
# X_tc_std = X_tc_w.to_numpy()

lang_data = Utils.DataAnalyser(
    group_by=['ST_LANG'],
    ignore=ignore
)


features = X_tc_std.columns
f_index = 0
f_range = 30
top_n_number = 5
static_features = ['ST_LANG']
dynamic_y = ['y', 'ST_Sgrade_Math', 'ST_Sgrade_Read_Lang', 'ST_Sgrade_Arts']
for y_name in dynamic_y:
    current_y = data_together[y_name]
    # while f_index < len(features):
    #     f_range_top = f_index + f_range
    #
    #     if f_range_top >= len(features):
    #         f_range_top = len(features) - 1
    #
    #     current_features = features[f_index:f_range_top].append(pd.Index(static_features)).unique()
    #     preprocessed_data = lang_data.analyse_data(X_tc_std[current_features],
    #                                                current_y, Ridge(alpha=1))
    #
    #     # lang_data.display_data(preprocessed_data, plt.scatter, separate_graphs=False)
    #     print(lang_data.analyse_metadata(preprocessed_data, standardized=True, display=False).keys())
    #
    #     f_index += f_range

    current_features = features.append(pd.Index(static_features)).unique()
    preprocessed_data = lang_data.analyse_data(X_tc_std[current_features],
                                               current_y,
                                               LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], max_iter=10000),
                                               # RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10])
                                               )

    # lang_data.display_data(preprocessed_data, plt.scatter, separate_graphs=False)
    return_dict = lang_data.analyse_metadata(preprocessed_data, standardized=False, display=False)

    feature_count_dict = {}
    heat_map_data = None

    for group_name in return_dict:
        group_data = return_dict[group_name]
        group_data.name = group_name.replace('ST_LANG_', '')

        if heat_map_data is None:
            heat_map_data = group_data
        else:
            heat_map_data = pd.merge(heat_map_data, group_data, left_index=True, right_index=True)

        group_data = group_data.reindex(group_data.abs().sort_values(ascending=False).index)
        top_n = list(group_data[:top_n_number].index)

        for item in top_n:
            if item in feature_count_dict:
                feature_count_dict[item] += 1
            else:
                feature_count_dict[item] = 1

    feature_count_dict = dict(sorted(feature_count_dict.items(), key=lambda x: x[1], reverse=True))
    # group_data_dict -> 2D array
    heat_map_data_reduced = heat_map_data.transpose()[list(feature_count_dict.keys())]
    # return_dict -> labels x
    heatmap_x_labels = list(heat_map_data_reduced.columns)
    # feature_count_dict -> labels y
    heatmap_y_labels = list(heat_map_data_reduced.index)
    heat_map_data_numpy = heat_map_data_reduced.to_numpy().astype(float)

    # print(heat_map_data_numpy)
    # print(heatmap_x_labels)
    # print(heatmap_y_labels)
    # print(heat_map_data_numpy.dtype)
    Utils.plot_heatmap(heat_map_data_numpy, heatmap_x_labels,
                       heatmap_y_labels, f"Top {top_n_number} features for {y_name} value")
