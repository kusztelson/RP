import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
import os


class DataAnalyser:
    def __init__(self, group_by=None, ignore=None):
        self.group_by = group_by
        self.ignore = ignore

    def display_data(self, data_dict: dict,
                     plot_func=plt.plot, separate_graphs=False,
                     **kwargs):
        if self.group_by is None:
            fig, ax = plt.subplots()

            x_data = data_dict[f"data_singular"]['x']
            y_data = data_dict[f"data_singular"]['y']
            model = data_dict[f"data_singular"]['model']

            plot_func = getattr(ax, plot_func.__name__)
            plot_func(model.predict(x_data), y_data, **kwargs)

            plt.tight_layout()
            plt.show()

        elif separate_graphs:
            for column_name in self.group_by:
                column_data_dict = data_dict[f"col_{column_name}"]

                key_number = len(column_data_dict)

                x_axe_range = math.ceil(math.sqrt(key_number))
                y_axe_range = math.ceil(key_number / x_axe_range)
                fig, ax = plt.subplots(x_axe_range, y_axe_range)

                for y in range(y_axe_range):
                    for x in range(x_axe_range):
                        count = y * x_axe_range + x

                        if count == key_number:
                            break

                        code = column_data_dict[count]
                        x_data = data_dict[f"data_{column_name}_{code}"]['x']
                        y_data = data_dict[f"data_{column_name}_{code}"]['y']
                        model = data_dict[f"data_singular"]['model']

                        plot_func = getattr(ax[x, y], plot_func.__name__)
                        plot_func(model.predict(x_data), y_data, **kwargs)
                        ax[x, y].set_title(column_data_dict[count])

                plt.legend()
                plt.tight_layout()
                plt.show()
        else:
            for column_name in self.group_by:
                column_data_dict = data_dict[f"col_{column_name}"]

                key_number = len(column_data_dict)

                fig, ax = plt.subplots()

                for code_ind in range(key_number):
                    code = column_data_dict[code_ind]
                    x_data = data_dict[f"data_{column_name}_{code}"]['x']
                    y_data = data_dict[f"data_{column_name}_{code}"]['y']
                    model = data_dict[f"data_singular"]['model']

                    plot_func = getattr(ax, plot_func.__name__)
                    plot_func(model.predict(x_data), y_data, **kwargs)
                    ax.set_title(code)
                plt.title(column_name)
                # plt.legend()
                plt.tight_layout()
                plt.show()

    def analyse_data(self, x, y, model):
        return_dict = {}

        if self.group_by is None:
            x = x[[column_name for column_name in x.columns
                   if column_name not in self.ignore]]

            model.fit(x, y)

            return_dict[f"data_singular"] = {
                "x": x,
                "y": y,
                "model": model
            }

        else:
            x = x.reset_index(drop=True)
            y = y.reset_index(drop=True)

            for column_name in self.group_by:
                unique_group_codes = x[column_name].unique().tolist()

                for code in unique_group_codes:
                    model = clone(model)
                    data_mask = x[column_name] == code

                    group_x = x.loc[data_mask]
                    group_y = y.loc[data_mask]

                    group_x = group_x[[column_name for column_name in group_x.columns
                                       if column_name not in self.group_by]]

                    group_x = group_x[[column_name for column_name in x.columns
                                       if column_name not in self.ignore]]

                    model.fit(group_x, group_y)

                    # print(f"{column_name}_{code}: " + str(np.max(model.coef_)))
                    return_dict[f"data_{column_name}_{code}"] = {
                        "x": group_x,
                        "y": group_y,
                        "model": model
                    }

                return_dict[f"col_{column_name}"] = unique_group_codes

        return return_dict

    def analyse_metadata(self, data_dict: dict, standardized=False, display=True, **kwargs):
        return_struct = {}
        if self.group_by is None:
            fig, ax = plt.subplots()

            x_data = data_dict[f"data_singular"]['x']
            y_data = data_dict[f"data_singular"]['y']
            model = data_dict[f"data_singular"]['model']
            feature_names = x_data.columns

            coefs = pd.DataFrame(
                model.coef_,
                columns=["Coefficients"],
                index=feature_names,
            )

            if standardized:
                coefs = pd.DataFrame(
                    model.coef_ * x_data.std(axis=0),
                    columns=["Coefficient importance"],
                    index=feature_names,
                )

            coefs = np.squeeze(coefs)
            if display:
                ax.barh(feature_names, coefs, **kwargs)

            return_struct['singular'] = coefs
        else:
            for column_name in self.group_by:
                column_data_dict = data_dict[f"col_{column_name}"]

                fig, ax = plt.subplots()

                for code in column_data_dict:
                    x_data = data_dict[f"data_{column_name}_{code}"]['x']
                    y_data = data_dict[f"data_{column_name}_{code}"]['y']
                    model = data_dict[f"data_{column_name}_{code}"]['model']
                    # print(f"{column_name}_{code}: " + str(np.max(model.coef_)))
                    feature_names = x_data.columns

                    coefs = pd.DataFrame(
                        model.coef_,
                        columns=["Coefficients"],
                        index=feature_names,
                    )

                    if standardized:
                        coefs = pd.DataFrame(
                            model.coef_ * x_data.std(axis=0),
                            columns=["Coefficient importance"],
                            index=feature_names,
                        )

                    coefs = np.squeeze(coefs)

                    if display:
                        ax.barh(feature_names, coefs, **kwargs)

                    return_struct[f'{column_name}_{code}'] = coefs

        if display:
            plt.xlabel("Coefficient values corrected by the feature's std. dev.")
            # plt.title("Ridge model, small regularization")
            plt.axvline(x=0, color=".5")
            plt.tight_layout()
            plt.show()

        plt.close()
        return return_struct


def plot_heatmap(data_2d, x_labels, y_labels, title):
    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(data_2d)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(x_labels)), labels=x_labels,
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(y_labels)), labels=y_labels)

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, str(np.round(data_2d[i, j], 2)),
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def get_data_path():
    path = os.path.abspath("C:/Users/ppp/Documents/Witek/Studia/Laby/Research Project/main project/data")
    return path
