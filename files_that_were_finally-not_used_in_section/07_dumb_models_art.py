import pandas as pd
import numpy as np
import pyreadstat
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# assumption that files generated in step 5 are 



df = pd.read_csv("INT_Final_Merged_Prefixed.csv")
df_st_filter = pd.read_csv("data_full_data.csv")
df = df[df['Username_Std'].isin(df_st_filter['Username_Std'])]


df['y'] = df['ST_Sgrade_Arts'] # for initial trial let's take mean 


df['MAX_PARENT_Occupation'] = df[['PA_MISEI_PA','PA_FISEI_PA']].max(axis=1)  # max from quality score of pparent occupation 
df['MAX_PARENT_Education'] = df[['PA_PAQM00601','PA_PAQM00602']].max(axis = 1)  # max from education of parent 


# to make case as simple as possible for start we will remove rows containing any na in newly craftet features columns 

data_together = df[['MAX_PARENT_Occupation','MAX_PARENT_Education','ST_PER_WLE_ADJ','ST_CUR_WLE_ADJ','y']]
data_together = data_together.dropna()



#  data for socioeconomic model 

data_socio = data_together[['MAX_PARENT_Occupation','MAX_PARENT_Education','y']]

# data for skill mdoel 

data_skill = data_together[['ST_PER_WLE_ADJ','ST_CUR_WLE_ADJ','y']]

# --- Standardization and Linear Regression ---

# 1. Socioeconomic Model
print("--- Socioeconomic Model ---")
X_socio = data_socio[['MAX_PARENT_Occupation', 'MAX_PARENT_Education']]
y_socio = data_socio['y']

scaler_socio = StandardScaler()
X_socio_std = scaler_socio.fit_transform(X_socio)

model_socio = LinearRegression()
model_socio.fit(X_socio_std, y_socio)

print("Coefficients:", model_socio.coef_)
print("Intercept:", model_socio.intercept_)
print("R^2:", model_socio.score(X_socio_std, y_socio))

# 2. Skill Model
print("\n--- Skill Model ---")
X_skill = data_skill[['ST_PER_WLE_ADJ', 'ST_CUR_WLE_ADJ']]
y_skill = data_skill['y']

scaler_skill = StandardScaler()
X_skill_std = scaler_skill.fit_transform(X_skill)

model_skill = LinearRegression()
model_skill.fit(X_skill_std, y_skill)

print("Coefficients:", model_skill.coef_)
print("Intercept:", model_skill.intercept_)
print("R^2:", model_skill.score(X_skill_std, y_skill))


# --- Visualizations ---
sns.set_theme(style="whitegrid")

# Visualization 1: Coefficients Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Socio Coefficients
sns.barplot(x=X_socio.columns, y=model_socio.coef_, ax=axes[0], hue=X_socio.columns, legend=False, palette="viridis")
axes[0].set_title('Impact of Socioeconomic Factors (Standardized) - Art Score')
axes[0].set_ylabel('Coefficient Value (Effect Size)')
axes[0].set_xlabel('Features')
axes[0].tick_params(axis='x', rotation=15)

# Skill Coefficients
sns.barplot(x=X_skill.columns, y=model_skill.coef_, ax=axes[1], hue=X_skill.columns, legend=False, palette="magma")
axes[1].set_title('Impact of Skill Factors (Standardized) - Art Score')
axes[1].set_ylabel('Coefficient Value (Effect Size)')
axes[1].set_xlabel('Features')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('07_coefficients_comparison_art.jpg', dpi=300)
plt.show()

# Visualization 2: Actual vs Predicted
y_pred_socio = model_socio.predict(X_socio_std)
y_pred_skill = model_skill.predict(X_skill_std)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Socio Actual vs Predicted
sns.scatterplot(x=y_socio, y=y_pred_socio, ax=axes[0], alpha=0.3, color='blue')
axes[0].plot([y_socio.min(), y_socio.max()], [y_socio.min(), y_socio.max()], 'r--', lw=2) # Perfect prediction line
axes[0].set_title(f'Socio Model: Actual vs Predicted (Art Score) (R^2: {model_socio.score(X_socio_std, y_socio):.2f})')
axes[0].set_xlabel('Actual Grades')
axes[0].set_ylabel('Predicted Grades')

# Skill Actual vs Predicted
sns.scatterplot(x=y_skill, y=y_pred_skill, ax=axes[1], alpha=0.3, color='green')
axes[1].plot([y_skill.min(), y_skill.max()], [y_skill.min(), y_skill.max()], 'r--', lw=2)
axes[1].set_title(f'Skill Model: Actual vs Predicted (Art Score) (R^2: {model_skill.score(X_skill_std, y_skill):.2f})')
axes[1].set_xlabel('Actual Grades')
axes[1].set_ylabel('Predicted Grades')

plt.tight_layout()
plt.savefig('07_actual_vs_predicted_art.jpg', dpi=300)
plt.show()

# Visualization 3: 3D Plots (Features vs Target + Prediction Plane)
def plot_3d_model(X, y, model, title, filename):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot actual data points
    # We use the first two columns of X (which are the features)
    # X is expected to be a numpy array (standardized features)
    xs = X[:, 0]
    ys = X[:, 1]
    zs = y
    
    ax.scatter(xs, ys, zs, c='blue', marker='o', alpha=0.1, label='Actual Data')

    # Create a meshgrid for the prediction plane
    x_surf = np.linspace(xs.min(), xs.max(), 20)
    y_surf = np.linspace(ys.min(), ys.max(), 20)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    
    # Predict z values for the meshgrid
    # We need to flatten the meshgrid arrays to pass them to the model
    X_mesh = np.array([x_surf.flatten(), y_surf.flatten()]).T
    
    z_surf = model.predict(X_mesh)
    z_surf = z_surf.reshape(x_surf.shape)

    # Plot the prediction plane
    ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3, label='Prediction Plane')

    ax.set_xlabel('Feature 1 (Std)')
    ax.set_ylabel('Feature 2 (Std)')
    ax.set_zlabel('Target (y)')
    ax.set_title(title)
    
    # Add a legend (proxy artists needed for surface)
    import matplotlib.lines as mlines
    blue_proxy = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=5, label='Actual Data')
    red_proxy = mlines.Line2D([], [], color='red', alpha=0.5, linewidth=2, label='Prediction Plane')
    ax.legend(handles=[blue_proxy, red_proxy])

    plt.savefig(filename, dpi=300)
    plt.show()

# Plot 3D for Socio Model
plot_3d_model(X_socio_std, y_socio, model_socio, 'Socio Model: 3D Features vs Target (Art Score)', '07_socio_3d_plot_art.jpg')

# Plot 3D for Skill Model
plot_3d_model(X_skill_std, y_skill, model_skill, 'Skill Model: 3D Features vs Target (Art Score)', '07_skill_3d_plot_art.jpg')


