import matplotlib.pyplot as plt
import pandas as pd

# Your data (partial, as it's quite extensive)
data = {
    'Baseline': ['ANN2D', 'ANN2D', 'ANN3D', 'ANN3D', 'SNN3', 'SNN3', 'SNN2', 'SNN2'],
    'R^2': [0.593, 0.558, 0.642, 0.589, 0.642, 0.624, 0.591, 0.576],
    'Footprint (bytes)': [20824, 33496, 94552, 180952, 24972, 43020, 19648, 38848],
    'Dense SynOps': [4704, 7776, 23136, 44640, 32928, 54432, 4900, 9700],
    'Indy': ['Indy', 'Loco', 'Indy', 'Loco', 'Indy', 'Loco', 'Indy', 'Loco'],
    # ... Add other columns as needed
}

df = pd.DataFrame(data)

# Define colors for Indy/Loco and shapes for each model
colors = {'Indy': 'r', 'Loco': 'g'}
shapes = {'ANN2D': 'o', 'ANN3D': 's', 'SNN3': '^', 'SNN2': 'D'}

# Plot 1: Accuracy vs Footprint
fig, ax1 = plt.subplots()
for model, shape in shapes.items():
    for subject, color in colors.items():
        subset = df[(df['Baseline'] == model) & (df['Indy'] == subject)]
        ax1.scatter(subset['R^2'], subset['Footprint (bytes)'], color=color, marker=shape)

# Create a legend
legend_labels = [f'{color} {subject}' for subject, color in colors.items()]
legend_labels.extend([f'{shape} {model}' for model, shape in shapes.items()])
ax1.legend(legend_labels, loc='upper left')

ax1.set_xlabel('Accuracy (R^2)')
ax1.set_ylabel('Footprint (bytes)')
plt.title('Accuracy vs Footprint')

# Plot 2: Accuracy vs Dense Synaptic Operations
fig, ax2 = plt.subplots()
for model, shape in shapes.items():
    for subject, color in colors.items():
        subset = df[(df['Baseline'] == model) & (df['Indy'] == subject)]
        ax2.scatter(subset['R^2'], subset['Dense SynOps'], color=color, marker=shape)

# Create a legend
ax2.legend(legend_labels, loc='upper left')

ax2.set_xlabel('Accuracy (R^2)')
ax2.set_ylabel('Dense Synaptic Operations')
plt.title('Accuracy vs Dense Synaptic Operations')

plt.show()
