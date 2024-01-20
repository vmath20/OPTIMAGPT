import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['ChatGPT-3.5', 'Bard', 'Bing']

precision_values = [0.950, 0.946, 0.958]
recall_values = [0.765, 0.921, 0.951]
f1_score_values = [0.848, 0.933, 0.955]
accuracy_values = [0.918, 0.960, 0.974]

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

bar_width = 0.2
bar_positions = np.arange(len(models))

# Grouped bar chart
ax.bar(bar_positions - bar_width, precision_values, bar_width, label='Precision', color='#FFB14E')
ax.bar(bar_positions, recall_values, bar_width, label='Recall/Sensitivity', color='#FA8775')
ax.bar(bar_positions + bar_width, f1_score_values, bar_width, label='F1 Score', color='#9D02D7')
ax.bar(bar_positions + 2 * bar_width, accuracy_values, bar_width, label='Accuracy', color='#5252E3')

# X-axis and labels
ax.set_xticks(bar_positions)
ax.set_xticklabels(models)
ax.set_xlabel('Models')

# Title and legend
plt.title('Performance Comparison LLM on Drug Label Extraction')
plt.legend(loc='upper left', bbox_to_anchor=(0.75, 0.2))

# Show the plot
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['ChatGPT-3.5', 'Bard', 'Bing']

precision_values = [0.848, 0.917, 0.991]
recall_values = [0.873, 0.991, 0.991]
f1_score_values = [0.860, 0.953, 0.991]
accuracy_values = [0.942, 0.978, 0.996]

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

bar_width = 0.2
bar_positions = np.arange(len(models))

# Grouped bar chart
ax.bar(bar_positions - bar_width, precision_values, bar_width, label='Precision', color='#FFB14E')
ax.bar(bar_positions, recall_values, bar_width, label='Recall/Sensitivity', color='#FA8775')
ax.bar(bar_positions + bar_width, f1_score_values, bar_width, label='F1 Score', color='#9D02D7')
ax.bar(bar_positions + 2 * bar_width, accuracy_values, bar_width, label='Accuracy', color='#5252E3')

# X-axis and labels
ax.set_xticks(bar_positions)
ax.set_xticklabels(models)
ax.set_xlabel('Models')

# Title and legend outside the graph
plt.title('Performance Comparison LLM on Laterality Label Extraction')
plt.legend(loc='upper left', bbox_to_anchor=(0.75, 0.2))

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['ChatGPT-3.5', 'Bard', 'Bing']

precision_values = [0.740, 0.944, 0.929]
recall_values = [0.755, 0.967, 0.983]
f1_score_values = [0.748, 0.955, 0.955]
accuracy_values = [0.895, 0.978, 0.978]

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

bar_width = 0.2
bar_positions = np.arange(len(models))

# Grouped bar chart
ax.bar(bar_positions - bar_width, precision_values, bar_width, label='Precision', color='#FFB14E')
ax.bar(bar_positions, recall_values, bar_width, label='Recall/Sensitivity', color='#FA8775')
ax.bar(bar_positions + bar_width, f1_score_values, bar_width, label='F1 Score', color='#9D02D7')
ax.bar(bar_positions + 2 * bar_width, accuracy_values, bar_width, label='Accuracy', color='#5252E3')

# X-axis and labels
ax.set_xticks(bar_positions)
ax.set_xticklabels(models)
ax.set_xlabel('Models')

# Title and legend outside the graph
plt.title('Performance Comparison LLM on Frequency Label Extraction')
plt.legend(loc='upper left', bbox_to_anchor=(0.75, 0.2))

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['BERT', 'RoBERTa', 'BioBERT', 'ClinicalBERT', 'DistilBERT', 'ChatGPT-3.5']
precision_drug = [1.00, 0.89, 1.00, 0.91, 1.00, 0.92]
recall_drug = [0.76, 0.76, 0.72, 0.73, 0.73, 0.81]
f1_drug = [0.86, 0.82, 0.84, 0.81, 0.85, 0.87]

precision_laterality = [0.77, 0.80, 0.79, 0.86, 0.78, 0.72]
recall_laterality = [0.76, 0.74, 0.84, 0.84, 0.83, 0.94]
f1_laterality = [0.76, 0.77, 0.82, 0.85, 0.80, 0.82]

# Set up the figure and axes
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot grouped bar charts for Drug Label
width = 0.2
x = np.arange(len(models))

ax[0].bar(x - width, precision_drug, width, label='Precision')
ax[0].bar(x, recall_drug, width, label='Recall')
ax[0].bar(x + width, f1_drug, width, label='F1 Score')

ax[0].set_title('Drug Label Performance Metrics')
ax[0].set_xticks(x)
ax[0].set_xticklabels(models)
ax[0].legend(loc='upper left', bbox_to_anchor=(0.85, 0.3))

# Plot grouped bar charts for Laterality Label
ax[1].bar(x - width, precision_laterality, width, label='Precision')
ax[1].bar(x, recall_laterality, width, label='Recall')
ax[1].bar(x + width, f1_laterality, width, label='F1 Score')

ax[1].set_title('Laterality Label Performance Metrics')
ax[1].set_xticks(x)
ax[1].set_xticklabels(models)
ax[1].legend(loc='upper left', bbox_to_anchor=(0.85, 0.3))

# Show the plots
plt.tight_layout()
plt.show()
