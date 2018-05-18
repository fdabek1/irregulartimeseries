import matplotlib.pyplot as plt
import pandas as pd
import os

RESULTS_PATH = 'results/'

# Get all of the result files for the notebooks
results = []
for root, dirs, files in os.walk(RESULTS_PATH):
    for file in files:
        if file.endswith('.csv'):  # Must be a CSV
            results.append(file)

# Loop through the results
for result in results:
    # Get the title of the notebook
    title = result.replace('.csv', '')

    df = pd.read_csv('results/' + result)

    df = df[df['model'] != 'SGD']
    df = df[df['model'] != 'RNNMultiple']
    df = df[df['model'] != 'RNNAll']

    df.plot(x='model', y=['train', 'test'], kind='bar', title=result, figsize=(15, 10), legend=True, fontsize=12)
    plt.ylabel('MAE Loss (Mean Average Error)')
    plt.show()
