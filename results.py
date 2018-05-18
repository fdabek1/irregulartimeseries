import pandas as pd
import os

# Get all of the result files for the notebooks
results = []
for root, dirs, files in os.walk('results/'):
    for file in files:
        if file.endswith('.csv'):  # Must be a CSV
            results.append(file)

# Loop through the results
for result in results:
    # Get the title of the notebook
    title = result.replace('.csv', '')

    df = pd.read_csv('results/' + result)

    print('blah')
