"""
import pandas as pd

# Load the CSV file
df = pd.read_csv('pred_coords_35_mod.csv')

# Get the second column
first_column = df.iloc[2:, 1].astype(float)
first_column_mod =first_column + 500


# Find the first free column at the end of the file
first_free_column = len(df.columns)

# Insert the second column into the first free column
df.insert(first_free_column, f'column_{first_free_column}', first_column_mod)

# Save the DataFrame back to the CSV file
df.to_csv('pred_coords_35_mod.csv', index=False)
"""

###########################################

import pandas as pd

# Load the CSV file
df = pd.read_csv('pred_coords_35_mod.csv')

# Get the second column
second_column = df.iloc[2:, 2]#.astype(float)

#second_column_mod = second_column +500

# Find the first free column at the end of the file
first_free_column = len(df.columns)

# Insert the second column into the first free column
df.insert(first_free_column, f'column_{first_free_column}', second_column)

# Save the DataFrame back to the CSV file
df.to_csv('pred_coords_35_mod.csv', index=False)

######################################################à

"""
import pandas as pd

# Load the CSV file
df = pd.read_csv('pred_coords_35_mod.csv')

# Delete the nth column
n = 17  # Replace with the index of the column you want to delete
df.drop(df.columns[n-1], axis=1, inplace=True)

# Save the DataFrame back to the CSV file
df.to_csv('pred_coords_35_mod.csv', index=False)
"""