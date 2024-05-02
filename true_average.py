import pandas as pd

# Read the CSV file
df = pd.read_csv(r'PATH TO THE CSV FILE (chamber_time.csv)')
# Initialize an empty DataFrame to store the merged rows
merged_df = pd.DataFrame(columns=df.columns)

# Iterate over the rows of the DataFrame
prev_row = df.iloc[0]
for i in range(1, len(df)):
    current_row = df.iloc[i]
    # If the end time of the previous row is equal to the start time of the current row
    # and the 'Chamber' of the previous row is the same as the 'Chamber' of the current row
    if prev_row['End Time'] == current_row['Start Time'] and prev_row['Chamber'] == current_row['Chamber']:
        # Merge the rows by summing the times
        prev_row['End Time'] = current_row['End Time']
        prev_row['Average Time'] += current_row['Average Time']
    else:
        # Add the previous row to the new DataFrame
        merged_df = merged_df.append(prev_row)
        prev_row = current_row

# Add the last row to the new DataFrame
merged_df = merged_df.append(prev_row)

# Write the new DataFrame to a CSV file
merged_df.to_csv(r'PATH WHERE YOU WANT THE NEW CSV FILE', index=False)