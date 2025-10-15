import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import ast

# data summary

import pandas as pd
import numpy as np
import ast
# Load the CSV file (replace with your file path)
import pandas as pd
import numpy as np
import ast

# Load the CSV file (replace with your file path)
file_path = '/Users/ivy/Desktop/spot_gesture_eval/0926/merged_output.csv'
df = pd.read_csv(file_path)

# Extract target distance (assuming z-distance is the third value in the target_location list)
df['target_distance'] = df['target_location'].apply(lambda x: ast.literal_eval(x)[2])

# Define a function to remove outliers using the IQR method based on 'gesture_duration'
def remove_outliers_iqr(group):
    Q1 = group['gesture_duration'].quantile(0.25)
    Q3 = group['gesture_duration'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group['gesture_duration'] >= lower_bound) & (group['gesture_duration'] <= upper_bound)]

# Apply the IQR-based outlier removal for each pointing count
df_filtered = df.groupby('pointing_count').apply(remove_outliers_iqr)

# Drop NaN values that may have been introduced during filtering
df_filtered = df_filtered.dropna()

# Group the filtered data by target distance and calculate mean, min, max, and std for each metric
grouped_df = df_filtered.groupby('target_distance').agg({
    'xz_distance_to_target(ground)': ['mean', 'min', 'max', 'std'],
    'xy_distance_to_target': ['mean', 'min', 'max', 'std'], 
    'yz_distance_to_target': ['mean', 'min', 'max', 'std'],  # Assuming xy_distance is stored as 'xy_distance_to_target'
    'angle_to_target': ['mean', 'min', 'max', 'std'],  # Assuming angle is stored as 'angle_to_target'

}).reset_index()

# Flatten the multi-level column names for readability
grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]

# Display the resulting grouped table
print(grouped_df)

# Save the resulting table to a CSV file (optional)
output_file = '/Users/ivy/Desktop/spot_gesture_eval/0926/Table Summary.csv'
grouped_df.to_csv(output_file, index=False)

print(f"Table saved to {output_file}")
# Load the CSV file (replace with your file path)
file_path = '/Users/ivy/Desktop/spot_gesture_eval/0926/Table Summary.csv'
df = pd.read_csv(file_path)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# Load the CSV file (replace with your file path)
file_path = '/Users/ivy/Desktop/spot_gesture_eval/0926/merged_output.csv'
df = pd.read_csv(file_path)

# Extract the target z-distance (or meters away) from the 'target_location' column
df['target_distance_m'] = df['target_location'].apply(lambda x: ast.literal_eval(x)[2])  # Extracting z-distance

# Group the data by target distance (e.g., 2m, 3m, 4m)
df_grouped = df.groupby('target_distance_m')

# Create a list to store the melted DataFrame for each group
melted_data = []

# Loop over each unique target distance
for target_distance, group in df_grouped:
    # Select the relevant distance columns (adjust column names if necessary)
    data_to_plot = group[['xz_distance_to_target(ground)', 'yz_distance_to_target', 'xy_distance_to_target']]
    
    # Melt the DataFrame to transform it into a long format for plotting
    melted_df = pd.melt(data_to_plot, var_name="Distance_Type", value_name="Distance")
    
    # Add the target distance as a column for labeling in the plot
    melted_df['Target Distance (m)'] = f"{target_distance} meters"
    
    # Append the melted DataFrame to the list
    melted_data.append(melted_df)

# Concatenate all the melted data for plotting
final_data = pd.concat(melted_data)

# Create the violin plot using seaborn
plt.figure(figsize=(12, 8))
sns.violinplot(x='Distance_Type', y='Distance', hue='Target Distance (m)', data=final_data, split=True)

# Add labels and title
plt.title('Violin Plot of Distances Grouped by Target Distance')
plt.xlabel('Distance Type')
plt.ylabel('Distance to Target')

# Show the plot
plt.grid(True)
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file (replace with your file path)
file_path = '/Users/ivy/Desktop/spot_gesture_eval/0926/merged_output.csv'
df = pd.read_csv(file_path)

# Function to remove outliers using the IQR method
# Function to remove outliers using the IQR method based on 'gesture_duration'
def remove_outliers_iqr(group):
    Q1 = group['gesture_duration'].quantile(0.25)
    Q3 = group['gesture_duration'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group['gesture_duration'] >= lower_bound) & (group['gesture_duration'] <= upper_bound)]


# Filter out outliers for each 'pointing_count' group
# df_filtered = df.groupby('pointing_count').apply(lambda group: group.apply(remove_outliers_iqr))
df_filtered = df.groupby('pointing_count').apply(remove_outliers_iqr)
# Drop any remaining NaN values that may have been introduced during filtering
df_filtered = df_filtered.dropna()

# Assuming your data has columns like 'xz_distance', 'yz_distance_to_target', 'xy_distance_to_target'
# Select the relevant columns for plotting
data_to_plot = df_filtered[['xz_distance_to_target(ground)', 'yz_distance_to_target', 'xy_distance_to_target']]
# data_to_plot = df_filtered[['angle_to_target', 'angles_eye_to_wrist',	'angles_shoulder_to_wrist',	'angles_elbow_to_wrist',	'angles_nose_to_wrist']]

# Create a new DataFrame to transform the data into a long format for seaborn
melted_df = pd.melt(data_to_plot, var_name="Distance_Type", value_name="Distance")

# Create the violin plot using seaborn
plt.figure(figsize=(10, 6))
sns.violinplot(x='Distance_Type', y='Distance', data=melted_df)

# Add labels and title
plt.title('Violin Plot of Distances')
plt.xlabel('Distance Type')
plt.ylabel('Angles[deg]')

# Show the plot
plt.grid(True)
plt.show()


# Fig: line plot to see how parameter changes over time for each unique pointing

# Define the number of points to interpolate to (e.g., 50 points for all curves)
num_points = 50

# Filter the data to include only specific pointing counts
# df = df[df['pointing_count'] < 13]

# Define a color map for different target z distances
color_map = {2: 'brown', 3: 'orange', 4: 'green', 5: 'yellow'}

# Keep track of which labels have been added to avoid duplicates
added_labels = set()

# Create a figure for plotting
plt.figure(figsize=(10, 6))

for count in df['pointing_count'].unique():
    # Filter data for the current pointing count
    subset = df[df['pointing_count'] == count]

    # Get frame and min_angle_to_target values
    x = subset['frame'].values
    y = subset['angle_to_target'].values

    # Normalize the curve using interpolation to have 'num_points' points
    interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")
    x_new = np.linspace(x.min(), x.max(), num_points)  # Create 'num_points' equally spaced points
    y_new = interp_func(x_new)  # Interpolate the y values

    # Extract target z distance from 'target_location'
    target_distance = int(ast.literal_eval(subset['target_location'].values[0])[2])
    
    # Check if we already added a label for this target distance
    label = f'Target Z Distance: {target_distance}' if target_distance not in added_labels else ""
    added_labels.add(target_distance)  # Add the distance to the set

    # Plot the normalized curve with color based on target distance
    plt.plot(y_new, color=color_map[target_distance], label=label)
    plt.plot(y_new.argmin(),y_new.min(), marker='o',color=color_map[target_distance])

# plt.xlabel('Normalized Frame')
plt.ylim([0, 120])
plt.ylabel('Angle to Target(deg)')
plt.title('Min angle across all vectors')

# Add the legend, ensuring only one label per distance is shown
plt.legend()

plt.grid(True)
plt.show()

# Fig: line plot to see how parameter changes over time for each unique pointing for each vector
plt.figure(figsize=(10, 6))
plt.subplot(221)
# Iterate through each unique pointing count
for count in df['pointing_count'].unique():
    # Filter data for the current pointing count
    subset = df[df['pointing_count'] == count]

    # Get frame and min_angle_to_target values
    x = subset['frame'].values
    y = subset['angles_shoulder_to_wrist'].values

    # Normalize the curve using interpolation to have 'num_points' points
    interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")
    x_new = np.linspace(x.min(), x.max(), num_points)  # Create 'num_points' equally spaced points
    y_new = interp_func(x_new)  # Interpolate the y values

    # Extract target z distance from 'target_location'
    target_distance = int(ast.literal_eval(subset['target_location'].values[0])[2])
    
    # Check if we already added a label for this target distance
    label = f'Target Z Distance: {target_distance}' if target_distance not in added_labels else ""
    added_labels.add(target_distance)  # Add the distance to the set

    # Plot the normalized curve with color based on target distance
    plt.plot(y_new, color=color_map[target_distance], label=label)
    plt.plot(y_new.argmin(),y_new.min(), marker='o',color=color_map[target_distance])

# plt.xlabel('Normalized Frame')
plt.ylim([0, 120])
plt.ylabel('Angle to Target')
plt.title('shoulder-to-wrist angle')

# Add the legend, ensuring only one label per distance is shown
plt.legend()

plt.grid(True)

plt.subplot(223)
# Iterate through each unique pointing count
for count in df['pointing_count'].unique():
    # Filter data for the current pointing count
    subset = df[df['pointing_count'] == count]

    # Get frame and min_angle_to_target values
    x = subset['frame'].values
    y = subset['angles_elbow_to_wrist'].values

    # Normalize the curve using interpolation to have 'num_points' points
    interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")
    x_new = np.linspace(x.min(), x.max(), num_points)  # Create 'num_points' equally spaced points
    y_new = interp_func(x_new)  # Interpolate the y values

    # Extract target z distance from 'target_location'
    target_distance = int(ast.literal_eval(subset['target_location'].values[0])[2])
    
    # Check if we already added a label for this target distance
    label = f'Target Z Distance: {target_distance}' if target_distance not in added_labels else ""
    added_labels.add(target_distance)  # Add the distance to the set

    # Plot the normalized curve with color based on target distance
    plt.plot(y_new, color=color_map[target_distance], label=label)
    plt.plot(y_new.argmin(),y_new.min(), marker='o',color=color_map[target_distance])
    
# Add labels and title
# plt.xlabel('Normalized Frame')
plt.ylim([0, 120])
plt.ylabel('Angle to Target')
plt.title('elbow-to-wrist angle')

# Add the legend, ensuring only one label per distance is shown
plt.legend()

plt.grid(True)

plt.subplot(224)
# Iterate through each unique pointing count
for count in df['pointing_count'].unique():
    # Filter data for the current pointing count
    subset = df[df['pointing_count'] == count]

    # Get frame and min_angle_to_target values
    x = subset['frame'].values
    y = subset['angles_eye_to_wrist'].values

    # Normalize the curve using interpolation to have 'num_points' points
    interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")
    x_new = np.linspace(x.min(), x.max(), num_points)  # Create 'num_points' equally spaced points
    y_new = interp_func(x_new)  # Interpolate the y values

    # Extract target z distance from 'target_location'
    target_distance = int(ast.literal_eval(subset['target_location'].values[0])[2])
    
    # Check if we already added a label for this target distance
    label = f'Target Z Distance: {target_distance}' if target_distance not in added_labels else ""
    added_labels.add(target_distance)  # Add the distance to the set

    # Plot the normalized curve with color based on target distance
    plt.plot(y_new, color=color_map[target_distance], label=label)
    plt.plot(y_new.argmin(),y_new.min(), marker='o',color=color_map[target_distance])
    
# Add labels and title
# plt.xlabel('Normalized Frame')
plt.ylim([0, 120])
plt.ylabel('Angle to Target')
plt.title('eye-to-wrist angle')

# Add the legend, ensuring only one label per distance is shown
plt.legend()

plt.grid(True)

plt.subplot(222)
# Iterate through each unique pointing count
for count in df['pointing_count'].unique():
    # Filter data for the current pointing count
    subset = df[df['pointing_count'] == count]

    # Get frame and min_angle_to_target values
    x = subset['frame'].values
    y = subset['angles_nose_to_wrist'].values

    # Normalize the curve using interpolation to have 'num_points' points
    interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")
    x_new = np.linspace(x.min(), x.max(), num_points)  # Create 'num_points' equally spaced points
    y_new = interp_func(x_new)  # Interpolate the y values

    # Extract target z distance from 'target_location'
    target_distance = int(ast.literal_eval(subset['target_location'].values[0])[2])
    
    # Check if we already added a label for this target distance
    label = f'Target Z Distance: {target_distance}' if target_distance not in added_labels else ""
    added_labels.add(target_distance)  # Add the distance to the set

    # Plot the normalized curve with color based on target distance
    plt.plot(y_new, color=color_map[target_distance], label=label)
    plt.plot(y_new.argmin(),y_new.min(), marker='o',color=color_map[target_distance])
    
# Add labels and title
# plt.xlabel('Normalized Frame')
plt.ylim([0, 120])
plt.ylabel('Angle to Target')
plt.title('nose-to-wrist angle')

# Add the legend, ensuring only one label per distance is shown
plt.legend()

plt.grid(True)
# Show the plot
plt.show()

