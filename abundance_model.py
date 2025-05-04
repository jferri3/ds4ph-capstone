import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import layers, models

# File upload section
otu_file = pd.read_excel('/users/jacquelineferri/16s_app/jhmi-esca-microbiota.counts.species-otu.xlsx')
alpha_file = pd.read_excel('/users/jacquelineferri/16s_app/jhmi-esca-microbiota.alpha-diversity.xlsx')
metad_file = pd.read_excel('/users/jacquelineferri/16s_app/meta.2022-11-07.xlsx')
pcoa_file = pd.read_excel('/users/jacquelineferri/16s_app/bray-curtis.pcoa.1.1.xlsx')

# Set '#OTU_ID' as the index for otu_file
otu_file = otu_file.set_index('#OTU_ID')

print(otu_file.head(5))

print(metad_file.head(10))

# Ensure all columns in the DataFrame are numeric before summing
otu_file = otu_file.apply(pd.to_numeric, errors='coerce')

# Remove OTUs with total abundance below a threshold (e.g., 10)
otu_file = otu_file.loc[:, otu_file.sum(axis=0) > 10]

print(otu_file.head(5))

otu_file = otu_file.transpose()

# Check the first few rows to confirm the swap
print(otu_file.head())

# Calculate total abundance for each species
total_abundance = otu_file.sum(axis=1)

# Display the 10 most abundant species
most_abundant_species = total_abundance.nlargest(10)
print("10 Most Abundant Species:")
print(most_abundant_species)

# Display the 10 least abundant species
least_abundant_species = total_abundance.nsmallest(10)
print("\n10 Least Abundant Species:")
print(least_abundant_species)

print(metad_file.columns)

grouped = metad_file.groupby("CAP regression by central review")["SimpleID"].apply(list)
print(grouped)

for group, samples in grouped.items():
    print(f"Group: {group}")
    
    # Filter otu_file for the samples in the current group
    group_data = otu_file[samples]
    
    # Display the most abundant species for the group
    print("Most Abundant Species:")
    print(group_data.loc[most_abundant_species.index])
    
    # Display the least abundant species for the group
    print("\nLeast Abundant Species:")
    print(group_data.loc[least_abundant_species.index])
    
    print("\n" + "-" * 50 + "\n")
    
    # Plotting
    most_abundant_data = group_data.loc[most_abundant_species.index]
    most_abundant_data = most_abundant_data.transpose()
    
    # Normalize the data so that each sample's abundance sums to 1
    most_abundant_data_normalized = most_abundant_data.div(most_abundant_data.sum(axis=1), axis=0)
    
    # Adjust plot size and legend
    most_abundant_data_normalized.plot(kind='bar', stacked=True, figsize=(16, 8), title=f"Most Abundant Species in Group {group} (Normalized)")
    plt.xlabel("SampleID")
    plt.ylabel("Normalized Abundance")
    plt.legend(title="Species", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

# Combine most abundant and least abundant species into a single DataFrame
abundant_species = pd.concat([most_abundant_species, least_abundant_species], axis=0)
abundant_species_data = otu_file.loc[abundant_species.index].transpose()

# Ensure the target variable is numeric
label_encoder = LabelEncoder()
metad_file["CAP regression by central review"] = label_encoder.fit_transform(
    metad_file["CAP regression by central review"].astype(str)
)

# Align the samples between the species data and metadata
common_samples = abundant_species_data.index.intersection(metad_file.index)
X = abundant_species_data.loc[common_samples]
y = metad_file.loc[common_samples, "CAP regression by central review"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot testing and training data linearly
plt.figure(figsize=(12, 6))
plt.plot(y_train.values, label='Training Data', linestyle='-', marker='o', alpha=0.7)
plt.plot(y_test.values, label='Testing Data', linestyle='--', marker='x', alpha=0.7)
plt.title('Training and Testing Data (Linear Plot)')
plt.xlabel('Index')
plt.ylabel('CAP Regression by Central Review')
plt.legend()
plt.grid(True)
plt.show()

# Plot testing and training data as bar plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Bar plot for training data
axes[0].bar(range(len(y_train)), y_train.values, color='blue', alpha=0.7)
axes[0].set_title('Training Data (Bar Plot)')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('CAP Regression by Central Review')

# Bar plot for testing data
axes[1].bar(range(len(y_test)), y_test.values, color='orange', alpha=0.7)
axes[1].set_title('Testing Data (Bar Plot)')
axes[1].set_xlabel('Index')

plt.tight_layout()
plt.show()

# Normalize the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the data for CNN (e.g., as a 2D "image" with 1 channel)
X_cnn = X_scaled.reshape(X_scaled.shape[0],