import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import layers, models
import streamlit as st

# File upload section
# Streamlit file uploader in the sidebar
st.sidebar.title("Upload Your Files")
uploaded_files = {
    "OTU File": st.sidebar.file_uploader("Upload OTU File", type=["xlsx"]),
    "Alpha Diversity File": st.sidebar.file_uploader("Upload Alpha Diversity File", type=["xlsx"]),
    "Metadata File": st.sidebar.file_uploader("Upload Metadata File", type=["xlsx"]),
    "PCoA File": st.sidebar.file_uploader("Upload PCoA File", type=["xlsx"])
}

# Check if all files are uploaded
if all(uploaded_files.values()):
    # Read the uploaded files into pandas DataFrames
    otu_file = pd.read_excel(uploaded_files["OTU File"])
    alpha_file = pd.read_excel(uploaded_files["Alpha Diversity File"])
    metad_file = pd.read_excel(uploaded_files["Metadata File"])
    pcoa_file = pd.read_excel(uploaded_files["PCoA File"])

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
else:
    st.sidebar.warning("Please upload all required files to proceed.")