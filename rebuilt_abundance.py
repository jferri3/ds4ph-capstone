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

    # Transpose otu_file and set #OTU_ID as the index
    otu_file.set_index("#OTU_ID", inplace=True)
    otu_file = otu_file.transpose()

    # Ensure the index of otu_file matches the SimpleID column in metad_file
    otu_file.index.name = "SimpleID"
    merged_data = otu_file.join(metad_file.set_index("SimpleID"), how="inner")

    # Calculate total species abundance for each sample
    # Select only numeric columns for summation
    numeric_columns = merged_data.select_dtypes(include=[np.number])
    total_species_abundance = numeric_columns.sum(axis=1)

    # Display the result
    st.write("Total Species Abundance:")
    st.dataframe(total_species_abundance)

    # Group SimpleIDs based on "CAP regression by central review" column
    grouped = metad_file.groupby("CAP regression by central review").groups

    # Extract SimpleIDs for each group
    G0 = grouped.get("G0", [])
    G1 = grouped.get("G1", [])
    G2 = grouped.get("G2", [])
    G3 = grouped.get("G3", [])

    # Display the grouped SimpleIDs
    st.write("Grouped SimpleIDs:")
    st.write({"G0": G0, "G1": G1, "G2": G2, "G3": G3})

    # Add an option for "Abundance Analysis" in the Streamlit app
    st.sidebar.title("Analysis Options")
    analysis_option = st.sidebar.selectbox("Choose Analysis", ["None", "Abundance Analysis"])

    if analysis_option == "Abundance Analysis":
        # Select CAP regression group
        selected_group = st.sidebar.selectbox("Select CAP Regression Group", ["G0", "G1", "G2", "G3"])
        
        # Get the corresponding SimpleIDs for the selected group
        selected_simple_ids = grouped.get(selected_group, [])
        
        if selected_simple_ids:
            # Filter the merged data for the selected SimpleIDs
            group_data = merged_data.loc[selected_simple_ids]
            
            # Calculate the total abundance for each species
            species_abundance = group_data.sum(axis=0).sort_values(ascending=False)
            
            # Get the top 10 most abundant species
            top_10_species = species_abundance.head(10)
            
            # Normalize the data for the top 10 species
            normalized_data = group_data[top_10_species.index].div(group_data[top_10_species.index].sum(axis=1), axis=0)
            
            # Display the top 10 most abundant species in a table
            st.write(f"Top 10 Most Abundant Species for {selected_group}:")
            st.dataframe(top_10_species)
            
            # Plot a stacked bar chart for the normalized data
            st.write("Normalized Abundance (Stacked Bar Chart):")
            fig, ax = plt.subplots(figsize=(10, 6))
            normalized_data.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
            ax.set_title(f"Normalized Abundance for {selected_group}")
            ax.set_ylabel("Proportion")
            ax.set_xlabel("Sample ID")
            st.pyplot(fig)
        else:
            st.write(f"No data available for group {selected_group}.")