import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import layers, models
import io  # Import io module

# Streamlit file uploader in the sidebar
st.sidebar.title("Upload Your Files")
otu_file_raw = st.sidebar.file_uploader("Upload OTU File", type=["xlsx", "csv"])
metadata_file_raw = st.sidebar.file_uploader("Upload Metadata File", type=["xlsx", "csv"])

# Set the title of the Streamlit app
st.title("Microbial Analysis Dashboard")
st.subheader("Jacqueline Ferri's DS4PH capstone project")
st.write("May 8th, 2025")
st.write("Here you can upload files for analysis of 16s sequencing data.")

def load_data(otu_file_content, metadata_file_content):
    """Loads and preprocesses OTU and metadata, handling potential errors."""
    if otu_file_content is None or metadata_file_content is None:
        return None, None  # Return Nones if files not uploaded

    try:
        # Read the files using pandas, handling both xlsx and csv
        otu_file = pd.read_excel(otu_file_content) if otu_file_content.name.endswith("xlsx") else pd.read_csv(io.BytesIO(otu_file_content.read()))
        metadata_file = pd.read_excel(metadata_file_content) if metadata_file_content.name.endswith("xlsx") else pd.read_csv(io.BytesIO(metadata_file_content.read()))


        # Transpose otu_file and set #OTU_ID as the index
        otu_file.set_index("#OTU_ID", inplace=True)
        otu_file = otu_file.transpose()
        otu_file.index.name = "SimpleID"  # Set index name *before* the join
        st.write("Transposed OTU File:") #prints the otu_file
        st.dataframe(otu_file)

        return otu_file, metadata_file

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None, None

def perform_analysis(otu_file, metad_file):  # Changed function name
    """Performs the analysis and displays results, handling missing data."""
    # Check if dataframes are valid
    if otu_file is None or metad_file is None:
        st.warning("Please upload both OTU and Metadata files.")
        return

    # Ensure the index of otu_file matches the SimpleID column in metad_file
    merged_data = otu_file.join(metad_file.set_index("SimpleID"), how="inner")

    if merged_data.empty:
        st.error("Error: No matching samples found between OTU and Metadata files. Please check that the SimpleID column exists and contains matching values.")
        return

    # Calculate total species abundance for each sample
    numeric_columns = merged_data.select_dtypes(include=[np.number])
    total_species_abundance = numeric_columns.sum(axis=1)

    # Group SimpleIDs based on "CAP regression by central review" column
    if "CAP regression by central review" not in metad_file.columns:
        st.error("Error: The Metadata file must contain a column named 'CAP regression by central review'.")
        return

    grouped = metad_file.groupby("CAP regression by central review").groups
    
    # Display the grouped data in the Streamlit app
    st.write("Grouped Data by 'CAP regression by central review':")
    st.write(grouped)

    # Display the 10 most abundant species
    most_abundant_species = total_species_abundance.nlargest(10)

    # Add an option for "Abundance Analysis" in the Streamlit app
    st.subheader("Analysis Options")
    analysis_option = st.selectbox("Choose Analysis", ["None", "Abundance Analysis"])

    if analysis_option == "Abundance Analysis":
        # Select CAP regression group
        cap_groups = list(grouped.keys())  # Get group names dynamically
        if not cap_groups:
            st.warning("No groups found in 'CAP regression by central review' column.")
            return

        selected_group = st.selectbox("Select CAP Regression Group", cap_groups)

        # Get the corresponding SimpleIDs for the selected group
        selected_simple_ids = grouped[selected_group]  # Access directly from the dict

        if not selected_simple_ids.empty: # corrected this line
            st.warning(f"No data available for group {selected_group}.")
        else:
            try:
                # Filter the merged data for the selected SimpleIDs
                group_data = merged_data.loc[selected_simple_ids]  # Use .loc
                # Iterate through each group (G0, G1, G2, G3)
                for group_name in cap_groups: #changed from hardcoded
                    # Get the corresponding SimpleIDs for the group
                    group_simple_ids = grouped.get(group_name, [])

                    if group_simple_ids:
                        # Filter the merged data for the group's SimpleIDs
                        group_data_for_plot = merged_data.loc[group_simple_ids]

                        # Calculate the total abundance for each species
                        species_abundance = group_data_for_plot.sum(axis=0).sort_values(ascending=False)

                        # Get the top 10 most abundant species
                        top_10_species = species_abundance.head(10)

                        # Normalize the data for the top 10 species
                        normalized_data = group_data_for_plot[top_10_species.index].div(
                            group_data_for_plot[top_10_species.index].sum(axis=1), axis=0
                        )

                        # Display the top 10 most abundant species in a table
                        st.write(f"Top 10 Most Abundant Species for {group_name}:")
                        st.dataframe(top_10_species)

                        # Plot a stacked bar chart for the normalized data
                        st.write(f"Normalized Abundance (Stacked Bar Chart) for {group_name}:")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        normalized_data.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
                        ax.set_title(f"Normalized Abundance for {group_name}")
                        ax.set_ylabel("Proportion")
                        ax.set_xlabel("Sample ID")
                        st.pyplot(fig)
                    else:
                        st.write(f"No data available for group {group_name}.")
            except KeyError as e:
                st.error(f"Error: {e}.  Check if sample IDs in metadata match OTU table index.\")\n\ndef main():\n    \"\"\"Main function to run the Streamlit app.\"\"\"\n    otu_file_content = st.session_state.get(\'otu_file_uploader\')\n    metadata_file_content = st.session_state.get(\'metadata_file_uploader\')\n    otu_file, metad_file = load_data(otu_file_content, metadata_file_content)\n    perform_analysis(otu_file, metad_file) # Moved the function call\n\nif __name__ == \"__main__\":\n    main()
