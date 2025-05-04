import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Correct placement

def load_data():
    """
    Loads and processes the OTU and metadata files, handling file uploads with caching.
    """
    otu_file = st.sidebar.file_uploader("Upload OTU File", type=["csv", "xlsx"], key="otu_file_uploader")
    metad_file = st.sidebar.file_uploader("Upload Metadata File", type=["csv", "xlsx"], key="metadata_file_uploader")

    if otu_file is not None and metad_file is not None:
        try:
            otu_df = pd.read_csv(otu_file) if otu_file.name.endswith(".csv") else pd.read_excel(otu_file)
            metad_df = pd.read_csv(metad_file) if metad_file.name.endswith(".csv") else pd.read_excel(metad_file)
            return otu_df, metad_df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None
    else:
        return None, None

@st.cache_data
def calculate_abundance(otu_df):
    """
    Calculates and displays the most and least abundant species.  Cached for performance.
    """
    if otu_df is None or otu_df.empty:
        return pd.Series(), pd.Series()

    total_abundance = otu_df.sum(axis=1)
    most_abundant_species = total_abundance.nlargest(10)
    least_abundant_species = total_abundance.nsmallest(10)
    return most_abundant_species, least_abundant_species

def group_analysis(otu_df, metad_df, most_abundant_species, least_abundant_species):
    """
    Performs group analysis and generates plots.
    """
    if otu_df is None or metad_df is None or otu_df.empty or metad_df.empty:
        st.warning("Please upload both OTU and Metadata files to perform group analysis.")
        return

    if "CAP regression by central review" not in metad_df.columns:
        st.error("Metadata file must contain the column 'CAP regression by central review'")
        return

    grouped = metad_df.groupby("CAP regression by central review")["SimpleID"].apply(list)

    for group, samples in grouped.items():
        st.subheader(f"Group: {group}")
        try: #added try and except
            # Filter otu_df for the samples in the current group
            group_data = otu_df[samples]

            # Display the most abundant species for the group
            st.write("Most Abundant Species:")
            st.write(group_data.loc[most_abundant_species.index])

            # Display the least abundant species for the group
            st.write("\nLeast Abundant Species:")
            st.write(group_data.loc[least_abundant_species.index])

            plot_group_abundance(group, samples, otu_df, most_abundant_species)
        except KeyError as e:
            st.error(f"Error processing group {group}: {e}.  Check if sample IDs in metadata match OTU table.")

def plot_group_abundance(group, samples, otu_df, most_abundant_species):
    """Generates the stacked bar plot for a given group."""
    try:
        # Filter otu_file for the samples in the current group
        group_data = otu_df[samples]
        # Filter for the most abundant species
        most_abundant_data = group_data.loc[most_abundant_species.index]
        # Transpose for plotting
        most_abundant_data_transposed = most_abundant_data.transpose()

        # Normalize the data so that each sample's abundance sums to 1
        most_abundant_data_normalized = most_abundant_data_transposed.div(most_abundant_data_transposed.sum(axis=1), axis=0)

        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 8))
        most_abundant_data_normalized.plot(kind='bar', stacked=True, ax=ax, title=f"Most Abundant Species in Group {group} (Normalized)")
        ax.set_xlabel("SampleID")
        ax.set_ylabel("Normalized Abundance")
        ax.legend(title="Species", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)
    except KeyError as e:
        st.error(f"Error plotting group abundance for group {group}: {e}. Check data consistency.")

def main():
    """Main function to run the Streamlit app."""
    st.title("16S Abundance Analysis")
    otu_df, metad_df = load_data() # Moved inside main

    if otu_df is not None and metad_df is not None:
        most_abundant_species, least_abundant_species = calculate_abundance(otu_df)
        group_analysis(otu_df, metad_df, most_abundant_species, least_abundant_species)
    elif otu_df is None and metad_df is None:
        st.info("Please upload OTU and Metadata files to begin the analysis.") #Added info message
    #removed the else condition

if __name__ == "__main__":
    main()
