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

# Initialize variables outside of conditional blocks
otu_file = None
metad_file = None
merged_data = None
grouped = None
total_species_abundance = None
most_abundant_species = None
least_abundant_species = None  # Initialize least_abundant_species
G0 = None
G1 = None
G2 = None
G3 = None
fig = None  # Initialize the figure variable
normalized_data = None  # Initialize normalized_data

# Ensure the DataFrame is Arrow-compatible before displaying
def make_arrow_compatible(df):
    for col in df.columns:
        if df[col].dtype == 'object':  # Check for non-numeric columns
            df[col] = df[col].astype(str)  # Convert to string explicitly
    return df

# Streamlit file uploader in the sidebar
st.sidebar.title("Upload Your Files")
uploaded_files = {
    "OTU File": st.sidebar.file_uploader("Upload OTU File", type=["xlsx", "csv"]),  # Allow both xlsx and csv
    "Metadata File": st.sidebar.file_uploader("Upload Metadata File", type=["xlsx", "csv"])  # Allow both xlsx and csv
}

# FORMAT STREAMLIT APP PRESENTATION
# Set the title of the Streamlit app
st.title("Microbial Analysis Dashboard")
st.subheader("Jacqueline Ferri's DS4PH capstone project")
st.write("May 8th, 2025")
st.write("Here you can upload files for analysis of 16s sequencing data.")

# add option for "Set-up Check" in the Streamlit app
st.subheader("Set-up Check")
setup_check = st.selectbox("Choose Analysis", ["None", "Transpose OTU File", "Transpose Metadata File", "Merged File", "Group SimpleIDs"])

# Check if "Set-up Check" is selected
if setup_check != "None":
    # Check if all files are uploaded
    if all(uploaded_files.values()):
        try:
            # Read the uploaded files into pandas DataFrames, handle both xlsx and csv
            otu_file_content = uploaded_files["OTU File"]
            metadata_file_content = uploaded_files["Metadata File"]

            otu_file = pd.read_excel(otu_file_content) if otu_file_content.name.endswith(("xlsx", "xls")) else pd.read_csv(io.BytesIO(otu_file_content.read()))
            metad_file = pd.read_excel(metadata_file_content) if metadata_file_content.name.endswith(("xlsx", "xls")) else pd.read_csv(io.BytesIO(metadata_file_content.read()))

            # Transpose otu_file and set #OTU_ID as the index
            otu_file.set_index("#OTU_ID", inplace=True)
            otu_file = otu_file.transpose()

            # Ensure the index of otu_file matches the SimpleID column in metad_file
            otu_file.index.name = "SimpleID"
            try:
                merged_data = otu_file.join(metad_file.set_index("SimpleID"), how="inner")
            except KeyError as e:
                st.error(
                    f"Error: {e}. Please ensure 'SimpleID' column exists in Metadata File and has matching values with OTU File index.")
                merged_data = None  # Set merged_data to None to prevent further errors

            if merged_data is not None:
                # Display outputs only if "Transpose OTU File" or "Transpose Metadata File" is selected
                if setup_check == "Transpose OTU File":
                    st.write("Transposed OTU File:")
                    otu_file_arrow_compatible = make_arrow_compatible(otu_file)
                    st.dataframe(otu_file_arrow_compatible)

                if setup_check == "Transpose Metadata File":
                    # Set SimpleID as the index for metadata
                    metad_file.set_index("SimpleID", inplace=True)
                    st.write("Metadata File:")
                    metad_file_arrow_compatible = make_arrow_compatible(metad_file)
                    st.dataframe(metad_file_arrow_compatible)

                if setup_check == "Merged File":
                    # Create a copy for display, preserving the original merged_data
                    merged_data_for_display = merged_data.copy()
                    merged_data_for_display["SimpleID"] = merged_data_for_display.index
                    merged_data_for_display.set_index("SimpleID", inplace=True)
                    st.write("Merged Data:")
                    merged_data_arrow_compatible = make_arrow_compatible(merged_data_for_display)
                    st.dataframe(merged_data_arrow_compatible)
                    # Ensure all columns in the DataFrame are numeric before summing
                    merged_data = merged_data.apply(pd.to_numeric, errors='coerce')

                if setup_check == "Group SimpleIDs":
                    # Group SimpleIDs based on "CAP regression by central review" column
                    if "CAP regression by central review" in merged_data.columns:
                        grouped = merged_data.groupby("CAP regression by central review").groups
                        st.write("Grouped Data by 'CAP regression by central review':")
                        st.write(grouped)

                        # Calculate total species abundance for each sample
                        numeric_columns = merged_data.select_dtypes(include=[np.number])
                        total_species_abundance = numeric_columns.sum(axis=1)
                        # Remove 'FinalCleanSeqs' from the total species abundance if it exists
                        if 'FinalCleanSeqs' in total_species_abundance.index:
                            total_species_abundance = total_species_abundance.drop('FinalCleanSeqs')

                        # Display the 10 most abundant species
                        most_abundant_species = total_species_abundance.nlargest(10)

                        # Display the 10 least abundant species
                        least_abundant_species = total_species_abundance.nsmallest(10)

                        # Extract SimpleIDs for each group
                        G0 = grouped.get("G0", [])
                        G1 = grouped.get("G1", [])
                        G2 = grouped.get("G2", [])
                        G3 = grouped.get("G3", [])
                    else:
                        st.error(
                            "Error: 'CAP regression by central review' column not found in Metadata File.")
                        grouped = None  # Ensure grouped is None to prevent errors later
        except Exception as e:
            st.error(f"An error occurred: {e}")
            otu_file = None
            metad_file = None
            merged_data = None

    else:
        st.warning("Please upload both OTU and Metadata files to use this feature.")

# Add an option for "Abundance Analysis" in the Streamlit app
st.subheader("Analysis Options")
analysis_option = st.selectbox("Choose Analysis", ["None", "Abundance Analysis", "Neural Network"])

if analysis_option == "Abundance Analysis":
    if merged_data is not None and grouped is not None:  # Check if merged_data and grouped are valid
        # Calculate total species abundance for each sample
        numeric_columns = merged_data.select_dtypes(include=[np.number])
        total_species_abundance = numeric_columns.sum(axis=1)

        # Remove 'FinalCleanSeqs' from the total species abundance if it exists
        if 'FinalCleanSeqs' in total_species_abundance.index:
            total_species_abundance = total_species_abundance.drop('FinalCleanSeqs')

        # drop zero sum rows
        total_species_abundance = total_species_abundance[total_species_abundance != 0]

        # drop lowest 5% of species
        threshold = total_species_abundance.quantile(0.05)
        total_species_abundance = total_species_abundance[total_species_abundance > threshold]

        # Group SimpleIDs based on "CAP regression by central review" column
        grouped = metad_file.groupby("CAP regression by central review").groups

        # Display the 10 most abundant species
        # most_abundant_species = total_species_abundance.nlargest(10)

        # Display the 10 least abundant species
        # least_abundant_species = total_species_abundance.nsmallest(10)

        # Extract SimpleIDs for each group
        G0 = grouped.get("G0", [])
        G1 = grouped.get("G1", [])
        G2 = grouped.get("G2", [])
        G3 = grouped.get("G3", [])

        # Select CAP regression group
        selected_group = st.selectbox("Select CAP Regression Group", ["G0", "G1", "G2", "G3"])

        # Get the corresponding SimpleIDs for the selected group
        selected_simple_ids = grouped.get(selected_group, [])

        if len(selected_simple_ids) > 0:
            # Filter the merged data for the selected SimpleIDs
            group_data = merged_data.loc[selected_simple_ids]

            # Calculate the total abundance for each species
            species_abundance = group_data.sum(axis=0).sort_values(ascending=False)

            # Get the top 10 most abundant species
            top_10_species = species_abundance.head(10)

            # Get the least abundant species
            least_abundant_species = species_abundance.tail(10)

            # Display the top 10 species with their counts per SimpleID
            top_10_species_data = group_data[top_10_species.index].transpose()
            st.write(f"Most Abundant Species with Counts per SimpleID for {selected_group}:")
            st.dataframe(top_10_species_data)

            # display the bottom 10 species with their counts per SimpleID
            # least_abundant_species_data = group_data[least_abundant_species.index].transpose()
            # st.write(f"Least Abundant Species with Counts per SimpleID for {selected_group}:")
            # st.dataframe(least_abundant_species_data)

            # Normalize the data for the top 10 species
            normalized_data = group_data[top_10_species.index].div(group_data[top_10_species.index].sum(axis=1), axis=0)

            # Plot a stacked bar chart for the normalized data\
            st.write(f"Normalized Abundance (Stacked Bar Chart) for {selected_group}:")
            fig, ax = plt.subplots(figsize=(20, 10))
            normalized_data.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
            ax.set_title(f"Normalized Abundance for {selected_group}")
            ax.set_ylabel("Proportion")
            ax.set_xlabel("Sample ID")
            st.pyplot(fig)
        else:
            st.write(f"No data available for group {selected_group}.")
    else:
        st.warning("Please upload and process data in the 'Set-up Check' section first.")

# Add an option for "Neural Network" in the Streamlit app
if analysis_option == "Neural Network":
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder
    import streamlit as st

    # Placeholder functions for diversity and functional profiling.  Replace these with your actual implementations.
    def calculate_diversity_indices(otu_file):
        """
        Calculates Shannon diversity index.  Replace with more comprehensive diversity calculation if needed.

        Args:
            otu_file (pd.DataFrame): OTU abundance data (samples x species).

        Returns:
            pd.Series: Shannon diversity index for each sample.
        """
        # Ensure no zeros to avoid errors in log calculation.  A small constant is added.
        otu_file = otu_file.replace(0, 1e-9)
        probabilities = otu_file / otu_file.sum(axis=1, keepdims=True)
        shannon_diversity = -np.sum(probabilities * np.log(probabilities), axis=1)
        return pd.Series(shannon_diversity, index=otu_file.index, name="Shannon_Diversity")

    def calculate_functional_profile(otu_data, kegg_mapping):
        """
        Calculates a simplified functional profile based on OTU abundances and a placeholder KEGG mapping.
        Replace with your actual functional profiling method.

        Args:
            otu_data (pd.DataFrame): OTU abundance data (samples x species).
            kegg_mapping (dict): A dictionary mapping species (columns in otu_data) to KEGG pathways.
                For simplicity, this example uses a hardcoded mapping.  In reality, this would come from a database
                or annotation file.  The structure is: {species_name: [pathway1, pathway2, ...], ...}

        Returns:
            pd.DataFrame: Functional profile (samples x pathways).
        """
        # Placeholder KEGG mapping (replace with your actual mapping)
        # This is just an example; in reality, this would come from a database.
        kegg_mapping = {
            otu: [f"pathway_{i + 1}"]  # Assign each OTU to a unique pathway
            for i, otu in enumerate(otu_data.columns)
        }

        # Create a set of all unique pathways.
        all_pathways = set()
        for pathways in kegg_mapping.values():
            all_pathways.update(pathways)

        # Initialize the functional profile DataFrame.
        functional_profile = pd.DataFrame(index=otu_data.index,
                                          columns=sorted(list(all_pathways)))  # Sort for consistency
        functional_profile = functional_profile.fillna(0)  # Initialize all counts to zero.

        # Calculate pathway abundances by summing the abundances of OTUs associated with each pathway.
        for species, pathways in kegg_mapping.items():
            if species in otu_data.columns:  # Make sure the species is actually in the OTU data.
                for pathway in pathways:
                    functional_profile[pathway] += otu_data[species]
        return functional_profile

    def create_model(input_dim):
        """
        Creates a simple Multi-Layer Perceptron (MLP) model for regression.

        Args:
            input_dim (int): The number of input features.

        Returns:
            tf.keras.Model: A compiled neural network model.
        """
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                            solver='adam', random_state=42, max_iter=500)
        return model

    def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
        """
        Trains and evaluates the given model.

        Args:
            model (tf.keras.Model): The neural network model to train.
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training target variable.
            X_test (pd.DataFrame or np.ndarray): Testing features.
            y_test (pd.Series or np.ndarray): Testing target variable.

        Returns:
            tuple: (mse, r2) - Mean squared error and R-squared on the test set.
        """
        st.write("Training the neural network model...")
        model.fit(X_train, y_train)  # No verbose output here; use Streamlit for messages

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def run_neural_network_analysis(otu_file, metad_file):
        """
        Main function to run the neural network analysis.  This is designed to be called
        from within a Streamlit app as an analysis option.

        Args:
            otu_file (pd.DataFrame): OTU abundance data.
            metad_file (pd.DataFrame): Metadata file.
        """

        # Preprocess the data (as in your original code)
        merged_data = otu_file.merge(metad_file, left_index=True, right_index=True)
        grouped = merged_data.groupby('CAP regression by central review')

        # Calculate abundant species (as in your original code)
        most_abundant_species = otu_file.sum(axis=0).nlargest(50)
        least_abundant_species = otu_file.sum(axis=0).nsmallest(50)
        abundant_species = pd.concat([most_abundant_species, least_abundant_species], axis=0)
        abundant_species_data = otu_file.loc[abundant_species.index].transpose()

        # Ensure the target variable is numeric
        label_encoder = LabelEncoder()
        metad_file["CAP regression by central review"] = label_encoder.fit_transform(
            metad_file["CAP regression by central review"].astype(str))

        # Align samples
        common_samples = abundant_species_data.index.intersection(metad_file.index)
        if len(common_samples) == 0:
            st.error("No common samples found between species data and metadata. Please check your data.")
            return  # Stop processing if no common samples

        X_species = abundant_species_data.loc[common_samples]
        y = metad_file.loc[common_samples, "CAP regression by central review"]

        # Feature Engineering
        st.write("Performing feature engineering...")
        diversity_indices = calculate_diversity_indices(X_species)
        # Replace this with your actual KEGG mapping.
        kegg_mapping = {otu: [f"pathway_{i + 1}"] for i, otu in enumerate(X_species.columns)}
        functional_profile = calculate_functional_profile(X_species, kegg_mapping)

        # Combine features
        X_combined = pd.concat([X_species, diversity_indices, functional_profile], axis=1)

        # Handle missing values (important before scaling)
        X_combined = X_combined.fillna(
            0)  # Or use a more sophisticated imputation method if appropriate for your data

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create and train the model
        input_dim = X_train_scaled.shape[1]
        model = create_model(input_dim)
        mse, r2 = train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

        # Display results
        st.write("Neural Network Model Performance:")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")
        st.write(
            "Note:  This model includes feature engineering (diversity indices and a placeholder functional profile).  Ensure the `calculate_functional_profile` function is replaced with your actual functional profiling implementation and that the KEGG mapping is correct and comprehensive.")

    if __name__ == "__main__":
        # Load your data here
        # Example usage with placeholder data:
        data = {'Species1': [10, 5, 20, 15, 8],
                'Species2': [2, 8, 5, 12, 3],
                'Species3': [5, 10, 15, 8, 20],
                'Species4': [12, 3, 8, 5, 10],
                'Species5': [8, 20, 10, 15, 5]}
        otu_file = pd.DataFrame(data, index=['Sample1', 'Sample2', 'Sample3', 'Sample4', 'Sample5'])

        meta_data = {'CAP regression by central review': ['Low', 'Medium', 'High', 'Low', 'Medium'],
                    'Treatment': ['A', 'B', 'A', 'B', 'A']}
        metad_file = pd.DataFrame(meta_data, index=['Sample1', 'Sample2', 'Sample3', 'Sample4', 'Sample5'])
        run_neural_network_analysis(otu_file, metad_file)

