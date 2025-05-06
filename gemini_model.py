import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import altair as alt

# Define a function to load data
def load_data(otu_file, metad_file):
    """
    Loads OTU and metadata files from CSV or TSV formats.

    Args:
        otu_file (UploadedFile): Streamlit UploadedFile object for OTU data.
        metad_file (UploadedFile): Streamlit UploadedFile object for metadata.

    Returns:
        tuple: A tuple containing OTU DataFrame, metadata DataFrame, and a list of
               warnings encountered during data loading.  Returns (None, None, warnings)
               if there are errors.
    """
    warnings = []
    otu_df = None
    metad_df = None
    try:
        # Read OTU data, try CSV first, then TSV
        try:
            otu_df = pd.read_csv(otu_file)
        except Exception as e:
            otu_file.seek(0)  # Reset file pointer
            try:
                otu_df = pd.read_csv(otu_file, sep='\t')
            except Exception as e:
                warnings.append(f"Error reading OTU file: {e}")
                return None, None, warnings

        # Read metadata, try CSV first, then TSV
        try:
            metad_df = pd.read_csv(metad_file)
        except Exception as e:
            metad_file.seek(0)  # Reset file pointer
            try:
                metad_df = pd.read_csv(metad_file, sep='\t')
            except Exception as e:
                warnings.append(f"Error reading metadata file: {e}")
                return None, None, warnings

        # Basic data validation
        if otu_df is not None and otu_df.empty:
            warnings.append("OTU file is empty.")
            return None, None, warnings
        if metad_df is not None and metad_df.empty:
            warnings.append("Metadata file is empty.")
            return None, None, warnings
        if otu_df is not None and not all(isinstance(col, (str, np.str_)) for col in otu_df.columns):
            warnings.append("OTU file: All column names should be strings.")
            return None, None, warnings
        if metad_df is not None and not all(isinstance(col, (str, np.str_)) for col in metad_df.columns):
            warnings.append("Metadata file: All column names should be strings.")
            return None, None, warnings

        if otu_df is not None and metad_df is not None:
            # Transpose OTU data and Check for common sample IDs, case-insensitively
            otu_df = otu_df.set_index("#OTU_ID").T.reset_index().rename(columns={"index": "id_name"})

            otu_sample_ids = set(otu_df.columns.str.lower())
            metadata_sample_ids = set(metad_df.columns.str.lower())

            if "id_name" not in otu_sample_ids:
                warnings.append("Error: '#OTU_ID' column not found in OTU file.")
                return None, None, warnings
            if "simpleid" not in metadata_sample_ids:
                warnings.append("Error: 'SimpleID' column not found in metadata file.")
                return None, None, warnings

            # Ensure the correct column names are used for merging
            metad_df = metad_df.rename(columns={"SimpleID": "id_name"})

            common_ids = set(otu_df["id_name"].str.lower()).intersection(set(metad_df["id_name"].str.lower()))
            if not common_ids:
                warnings.append("Error: No common sample IDs found between OTU and metadata files (case-insensitive check).")
                return None, None, warnings

        return otu_df, metad_df, warnings

    except Exception as e:
        warnings.append(f"An unexpected error occurred: {e}")
        return None, None, warnings

# Function to perform CAP regression
def cap_regression(otu_df, metad_df, group_column):
    """
    Performs Canonical Analysis of Principal coordinates (CAP) regression to find
    most and least abundant OTUs for each group.  This is a simplified version
    as the full CAP requires more complex ecological calculations (e.g., using
    the `vegan` R package).  This function approximates the desired output
    by performing a group-wise analysis of OTU abundances.

    Args:
        otu_df (pd.DataFrame): OTU abundance data.
        metad_df (pd.DataFrame): Metadata associated with the OTU data.
        group_column (str): The column in the metadata to group the samples by.

    Returns:
        tuple: Two dictionaries, `most_abundant` and `least_abundant`, containing
               the most and least abundant OTUs for each group, respectively.
               Returns (None, None) if an error occurs.
    """
    try:
        # Ensure that the first column is the sample ID for both DataFrames.
        # otu_df = otu_df.rename(columns={otu_df.iloc[:, 0].name: 'id_name'}) # Already renamed in load_data
        # metad_df = metad_df.rename(columns={metad_df.iloc[:, 0].name: 'id_name'}) # Already renamed in load_data

        # Merge OTU and metadata on the sample ID (first column)
        merged_df = pd.merge(otu_df, metad_df, on='id_name', how='inner')

        if group_column not in merged_df.columns:
            st.error(f"Group column '{group_column}' not found in metadata.")
            return None, None

        # Group by the specified column
        grouped = merged_df.groupby(group_column)

        most_abundant = {}
        least_abundant = {}

        for group_name, group_data in grouped:
            # Exclude the id_name and grouping column for the mean calculation
            otu_columns = group_data.select_dtypes(include=np.number).columns
            if len(otu_columns) == 0:
                most_abundant[group_name] = "No OTUs"
                least_abundant[group_name] = "No OTUs"
                continue
            mean_abundances = group_data[otu_columns].mean().sort_values(ascending=False)
            if not mean_abundances.empty:
                most_abundant[group_name] = mean_abundances.index[0]
                least_abundant[group_name] = mean_abundances.index[-1]
            else:
                most_abundant[group_name] = "No OTUs"
                least_abundant[group_name] = "No OTUs"

        return most_abundant, least_abundant

    except Exception as e:
        st.error(f"Error in CAP regression analysis: {e}")
        return None, None

# Function to predict groups based on OTU abundance
def predict_groups(otu_df, metad_df, group_column, test_size=0.2, random_state=42):
    """
    Predicts groups based on OTU abundance using several machine learning models.

    Args:
        otu_df (pd.DataFrame): OTU abundance data.
        metad_df (pd.DataFrame): Metadata associated with the OTU data.
        group_column (str): The column in the metadata to predict.
        test_size (float, optional): The proportion of the data to use for testing.
            Defaults to 0.2.
        random_state (int, optional): Random state for train-test split. Defaults to 42.

    Returns:
        dict: A dictionary containing the classification reports for each model.
              Returns an empty dictionary if an error occurs.
    """
    try:
        # Ensure that the first column is the sample ID for both DataFrames.
        # otu_df = otu_df.rename(columns={otu_df.iloc[:, 0].name: 'id_name'}) # Already renamed
        # metad_df = metad_df.rename(columns={metad_df.iloc[:, 0].name: 'id_name'}) # Already renamed

        # Merge OTU and metadata
        merged_df = pd.merge(otu_df, metad_df, on='id_name', how='inner')

        if group_column not in merged_df.columns:
            st.error(f"Group column '{group_column}' not found in metadata.")
            return {}

        # Prepare data for classification
        X = merged_df.select_dtypes(include=np.number)  # Use only numeric columns from OTU data
        X = X.drop(columns=['id_name', group_column], errors='ignore')
        y = merged_df[group_column]

        # Check if there are enough samples and features
        if X.shape[0] < 2:
            st.error("Insufficient data: Need at least 2 samples to perform classification.")
            return {}
        if X.shape[1] < 1:
            st.error("Insufficient data: Need at least 1 OTU column to perform classification.")
            return {}
        if len(np.unique(y)) < 2:
            st.error("Insufficient data: Need at least 2 groups to perform classification.")
            return {}

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state,
                                                            stratify=y) # Stratify based on the target variable

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define models
        models = {
            "Logistic Regression": LogisticRegression(random_state=random_state, solver='liblinear'),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(random_state=random_state),
            "Random Forest": RandomForestClassifier(random_state=random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Neural Network": MLPClassifier(random_state=random_state, max_iter=500),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
            "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
            "Gaussian Process": GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=random_state)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = classification_report(y_test, y_pred, zero_division=0)  # Handle cases with no predicted samples

        return results

    except Exception as e:
        st.error(f"Error in group prediction: {e}")
        return {}

def display_results(results):
    """
    Displays the results of the classification models.

    Args:
        results (dict): A dictionary containing the classification reports.
    """
    for model, report in results.items():
        st.subheader(model)
        st.text(report)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Microbiome Data Analysis App")

    # Sidebar for file uploads
    st.sidebar.header("Upload Data")
    otu_file = st.sidebar.file_uploader("Upload OTU Data (CSV or TSV)", type=["csv", "tsv"])
    metad_file = st.sidebar.file_uploader("Upload Metadata (CSV or TSV)", type=["csv", "tsv"])

    # Main page content
    if otu_file and metad_file:
        otu_df, metad_df, warnings = load_data(otu_file, metad_file)
        if warnings:
            for warning in warnings:
                st.warning(warning)
        if otu_df is not None and metad_df is not None:

            # Allow user to select the grouping column
            group_column = st.sidebar.selectbox("Select Grouping Column", metad_df.columns)

            st.header("CAP Regression Analysis")
            st.write("Most and Least Abundant OTUs by Group")
            most_abundant, least_abundant = cap_regression(otu_df, metad_df, group_column)
            if most_abundant and least_abundant:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Most Abundant OTUs")
                    for group, otu in most_abundant.items():
                        st.write(f"{group}: {otu}")
                with col2:
                    st.subheader("Least Abundant OTUs")
                    for group, otu in least_abundant.items():
                        st.write(f"{group}: {otu}")

            st.header("Predictive Model")
            st.write("Predicting Group Membership based on OTU Abundance")
            results = predict_groups(otu_df, metad_df, group_column)
            if results:
                display_results(results)
        elif otu_df is None or metad_df is None: # added this condition
            st.error("Please upload both OTU and Metadata files correctly.")

if __name__ == "__main__":
    main()
