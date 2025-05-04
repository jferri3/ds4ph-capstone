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
setup_check = st.selectbox("Choose Analysis", ["None", "Transpose OTU File", "Transpose Metadata File"])

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
                st.error(f"Error: {e}. Please ensure 'SimpleID' column exists in Metadata File and has matching values with OTU File index.")
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
                    # Ensure all columns in the DataFrame are numeric before summing
                    merged_data = merged_data.apply(pd.to_numeric, errors='coerce')

                    st.write("Merged Data:")
                    merged_data_arrow_compatible = make_arrow_compatible(merged_data)
                    st.dataframe(merged_data_arrow_compatible)

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
                    if "CAP regression by central review" in metad_file.columns:
                        grouped = metad_file.groupby("CAP regression by central review").groups
                        st.write("Grouped Data by 'CAP regression by central review':")
                        st.write(grouped)

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
                        st.error("Error: 'CAP regression by central review' column not found in Metadata File.")
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
analysis_option = st.selectbox("Choose Analysis", ["None", "Abundance Analysis", "Convoluted Neural Network (CNN)"])

if analysis_option == "Abundance Analysis":
    if merged_data is not None and grouped is not None: # Check if merged_data and grouped are valid
        # Calculate total species abundance for each sample
        numeric_columns = merged_data.select_dtypes(include=[np.number])
        total_species_abundance = numeric_columns.sum(axis=1)

        # Group SimpleIDs based on "CAP regression by central review" column
        grouped = metad_file.groupby("CAP regression by central review").groups

        # Display the 10 most abundant species
        most_abundant_species = total_species_abundance.nlargest(10)

        # Display the 10 least abundant species
        least_abundant_species = total_species_abundance.nsmallest(10)

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
            #least_abundant_species_data = group_data[least_abundant_species.index].transpose()
            #st.write(f"Least Abundant Species with Counts per SimpleID for {selected_group}:")
            #st.dataframe(least_abundant_species_data)
            
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

# Add an option for "Convolutional Neural Network (CNN)" in the Streamlit app
if analysis_option == "Convoluted Neural Network (CNN)":
    if merged_data is not None and grouped is not None:  # Check if merged_data and grouped are valid
        try:
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

            # Standardize the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Reshape the data for CNN (e.g., as a 2D "image" with 1 channel)
            X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
            X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

            # Define the CNN model
            cnn_model = models.Sequential([
                layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
                layers.MaxPooling1D(pool_size=2),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(1, activation='linear')  # For regression
            ])

            # Compile the model
            cnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

            # Train the model
            st.write("Training the CNN model...")
            cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

            # Make predictions
            y_pred_cnn = cnn_model.predict(X_test_cnn)

            # Evaluate the model
            mse_cnn = mean_squared_error(y_test, y_pred_cnn)
            r2_cnn = r2_score(y_test, y_pred_cnn)

            # Display performance metrics
            st.write("Convolutional Neural Network (CNN) Model Performance:")
            st.write(f"Mean Squared Error (CNN): {mse_cnn}")
            st.write(f"R-squared (CNN): {r2_cnn}")

            # Create and train the linear regression model for comparison
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            y_pred_linear = linear_model.predict(X_test)

            # Evaluate the linear regression model
            mse_linear = mean_squared_error(y_test, y_pred_linear)
            r2_linear = r2_score(y_test, y_pred_linear)

            # Display performance metrics for linear regression
            st.write("Linear Regression Model Performance:")
            st.write(f"Mean Squared Error (Linear Regression): {mse_linear}")
            st.write(f"R-squared (Linear Regression): {r2_linear}")

        except Exception as e:
            st.error(f"An error occurred while training the CNN model: {e}")
    else:
        st.warning("Please upload and process data in the 'Set-up Check' section first.")