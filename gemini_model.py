import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# File upload section
otu_file = st.sidebar.file_uploader("Upload OTU File (Excel)", type=["xlsx"])
alpha_file = st.sidebar.file_uploader("Upload Alpha Diversity File (Excel)", type=["xlsx"])
metad_file = st.sidebar.file_uploader("Upload Metadata File (Excel)", type=["xlsx"])
pcoa_file = st.sidebar.file_uploader("Upload PCoA File (Excel)", type=["xlsx"])

if otu_file and alpha_file and metad_file and pcoa_file:
    try:
        # Read the Excel files
        otu_df = pd.read_excel(otu_file)
        alpha_df = pd.read_excel(alpha_file)
        metad_df = pd.read_excel(metad_file)
        pcoa_df = pd.read_excel(pcoa_file)

        # tranpose otu_df
        otu_df = otu_df.transpose()
        # Reset the index of otu_df as SimpleID
        otu_df.reset_index(inplace=True)
        # make row 1 as header
        otu_df.columns = otu_df.iloc[0]
        otu_df = otu_df[1:]
        # in otu_df, rename the first column to 'SimpleID'
        otu_df.rename(columns={otu_df.columns[0]: 'SimpleID'}, inplace=True)

        # in alpha diversity file, rename the first column to 'SimpleID'
        alpha_df.rename(columns={alpha_df.columns[0]: 'SimpleID'}, inplace=True)

        # in pcoa file, rename the third column to 'SimpleID'   
        pcoa_df.rename(columns={pcoa_df.columns[2]: 'SimpleID'}, inplace=True)
        # in pcoa file, remove first column
        pcoa_df = pcoa_df.iloc[:, 1:]

        # print in streamlit
        st.header("Uploaded Dataframes")
        st.subheader("OTU Data")
        st.dataframe(otu_df.head())
        st.subheader("Alpha Diversity Data")
        st.dataframe(alpha_df.head())
        st.subheader("Metadata")
        st.dataframe(metad_df.head())
        st.subheader("PCoA Data")
        st.dataframe(pcoa_df.head())

        # Data Preprocessing
        otu_file = otu_df.apply(pd.to_numeric, errors='coerce')
        otu_file = otu_file.loc[:, otu_file.sum(axis=0) > 10]

        # Merge dataframes
        combined_df = metad_df.merge(alpha_df.loc[:, ~alpha_df.columns.isin(metad_df.columns) | (alpha_df.columns == 'SimpleID')], on='SimpleID', how='inner') \
                      .merge(pcoa_df.loc[:, ~pcoa_df.columns.isin(metad_df.columns) | (pcoa_df.columns == 'SimpleID')], on='SimpleID', how='inner')

       

        # Assign SimpleID's into groups based on "CAP regression by central review"
        grouped_simple_ids = combined_df[['SimpleID', 'CAP regression by central review']].copy()

        # Create a table listing SimpleIDs per group
        group_table = grouped_simple_ids.groupby('CAP regression by central review')['SimpleID'].apply(list).reset_index()
        group_table.columns = ['Group', 'SimpleIDs']

        # Display the table in Streamlit
        st.subheader("SimpleIDs per Group")
        st.dataframe(group_table)

        # Create a copy of otu_file named "otu_table" and transpose it
        otu_table = otu_file.transpose()

        # Calculate total species abundance
        otu_table['Total Abundance'] = otu_table.sum(axis=1)
        # Identify the top 10 most abundant species
        top_10_species = otu_table['Total Abundance'].nlargest(10)
        # Identify the 10 least abundant species
        least_10_species = otu_table['Total Abundance'].nsmallest(10)
        # Display the top 10 most abundant species in Streamlit
        st.subheader("Top 10 Most Abundant Species")
        st.dataframe(top_10_species)
        # Display the 10 least abundant species in Streamlit
        st.subheader("10 Least Abundant Species")
        st.dataframe(least_10_species)

        # Display the total species abundance in Streamlit
        st.subheader("Total Species Abundance")
        st.dataframe(otu_table[['Total Abundance']])

        # Determine the most abundant species for each group
        group_abundance = otu_file.groupby(combined_df['CAP regression by central review']).sum()

        # Identify the top 10 most abundant species for each group
        top_10_species_per_group = group_abundance.apply(lambda x: x.nlargest(10).index.tolist(), axis=1).transpose()

        # Display the top 10 most abundant species for each group in Streamlit
        st.subheader("Top 10 Most Abundant Species per Group")
        st.dataframe(top_10_species_per_group)

        
        # Select features and target
        features = ['goods_coverage', 'simpson_reciprocal', 'chao1', 'PD_whole_tree', 'observed_species', 'shannon', 'gini_index', 'fisher_alpha', 'margalef', 'brillouin_d', 'PC1', 'PC2', 'PC3', 'PC1.centroid', 'PC2.centroid']
        target_column = 'CAP regression by central review'

        # Handle missing values
        combined_df[features] = combined_df[features].fillna(combined_df[features].mean())
        combined_df[target_column] = combined_df[target_column].fillna(combined_df[target_column].mode()[0])

        # Encode the target variable
        label_encoder = LabelEncoder()
        combined_df[target_column] = label_encoder.fit_transform(combined_df[target_column])


        X = combined_df[features]
        y = combined_df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy:.2f}")

        # Display classification report
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Feature Importance
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.subheader("Feature Importance")
        st.dataframe(importance_df)

        # Display the predicted groups
        predicted_groups = model.predict(X_test)
        predicted_groups = np.round(predicted_groups).astype(int)

        # Create a DataFrame to compare actual and predicted groups
        comparison_table = pd.DataFrame({
            'SampleID': y_test.index,
            'Actual Group': y_test.values,
            'Predicted Group': predicted_groups
        })

        # Add a column indicating whether the prediction was accurate
        comparison_table['Correct Prediction'] = comparison_table['Actual Group'] == comparison_table['Predicted Group']

        # Display the table
        st.subheader("Comparison of Actual and Predicted Groups")
        st.dataframe(comparison_table)



    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload all four required files (OTU, Alpha Diversity, Metadata, PCoA) in Excel format.")
