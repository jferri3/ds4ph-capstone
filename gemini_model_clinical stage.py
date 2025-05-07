import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns


#format app
st.subheader("Jacqueline's Capstone Project - May 08, 2025")
st.title("Microbiome Data Analysis and Prediction")
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
        st.markdown("Visualize the first few rows of each dataframe to check the set up")
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

        # Print the first few rows of combined_df and its index
        st.subheader("Combined DataFrame (First 5 Rows):")
        st.dataframe(combined_df.head())
        st.write(f"Index of combined_df: {combined_df.index}")

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
        st.write("These are the 10 species with the highest total abundance across all samples:")
        st.dataframe(top_10_species)
        # Display the 10 least abundant species in Streamlit
        st.subheader("10 Least Abundant Species")
        st.write("These are the 10 species with the lowest total abundance across all samples:")
        st.dataframe(least_10_species)

        # Display the total species abundance in Streamlit
        st.subheader("Total Species Abundance")
        st.write("This table shows the total abundance for each species:")
        st.dataframe(otu_table[['Total Abundance']])

        # Determine the most abundant species for each group
        group_abundance = otu_file.groupby(combined_df['CAP regression by central review']).sum()

        # Identify the top 10 most abundant species for each group
        top_10_species_per_group = group_abundance.apply(lambda x: x.nlargest(10).index.tolist(), axis=1).transpose()

        # Display the top 10 most abundant species for each group in Streamlit
        st.subheader("Top 10 Most Abundant Species per Group")
        st.write("These are the top 10 most abundant species within each group:")
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
        st.write("Feature importances from the Random Forest model:")
        st.dataframe(importance_df)

        # Display the predicted groups
        predicted_groups = model.predict(X_test)
        predicted_groups = np.round(predicted_groups).astype(int)

        # Create a DataFrame to compare actual and predicted groups
        comparison_table = pd.DataFrame({
            'SimpleID': X_test.index, # Changed 'SampleID' to 'SimpleID' and used X_test index
            'Actual Group': y_test.values,
            'Predicted Group': predicted_groups,
        })

        # Add a column indicating whether the prediction was accurate
        comparison_table['Correct Prediction'] = comparison_table['Actual Group'] == comparison_table['Predicted Group']

        # Display the table
        st.subheader("Comparison of Actual and Predicted Groups")
        st.write("Comparison of actual and predicted groups for the testing set:")
        st.dataframe(comparison_table)

        # --- Feature Plots ---
        st.header("Feature Plots")
        st.write("Select features to visualize:")

        # Use st.columns to create a layout with two columns
        col1, col2 = st.columns(2)

        # Put the checkboxes in the first column
        with col1:
            # Create checkboxes for each feature
            selected_features = []
            for feature in features:
                if st.checkbox(feature):
                    selected_features.append(feature)

        # Add a multiselect for the groups
        selected_groups = st.multiselect("Select Groups to Display",
                                         options=combined_df['CAP regression by central review'].unique(),
                                         default=combined_df['CAP regression by central review'].unique())

        # In the second column, display the plots
        with col2:
            if selected_features:
                for feature in selected_features:
                    if feature in ['PC1', 'PC2', 'PC3', 'PC1.centroid', 'PC2.centroid']:
                        # Create a PCA scatter plot
                        chart = alt.Chart(combined_df).mark_circle().encode( # removed filtering
                            x=alt.X('PC1'),
                            y=alt.Y('PC2'),
                            color=alt.Color('CAP regression by central review', title = "Actual Group"),  # Use color to distinguish groups
                            tooltip=['SimpleID', 'PC1', 'PC2', 'CAP regression by central review']
                        ).properties(
                            title='PCA Plot (PC1 vs PC2)',
                            width=400,
                            height=300
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)
                    elif feature in ['goods_coverage', 'simpson_reciprocal', 'chao1', 'PD_whole_tree', 'observed_species', 'shannon', 'gini_index', 'fisher_alpha', 'margalef', 'brillouin_d']:
                         # Create a  scatter plot for alpha diversity indices, with SimpleID on the x-axis and the index on the y-axis
                        chart = alt.Chart(combined_df).mark_point().encode( # removed filtering
                            x=alt.X('SimpleID'),
                            y=alt.Y(feature),
                            color=alt.Color('CAP regression by central review', title = "Actual Group"),
                            tooltip=['SimpleID', feature]
                        ).properties(
                            title=f"{feature} vs. Sample ID",
                            width=400,
                            height=300
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        # Create a  histogram for other features
                        chart = alt.Chart(combined_df).mark_bar().encode( # removed filtering
                            x=alt.X(feature, bin=True),  # Use bin=True for histograms
                            y='count()',
                            color=alt.Color('CAP regression by central review', title = "Actual Group"),
                            tooltip=[feature, 'count()']
                        ).properties(
                            title=f"Distribution of {feature}",
                            width=400,  # Adjust as needed
                            height=300
                        ).interactive()  # Make the plot interactive

                        st.altair_chart(chart, use_container_width=True)
            else:
                st.write("Please select features to visualize.")

        # --- Prediction Section ---
        st.header("CAP Regression Prediction")
        st.write("Predict CAP regression by central review using the combined data.")

        # Use the combined_df for prediction
        if combined_df is not None:  # Check if combined_df is available
            try:
                # Prepare data for prediction
                X_pred = combined_df[features]  # Use the same features as the model was trained on

                # Make predictions using the trained model
                new_predictions = model.predict(X_pred)
                new_predictions = np.round(new_predictions).astype(int)

                #  Create a DataFrame to display the predictions, including SimpleID
                prediction_df = pd.DataFrame(
                    {'SimpleID': combined_df['SimpleID'],
                     'Predicted Group': new_predictions,
                     'Actual Group': combined_df['CAP regression by central review']
                     })
                st.subheader("Predictions")
                st.write("Predicted CAP regression by central review:")
                st.dataframe(prediction_df)

                # Create a scatter plot to visualize prediction accuracy
                st.subheader("Predicted vs. Actual Groups per Group")
                for group_val in sorted(prediction_df['Actual Group'].unique()):
                    group_df = prediction_df[prediction_df['Actual Group'] == group_val]
                    chart = alt.Chart(group_df).mark_circle().encode(
                        x=alt.X('SimpleID', title='Sample ID'),
                        y=alt.Y('Predicted Group', title='Predicted Group'),
                        color=alt.Color('Actual Group', title = "Actual Group"), # add legend title
                        tooltip=['SimpleID', 'Actual Group', 'Predicted Group']
                    ).properties(
                        title=f'Group {group_val} Predictions',
                        width=600,
                        height=400
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)

                # Calculate and display accuracy per group
                st.subheader("Accuracy per Group")
                group_accuracy = comparison_table.groupby('Actual Group')['Correct Prediction'].mean()
                st.dataframe(group_accuracy)

                # Display the confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(plt)

              # Display multiple ROC curves and AUC scores
                auc = 0  # Initialize auc here
                unique_classes = np.unique(y)

                # Create a figure for individual class plots
                fig_individual, axs = plt.subplots(len(unique_classes), figsize=(6, 4 * len(unique_classes)))

                if len(unique_classes) > 2:
                    # For multiclass, use one-vs-rest strategy
                    y_prob = model.predict_proba(X_test)
                    auc_scores = {}
                    plt.figure(figsize=(8, 6))
                    for i, cls in enumerate(unique_classes):
                        fpr, tpr, _ = roc_curve(y_test == cls, y_prob[:, i])
                        auc_scores[cls] = roc_auc_score(y_test == cls, y_prob[:, i])
                        axs[i].plot(fpr, tpr, label=f'Class {cls} (AUC = {auc_scores[cls]:.2f})')
                        axs[i].plot([0, 1], [0, 1], 'k--', label='Random')
                        axs[i].set_title(f'ROC Curve for Class {cls}')
                        axs[i].set_xlabel('False Positive Rate')
                        axs[i].set_ylabel('True Positive Rate')
                        axs[i].legend()

                        # Add to combined plot
                        plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {auc_scores[cls]:.2f})')

                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Combined ROC Curve')
                    plt.legend()
                    st.pyplot(fig_individual)
                    st.pyplot(plt)
                else:
                    # For binary classification
                    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    auc = roc_auc_score(y_test, y_prob)
                    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend()
                    st.pyplot(plt)

                st.write(f"AUC: {auc:.2f}")

                # Create and display the detailed table per group
                st.subheader("Detailed Predictions per Group")
                for group_name, group_data in comparison_table.groupby('Actual Group'):
                    st.write(f"Group: {group_name}")  # Display the group name
                    group_data['Accuracy'] = group_data['Actual Group'] == group_data['Predicted Group']
                    group_table = group_data[['SimpleID', 'Actual Group', 'Predicted Group', 'Accuracy']]
                    st.dataframe(group_table)

                # Display top 10 species per group
                st.subheader("Top 10 Most Abundant Species per Group")
                for group_name, group_species in top_10_species_per_group.items():
                    st.write(f"Group: {group_name}")
                    st.write(group_species)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Prediction requires the combined data. Please upload the necessary files to proceed.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload all four required files (OTU, Alpha Diversity, Metadata, PCoA) in Excel format.")

