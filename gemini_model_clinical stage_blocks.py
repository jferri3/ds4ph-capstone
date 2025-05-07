import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#format app
st.subheader("Jacqueline's Capstone Project - May 08, 2025")
st.title("Microbiome Data Analysis and Prediction")
# File upload section
otu_file = st.sidebar.file_uploader("Upload OTU File (Excel)", type=["xlsx"])
alpha_file = st.sidebar.file_uploader("Upload Alpha Diversity File (Excel)", type=["xlsx"])
metad_file = st.sidebar.file_uploader("Upload Metadata File (Excel)", type=["xlsx"])
pcoa_file = st.sidebar.file_uploader("Upload PCoA File (Excel)", type=["xlsx"])

# Add clickable sections for content
with st.expander("Introduction"):
    st.write("This section provides an overview of the application and its purpose.")

with st.expander("File Upload Instructions"):
    st.write("Upload the required files in Excel format. The files include OTU, Alpha Diversity, Metadata, and PCoA.")

with st.expander("Uploaded Dataframes"):
    st.write("View the first few rows of each uploaded dataframe to ensure proper formatting.")

with st.expander("Data Preprocessing"):
    st.write("Details about the preprocessing steps applied to the data.")

with st.expander("SimpleIDs per Group"):
    st.write("View the grouping of SimpleIDs based on Clinical Stage.")

with st.expander("Species Abundance Analysis"):
    st.write("Explore the total species abundance, top 10 most abundant species, and least abundant species.")

with st.expander("Random Forest Model"):
    st.write("Details about the Random Forest model training, evaluation metrics, and feature importance.")

with st.expander("Neural Network Model"):
    st.write("Details about the Neural Network model training and evaluation.")

with st.expander("Feature Plots"):
    st.write("Visualize selected features and their relationships with groups.")

with st.expander("Random Forest Prediction Results"):
    st.write("View predictions for CAP regression by central review using Random Forest and evaluate model performance.")

with st.expander("Neural Network Prediction Results"):
    st.write("View predictions for CAP regression by central review using Neural Network and evaluate model performance.")

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
        otu_file_processed = otu_df.apply(pd.to_numeric, errors='coerce')
        otu_file_processed = otu_file_processed.loc[:, otu_file_processed.sum(axis=0) > 10]

        # Merge dataframes
        combined_df = metad_df.merge(alpha_df.loc[:, ~alpha_df.columns.isin(metad_df.columns) | (alpha_df.columns == 'SimpleID')], on='SimpleID', how='inner') \
                      .merge(pcoa_df.loc[:, ~pcoa_df.columns.isin(metad_df.columns) | (pcoa_df.columns == 'SimpleID')], on='SimpleID', how='inner')

        # Print the first few rows of combined_df and its index
        st.subheader("Combined DataFrame (First 5 Rows):")
        st.dataframe(combined_df.head())
        st.write(f"Index of combined_df: {combined_df.index}")

        # Assign SimpleID's into groups based on "Clinical stage"
        grouped_simple_ids = combined_df[['SimpleID', 'Clinical stage']].copy()

        # Create a table listing SimpleIDs per group
        group_table = grouped_simple_ids.groupby('Clinical stage')['SimpleID'].apply(list).reset_index()
        group_table.columns = ['Group', 'SimpleIDs']

        # Display the table in Streamlit
        st.subheader("SimpleIDs per Group")
        st.dataframe(group_table)

        # Create a copy of otu_file named "otu_table" and transpose it
        otu_table = otu_file_processed.transpose()

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
        st.dataframe(otu_table[['Total Abundance']].head())

        # Determine the most abundant species for each group
        group_abundance = otu_file_processed.groupby(combined_df['Clinical stage']).sum()

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

        # --- Random Forest Model ---
        st.subheader("Random Forest Model")
        # Train a Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        rf_y_pred = rf_model.predict(X_test)

        # Evaluate the model
        rf_accuracy = accuracy_score(y_test, rf_y_pred)
        st.write(f"Random Forest Accuracy: {rf_accuracy:.2f}")

        # Display classification report
        st.text("Random Forest Classification Report:")
        st.text(classification_report(y_test, rf_y_pred))

        # Feature Importance
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.subheader("Feature Importance")
        st.write("Feature importances from the Random Forest model:")
        st.dataframe(importance_df)

        # Display the predicted groups
        rf_predicted_groups = rf_model.predict(X_test)
        rf_predicted_groups = np.round(rf_predicted_groups).astype(int)

        # Create a DataFrame to compare actual and predicted groups
        rf_comparison_table = pd.DataFrame({
            'SimpleID': X_test.index,
            'Actual Group': y_test.values,
            'Predicted Group': rf_predicted_groups,
        })
        rf_comparison_table['Correct Prediction'] = rf_comparison_table['Actual Group'] == rf_comparison_table['Predicted Group']

        # Display the table
        st.subheader("Comparison of Actual and Predicted Groups (Random Forest)")
        st.write("Comparison of actual and predicted groups for the testing set (Random Forest):")
        st.dataframe(rf_comparison_table)

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
                                         options=combined_df['Clinical stage'].unique(),
                                         default=combined_df['Clinical stage'].unique())

        # In the second column, display the plots
        with col2:
            if selected_features:
                for feature in selected_features:
                    if feature in ['PC1', 'PC2', 'PC3', 'PC1.centroid', 'PC2.centroid']:
                        # Create a PCA scatter plot
                        chart = alt.Chart(combined_df).mark_circle().encode(
                            x=alt.X('PC1'),
                            y=alt.Y('PC2'),
                            color=alt.Color('Clinical stage', title="Actual Group"),  # Use color to distinguish groups
                            tooltip=['SimpleID', 'PC1', 'PC2', 'Clinical stage']
                        ).properties(
                            title='PCA Plot (PC1 vs PC2)',
                            width=400,
                            height=300
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)
                    elif feature in ['goods_coverage', 'simpson_reciprocal', 'chao1', 'PD_whole_tree', 'observed_species', 'shannon', 'gini_index', 'fisher_alpha', 'margalef', 'brillouin_d']:
                        # Create a scatter plot for alpha diversity indices, with SimpleID on the x-axis and the index on the y-axis
                        chart = alt.Chart(combined_df).mark_point().encode(
                            x=alt.X('SimpleID', sort=alt.EncodingSortField(field='Clinical stage', order='ascending')),  # Sort by Clinical stage
                            y=alt.Y(feature),
                            color=alt.Color('Clinical stage', title="Actual Group"),
                            tooltip=['SimpleID', feature]
                        ).properties(
                            title=f"{feature} vs. Sample ID",
                            width=400,
                            height=300
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        # Create a histogram for other features
                        chart = alt.Chart(combined_df).mark_bar().encode(
                            x=alt.X(feature, bin=True),  # Use bin=True for histograms
                            y='count()',
                            color=alt.Color('CAP regression by central review', title="Actual Group"),
                            tooltip=[feature, 'count()']
                        ).properties(
                            title=f"Distribution of {feature}",
                            width=400,  # Adjust as needed
                            height=300
                        ).interactive()  # Make the plot interactive

                        st.altair_chart(chart, use_container_width=True)
            else:
                st.write("Please select features to visualize.")

        # --- Random Forest Prediction Section ---
        st.subheader("CAP Regression Prediction by Clinical Stage (Random Forest)")
        st.write("Predict CAP regression by central review using the combined data (Random Forest).")

        # Use the combined_df for prediction
        if combined_df is not None:
            try:
                # Prepare data for prediction
                X_pred_rf = combined_df[features]

                # Make predictions using the trained Random Forest model
                new_predictions_rf = rf_model.predict(X_pred_rf)
                new_predictions_rf = np.round(new_predictions_rf).astype(int)

                # Create a DataFrame to display the predictions, including SimpleID
                prediction_df_rf = pd.DataFrame(
                    {'SimpleID': combined_df['SimpleID'],
                     'Predicted Group': new_predictions_rf,
                     'Actual Group': combined_df['CAP regression by central review']
                     })
                st.subheader("Predictions (Random Forest)")
                st.write("Predicted CAP regression by central review by clinical stage (Random Forest):")
                st.dataframe(prediction_df_rf)

                # Calculate and display accuracy per group
                st.subheader("Accuracy per Group (Random Forest)")
                rf_group_accuracy = rf_comparison_table.groupby('Actual Group')['Correct Prediction'].mean()
                st.dataframe(rf_group_accuracy)

                # Display the confusion matrix
                st.subheader("Confusion Matrix (Random Forest)")
                rf_cm = confusion_matrix(y_test, rf_y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix (Random Forest)')
                st.pyplot(plt)

                # Display ROC curves and AUC scores
                rf_auc = 0
                unique_classes = np.unique(y)
                if len(unique_classes) > 2:
                    # Multiclass ROC
                    rf_y_prob = rf_model.predict_proba(X_test)
                    plt.figure(figsize=(8, 6))
                    for i, cls in enumerate(unique_classes):
                        fpr, tpr, _ = roc_curve(y_test == cls, rf_y_prob[:, i])
                        auc_score = roc_auc_score(y_test == cls, rf_y_prob[:, i])
                        plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {auc_score:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curves (Random Forest)')
                    plt.legend()
                    st.pyplot(plt)
                else:
                    # Binary ROC
                    rf_y_prob = rf_model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, rf_y_prob)
                    rf_auc = roc_auc_score(y_test, rf_y_prob)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f'AUC = {rf_auc:.2f}')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (Random Forest)')
                    plt.legend()
                    st.pyplot(plt)
                st.write(f"Random Forest AUC: {rf_auc:.2f}")

            except Exception as e:
                st.error(f"An error occurred during Random Forest prediction: {e}")

        # --- Neural Network Model ---
        st.subheader("Neural Network Model")
        try:
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define the model
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dropout(0.5),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(len(np.unique(y)), activation='softmax')  # Output layer
            ])

            # Compile the model
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',  # Use sparse for integer labels
                          metrics=['accuracy'])

            # Early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Train the model
            history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0) #Removed the  class_weight=class_weight

            # Evaluate the model
            nn_y_pred = model.predict(X_test_scaled)
            nn_y_pred_classes = np.argmax(nn_y_pred, axis=1)  # Get class predictions

            nn_accuracy = accuracy_score(y_test, nn_y_pred_classes)
            st.write(f"Neural Network Accuracy: {nn_accuracy:.2f}")

            # Display classification report
            st.text("Neural Network Classification Report:")
            st.text(classification_report(y_test, nn_y_pred_classes))

            # Confusion Matrix
            st.subheader("Neural Network Confusion Matrix")
            nn_cm = confusion_matrix(y_test, nn_y_pred_classes)
            plt.figure(figsize=(8, 6))
            sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Neural Network Confusion Matrix')
            st.pyplot(plt)

            # Display ROC and AUC
            nn_auc = 0
            unique_classes = np.unique(y)
            if len(unique_classes) <= 2:
                nn_y_prob = model.predict(X_test_scaled)
                fpr, tpr, _ = roc_curve(y_test, nn_y_prob[:, 1])
                nn_auc = roc_auc_score(y_test, nn_y_prob[:, 1])
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'AUC = {nn_auc:.2f}')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Neural Network ROC Curve')
                plt.legend()
                st.pyplot(plt)
                st.write(f"Neural Network AUC: {nn_auc:.2f}")
            else:
                st.write("ROC/AUC is not calculated for multiclass problems.")

            # Create a DataFrame to compare actual and predicted groups
            nn_comparison_table = pd.DataFrame({
                'SimpleID': X_test.index,
                'Actual Group': y_test.values,
                'Predicted Group': nn_y_pred_classes,
            })
            nn_comparison_table['Correct Prediction'] = nn_comparison_table['Actual Group'] == nn_comparison_table['Predicted Group']

            # Display the table
            st.subheader("Comparison of Actual and Predicted Groups (Neural Network)")
            st.write("Comparison of actual and predicted groups for the testing set (Neural Network):")
            st.dataframe(nn_comparison_table)

            # --- Neural Network Prediction ---
            st.subheader("CAP Regression Prediction by Clinical Stage (Neural Network)")
            st.write("Predict CAP regression by central review using the combined data (Neural Network).")

            # Prepare data for prediction
            X_pred_nn = combined_df[features]
            X_pred_scaled_nn = scaler.transform(X_pred_nn)  # Scale the data

            # Make predictions
            new_predictions_nn = model.predict(X_pred_scaled_nn)
            new_predictions_nn_classes = np.argmax(new_predictions_nn, axis=1)

            # Create a DataFrame to display the predictions
            prediction_df_nn = pd.DataFrame({
                'SimpleID': combined_df['SimpleID'],
                'Predicted Group': new_predictions_nn_classes,
                'Actual Group': combined_df['CAP regression by central review']
            })
            st.subheader("Predictions (Neural Network)")
            st.write("Predicted CAP regression by central review by clinical stage (Neural Network):")
            st.dataframe(prediction_df_nn)

            # Calculate and display accuracy per group
            st.subheader("Accuracy per Group (Neural Network)")
            nn_group_accuracy = nn_comparison_table.groupby('Actual Group')['Correct Prediction'].mean()
            st.dataframe(nn_group_accuracy)
        except Exception as e:
            st.error(f"An error occurred during Neural Network processing: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload all four required files (OTU, Alpha Diversity, Metadata, PCoA) in Excel format.")
