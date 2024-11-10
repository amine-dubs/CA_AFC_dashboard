import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Streamlit Interface
st.title("AFC Dashboard")
st.write("Upload your data file (.csv or .xlsx) for AFC analysis.")

# Function to load the data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a .csv or .xlsx file.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# File upload
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.subheader("Uploaded Data")
        st.write(data.head())  # Display only the first few rows for brevity

        # Define variables and index
        variables = data.columns.tolist()

        # Sidebar for Variable Type Selection and Ordinal Order
        st.sidebar.subheader("Variable Types and Ordinal Order")

        variable_types = []
        ordinal_order = {}
        
        for i, var in enumerate(variables):
            var_type = st.sidebar.selectbox(f"Type of '{var}'", options=['Nominal', 'Ordinal'], key=f"type_{i}")
            variable_types.append('N' if var_type == 'Nominal' else 'O')
            if var_type == 'Ordinal':
                unique_vals = sorted(data[var].unique())
                order = st.sidebar.multiselect(
                    f"Specify the order for '{var}' (from lowest to highest)",
                    options=unique_vals,
                    default=unique_vals
                )
                ordinal_order[i] = order

        # Function to create the original coding table with cumulative ordinal encoding
        def create_coding_table(data, variable_types, ordinal_order={}):
            coded_tables = []
            for j, var_type in enumerate(variable_types):
                column = data.iloc[:, j]
                if var_type == 'N':  # Nominal variable coding
                    categories = np.unique(column)
                    code = np.eye(len(categories))
                elif var_type == 'O':  # Ordinal variable coding with cumulative encoding
                    categories = np.array(ordinal_order.get(j, []))
                    code = np.tril(np.ones((len(categories), len(categories))))  # Cumulative encoding

                # Create column names
                col_names = [f"{variables[j]}_{cat}" for cat in categories]
                # Initialize coding matrix
                Tcod = np.zeros((len(column), len(categories)))
                for i, val in enumerate(column):
                    if val in categories:
                        idx = np.where(categories == val)[0][0]
                        Tcod[i, :] = code[idx]
                    else:
                        st.warning(f"Value '{val}' in column '{variables[j]}' not found in ordinal_order.")
                # Create DataFrame for this variable
                df_coded = pd.DataFrame(Tcod.astype(int), columns=col_names, index=data.index)
                coded_tables.append(df_coded)
            return pd.concat(coded_tables, axis=1)

        # Generate coding table with cumulative encoding for ordinal variables
        coding_table = create_coding_table(data, variable_types, ordinal_order)
        
        # Display the coding table
        st.subheader("Original Coding Table (with Ordinality)")
        st.write(coding_table)

        # Function to create the disjunctive table from the coding table
        def create_disjunctive_table_from_coding(coding_table, variable_types, ordinal_order):
            disjunctive_table = coding_table.copy()
            for j, var_type in enumerate(variable_types):
                if var_type == 'O':  # For ordinal variables only
                    categories = ordinal_order.get(j, [])
                    ordinal_cols = [f"{variables[j]}_{cat}" for cat in categories]
                    for idx in disjunctive_table.index:
                        row = disjunctive_table.loc[idx, ordinal_cols]
                        last_one = row[::-1].eq(1).idxmax()  # Find the last '1'
                        disjunctive_table.loc[idx, ordinal_cols] = 0
                        disjunctive_table.loc[idx, last_one] = 1
            return disjunctive_table

        # Create disjunctive table without cumulative encoding for Burt table calculation
        disjunctive_table = create_disjunctive_table_from_coding(coding_table, variable_types, ordinal_order)
        
        # Display disjunctive table
        st.subheader("Disjunctive Coding Table (for Burt Calculation)")
        st.write(disjunctive_table)

        # Function to create Burt table
        def create_burt_table(disjunctive_table, variables):
            burt_matrix = pd.DataFrame()
            for i, var_i in enumerate(variables):
                var_i_cols = [col for col in disjunctive_table.columns if col.startswith(var_i)]
                row = []
                for j, var_j in enumerate(variables):
                    var_j_cols = [col for col in disjunctive_table.columns if col.startswith(var_j)]
                    block = disjunctive_table[var_i_cols].T @ disjunctive_table[var_j_cols]
                    row.append(block)
                burt_matrix = pd.concat([burt_matrix, pd.concat(row, axis=1)], axis=0)

            burt_matrix.index = burt_matrix.columns = disjunctive_table.columns
            return burt_matrix

        # Generate Burt table
        burt_table = create_burt_table(disjunctive_table, variables)
        
        # Display Burt table with heatmap
        st.subheader("Burt Table (Heatmap)")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(burt_table, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1.5, ax=ax)

        # Draw red lines to separate variable blocks
        num_vars = len(variables)
        var_blocks = [len([col for col in disjunctive_table.columns if col.startswith(var)]) for var in variables]
        lines = np.cumsum(var_blocks)
        for line in lines[:-1]:
            ax.axhline(line, color="red", linewidth=2)
            ax.axvline(line, color="red", linewidth=2)

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig)

        # Extract Contingency Tables directly from the Burt table (submatrices)
        def extract_contingency_tables_from_burt(burt_table, variables):
            contingency_tables = {}
            num_vars = len(variables)
            for i in range(num_vars):
                for j in range(i + 1, num_vars):  # Avoid duplicate pairs
                    # Get the indices for the blocks corresponding to variable pair (i, j)
                    var_i_cols = [col for col in burt_table.columns if col.startswith(variables[i])]
                    var_j_cols = [col for col in burt_table.columns if col.startswith(variables[j])]
                    
                    # Extract the submatrix (block) for the variable pair
                    block = burt_table.loc[var_i_cols, var_j_cols]
                    contingency_tables[f"{variables[i]} vs {variables[j]}"] = block
            return contingency_tables

        # Extract contingency tables from the Burt table
        contingency_tables = extract_contingency_tables_from_burt(burt_table, variables)

        # Display each contingency table in Streamlit
        with st.expander("View Contingency Tables"):
            for pair, table in contingency_tables.items():
                st.write(f"Contingency Table: {pair}")
                st.write(table)
                
                # Display heatmap for each contingency table
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.heatmap(table, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                plt.title(f"Contingency Table: {pair}")
                st.pyplot(fig)

        # Optional: Visualizations like histograms and scatter plots
        with st.expander("Visualizations"):
            st.subheader("Histograms of Encoded Variables")
            for col in disjunctive_table.columns:
                st.write(f"Histogram for {col}")
                st.bar_chart(disjunctive_table[col])

            st.subheader("Scatter Plot (Example Visualization)")
            if disjunctive_table.shape[1] >= 2:
                st.scatter_chart(disjunctive_table.iloc[:, :2])

        # Descriptive statistics for additional insights
        st.subheader("Descriptive Statistics of Encoded Variables")
        st.write(disjunctive_table.describe())

        # Function to calculate resemblance and dissemblance tables
        def calculate_resemblance_dissemblance_tables(disjunctive_table):
            num_individuals = disjunctive_table.shape[0]
            resemblance = np.zeros((num_individuals, num_individuals))
            dissemblance = np.zeros((num_individuals, num_individuals))

            # Calculate resemblance and dissemblance
            for i in range(num_individuals):
                for j in range(num_individuals):
                    if i == j:
                        resemblance[i, j] = 1  # Diagonal is 1 for resemblance
                        dissemblance[i, j] = 0  # Diagonal is 0 for dissemblance
                    else:
                        matches = (disjunctive_table.iloc[i] == disjunctive_table.iloc[j]).sum()
                        mismatches = (disjunctive_table.iloc[i] != disjunctive_table.iloc[j]).sum()
                        total_categories = disjunctive_table.shape[1]

                        # Calculate resemblance and dissemblance values
                        resemblance[i, j] = matches / total_categories
                        dissemblance[i, j] = mismatches / total_categories

            # Convert to DataFrames for easier display
            resemblance_df = pd.DataFrame(resemblance, index=disjunctive_table.index, columns=disjunctive_table.index)
            dissemblance_df = pd.DataFrame(dissemblance, index=disjunctive_table.index, columns=disjunctive_table.index)

            return resemblance_df, dissemblance_df

        # Calculate the resemblance and dissemblance tables
        resemblance_df, dissemblance_df = calculate_resemblance_dissemblance_tables(disjunctive_table)

        # Display in Streamlit
        st.subheader("Resemblance (Similarity) Table")
        st.write(resemblance_df)

        st.subheader("Dissemblance (Dissimilarity) Table")
        st.write(dissemblance_df)

        # Resemblance Heatmap
        st.subheader("Resemblance Heatmap")
        fig_resemblance, ax_resemblance = plt.subplots(figsize=(10, 8))
        sns.heatmap(resemblance_df, annot=False, cmap="Greens", ax=ax_resemblance)
        plt.title("Resemblance (Similarity) Heatmap")
        st.pyplot(fig_resemblance)

        # Dissemblance Heatmap
        st.subheader("Dissemblance Heatmap")
        fig_dissemblance, ax_dissemblance = plt.subplots(figsize=(10, 8))
        sns.heatmap(dissemblance_df, annot=False, cmap="Reds", ax=ax_dissemblance)
        plt.title("Dissemblance (Dissimilarity) Heatmap")
        st.pyplot(fig_dissemblance)

else:
    st.info("Please upload a .csv or .xlsx file to begin.")
