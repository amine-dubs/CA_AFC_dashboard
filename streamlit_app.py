import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize sample data
data = np.array([
    ["Oui", "Souvent"],
    ["Non", "Jamais"],
    ["Oui", "Rarement"],
    ["Oui", "Toujours"],
    ["Non", "Jamais"],
    ["Oui", "Souvent"],
    ["Oui", "Souvent"],
    ["Oui", "Toujours"],
    ["Non", "Jamais"],
    ["Oui", "Souvent"],
    ["Oui", "Rarement"],
    ["Oui", "Souvent"]
])

variables = ["Réponse1", "Réponse2"]
index = [f"Ind{i+1}" for i in range(data.shape[0])]
table = pd.DataFrame(data, columns=variables, index=index)

# Function to create the original coding table with cumulative ordinal encoding
def create_coding_table(a, variable_types, ordinal_order={}):
    coded_tables = []
    for j, var_type in enumerate(variable_types):
        v1 = np.array(a[:, j])

        if var_type == 'N':  # Nominal variable coding
            un = np.unique(v1)
            code = np.eye(len(un))
        elif var_type == 'O':  # Ordinal variable coding with cumulative encoding
            un = np.array(ordinal_order.get(j, []))
            code = np.tril(np.ones((len(un), len(un))))  # Cumulative encoding for ordinality

        H = [f"{variables[j]} {label}" for label in un]
        Tcod = np.zeros((len(v1), len(un)))

        for i, val in enumerate(v1):
            Tcod[i, :] = code[np.where(un == val)[0][0]]

        h = [f'Ind{n+1}' for n in range(a.shape[0])]
        coded_tables.append(pd.DataFrame(Tcod.astype(int), columns=H, index=h))

    return pd.concat(coded_tables, axis=1)

# Variable types and ordinal order
variable_types = ['N', 'O']
ordinal_order = {1: ['Jamais', 'Rarement', 'Souvent', 'Toujours']}

# Generate coding table with cumulative encoding for ordinal variables
coding_table = create_coding_table(data, variable_types, ordinal_order)

# Function to create a disjunctive table (remove ordinality for Burt table calculation)
def create_disjunctive_table(a, variable_types, ordinal_order={}):
    disjunctive_tables = []
    for j, var_type in enumerate(variable_types):
        v1 = np.array(a[:, j])

        if var_type == 'N':  # Nominal variable coding
            un = np.unique(v1)
            code = np.eye(len(un))
        elif var_type == 'O':  # Treat ordinal variable as nominal (no cumulative encoding)
            un = np.array(ordinal_order.get(j, []))
            code = np.eye(len(un))

        H = [f"{variables[j]} {label}" for label in un]
        Tcod = np.zeros((len(v1), len(un)))

        for i, val in enumerate(v1):
            Tcod[i, :] = code[np.where(un == val)[0][0]]

        h = [f'Ind{n+1}' for n in range(a.shape[0])]
        disjunctive_tables.append(pd.DataFrame(Tcod.astype(int), columns=H, index=h))

    return pd.concat(disjunctive_tables, axis=1)

# Create disjunctive table without cumulative encoding for Burt table calculation
disjunctive_table = create_disjunctive_table(data, variable_types, ordinal_order)

# Burt table calculation using disjunctive coding table
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

# Streamlit Interface
st.title("AFC Dashboard")
st.write("Dashboard for Coding and Burt Tables of AFC analysis.")

# Display Coding Table with Ordinality
st.subheader("Original Coding Table (with Ordinality)")
st.write(coding_table)

# Display Disjunctive Coding Table (for Burt calculation)
st.subheader("Disjunctive Coding Table")
st.write(disjunctive_table)

# Display Burt Table with grouped red lines
st.subheader("Burt Table")
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

# Optional: Histogram for each category
st.subheader("Histograms of Encoded Variables")
for col in disjunctive_table.columns:
    st.write(f"Histogram for {col}")
    st.bar_chart(disjunctive_table[col])

# Optional: Scatter plot example (if sufficient columns are available)
st.subheader("Scatter Plot (Example Visualization)")
if disjunctive_table.shape[1] >= 2:
    st.scatter_chart(disjunctive_table.iloc[:, :2])
