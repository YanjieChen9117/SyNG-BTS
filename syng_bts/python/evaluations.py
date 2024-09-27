import umap.umap_ as umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
import os

import importlib.resources as pkg_resources


def heatmap_eval(dat_generated, dat_real):
    r"""
    This function creates a heatmap visualization comparing the generated data and the real data

    Parameters
    -----------
    dat_generated : pd.DataFrame
            the data generated from ApplyExperiment
    dat_real: pd.DataFrame
            the original copy of the data
    
    """
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6),
                            gridspec_kw=dict(width_ratios=[dat_generated.shape[1], dat_real.shape[1]]))

    sns.heatmap(dat_generated, ax=axs[0], cbar=False)
    axs[0].set_title('Generated Data')
    axs[0].set_xlabel('Features')
    axs[0].set_ylabel('Samples')

    sns.heatmap(dat_real, ax=axs[1], cbar=True)
    axs[1].set_title('Real Data')
    axs[1].set_xlabel('Features')
    axs[1].set_ylabel('Samples')

    plt.show()


def UMAP_eval(dat_generated, dat_real, groups_generated, groups_real, legend_pos="top"):
    r"""
    This function creates a UMAP visualization comparing the generated data and the real data

    Parameters
    -----------
    dat_generated : pd.DataFrame
            the data generated from ApplyExperiment
    dat_real: pd.DataFrame
            the original copy of the data
    groups_generated : pd.Series
            the groups generated
    groups_real : pd.Series
            the real groups
    legend_pos : string
            legend location
    
    """
    
    # Filter out features with zero variance in generated data
    non_zero_var_cols = dat_generated.var(axis=0) != 0

    # Use loc to filter columns by the non_zero_var_cols boolean mask
    dat_real = dat_real.loc[:, non_zero_var_cols]
    dat_generated = dat_generated.loc[:, non_zero_var_cols]

    # Combine datasets
    combined_data = np.vstack((dat_real.values, dat_generated.values))  # Ensure conversion to NumPy array if necessary
    combined_groups = np.concatenate((groups_real, groups_generated))
    combined_labels = np.array(['Real'] * dat_real.shape[0] + ['Generated'] * dat_generated.shape[0])

    # Ensure that group labels are hashable and can be used in seaborn plots
    combined_groups = [str(group) for group in combined_groups]  # Convert groups to string if not already

    # UMAP dimensionality reduction
    reducer = UMAP(random_state=42)
    embedding = reducer.fit_transform(combined_data)

    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df['Data Type'] = combined_labels
    umap_df['Group'] = combined_groups

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Data Type', style='Group', palette='bright')
    plt.legend(title='Data Type/Group', loc="best")
    plt.title('UMAP Projection of Real and Generated Data')
    plt.show()

def evaluation(generated_input: str = "BRCASubtypeSel_train.csv", 
               real_input: str = "BRCASubtypeSel_test.csv"):
    r"""
    This method provides preprocessing of the input data prior to creating the visualizations.
    This can also be used as inspiration for other ways of using the above evaluation methods.

    generated_input : string
        the generated dataset; a default set is also provided as an example
    real_input : string
        the real original dataset; a default set is also provided as an example
    
    """
    train_path = "../Case/BRCASubtype/" + generated_input
    if generated_input == 'BRCASubtypeSel_train.csv' and not os.path.exists(path=train_path):
        with pkg_resources.open_text('syng_bts_imports.Case.BRCASubtype', 'BRCASubtypeSel_train.csv') as data_file:
            generated = pd.read_csv(data_file)
    else:
        generated = pd.read_csv(train_path, header = 0)
    test_path = "../Case/BRCASubtype/" + real_input
    if real_input == 'BRCASubtypeSel_test.csv' and not os.path.exists(path=test_path):
        with pkg_resources.open_text('syng_bts_imports.Case.BRCASubtype', 'BRCASubtypeSel_test.csv') as data_file:
            real = pd.read_csv(data_file)
    else:
        real = pd.read_csv(test_path, header = 0)

    # Define the default group level
    level0 = real['groups'].iloc[0]
    level1 = list(set(real['groups']) - set([level0]))

    # Get sample groups
    groups_real = pd.Series(np.where(real['groups'] == "Infiltrating Ductal Carcinoma", "Ductal", "Lobular"))

    groups_generated = pd.Series(np.where(generated.iloc[:, -1] == 1, "Ductal", "Lobular"))

    # Get pure data matrices
    real = real.select_dtypes(include=[np.number])
    real = np.log2(real + 1)
    generated = generated.iloc[:, :real.shape[1]]
    generated.columns = real.columns

    # Select samples for analysis to save running time
    real_ind = list(range(200)) + list(range(len(real) - 200, len(real)))
    generated_ind = list(range(200)) + list(range(len(generated) - 200, len(generated)))

    # Call evaluation functions
    h_subtypes = heatmap_eval(dat_real = real.iloc[real_ind,], dat_generated = generated.iloc[generated_ind,])
    p_umap_subtypes = UMAP_eval(dat_real = real.iloc[real_ind,],
                                dat_generated = generated.iloc[generated_ind,],
                                groups_real = groups_real.iloc[real_ind],
                                groups_generated = groups_generated.iloc[generated_ind],
                                legend_pos = "bottom")

evaluation()