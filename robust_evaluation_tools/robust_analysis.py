import pandas as pd
import os

def calculate_precision_by_bundle(metrics_compilation_df):
    # ANALYZE BEST BUNDLES for F1, precision etc
    """
    Calcule le score de précision par bundle.

    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données avec les colonnes 'bundle' et 'is_malade'.

    Returns:
    pd.DataFrame: Un DataFrame avec les bundles et leurs scores de précision respectifs.
    """
    total = pd.DataFrame()

    for bundle_column in metrics_compilation_df.columns:
        if bundle_column in ['site','metric','num_patients','disease_ratio','num_diseased']:
            continue # Skip non-numeric columns
        bundle_df = metrics_compilation_df[[bundle_column, 'metric']].copy()
        grouped_df = bundle_df.groupby(['metric']).mean().reset_index()
        grouped_df.set_index('metric', inplace=True)
        total = pd.concat([total, grouped_df.T])
        
    return total


# COUNT BUNDLES PER OUTLIERS
def count_bundles_per_outliers(df):
    """
    Analyze outliers in the DataFrame and calculate the percentage of SIDs with a certain number of occurrences.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'sid', 'is_outlier', and 'is_sick' columns.

    Returns:
    dict: A dictionary with the percentage of SIDs with a certain number of occurrences for sick and healthy groups.
    """
    
    # Count the number of occurrences of each SID
    # Count the number of occurrences of each combination of SID and site
    sid_counts = df.groupby(['sid', 'site', 'is_malade']).size().reset_index(name='count_bundle')
    
    # Divide the dataset into two groups: sick and healthy
    sick_sids = sid_counts[sid_counts['is_malade'] == 1]
    healthy_sids = sid_counts[sid_counts['is_malade'] == 0]
    
    # Calculate the percentage of SIDs with a certain number of occurrences for sick group
    sick_counts = sick_sids.groupby(['count_bundle']).size().reset_index(name='prct_occurence')
    sick_counts['prct_occurence'] = sick_counts['prct_occurence']/sick_counts['prct_occurence'].sum()*100
    # Calculate the percentage of SIDs with a certain number of occurrences for healthy group
    healthy_counts = healthy_sids.groupby(['count_bundle']).size().reset_index(name='prct_occurence')
    healthy_counts['prct_occurence'] = healthy_counts['prct_occurence']/healthy_counts['prct_occurence'].sum()*100

    total = pd.merge(sick_counts, healthy_counts, on=['count_bundle'], suffixes=('_sick', '_healthy'))
    
    return total

# Example usage
#bundles_per_outliers = count_bundles_per_outliers(pd.read_csv(os.path.join(MAINFOLDER, robust_method, "outliers_compilation.csv")))
#bundles_per_outliers.head(10)