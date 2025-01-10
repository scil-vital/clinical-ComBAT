import pandas as pd
import numpy as np
import subprocess
import os

from scripts import combat_quick_apply
from scripts import combat_quick_QC
from robust_evaluation_tools.robust_utils import get_site, robust_text, rwp_text

from pptx import Presentation
from pptx.util import Inches

def fit(mov_data_file, ref_data_file, metric, harmonizartion_method, robust, rwp, directory, hc,):
    ###########
    ### fit ###
    ###########
    output_model_filename = (
            get_site(mov_data_file)
            + "."
            + metric
            + "."
            + harmonizartion_method
            + "."
            + robust_text(robust)
            + "."
            + rwp_text(rwp)
            + ".model.csv"
        )
    cmd = (
        "scripts/combat_quick_fit.py"
        + " "
        + ref_data_file
        + " "
        + mov_data_file
        + " --out_dir "
        + directory
        + " --output_model_filename "
        + output_model_filename
        + " --method "
        + harmonizartion_method
        + " --robust "
        + robust
        + " -f "
    )
    if rwp:
        cmd += ' --rwp'
    if hc: 
        cmd += ' --hc'
    subprocess.call(cmd, shell=True)
    return output_model_filename

def apply(mov_data_file, model_filename, metric, harmonizartion_method, robust, rwp, directory):
    output_filename = os.path.join(
            directory,
            get_site(mov_data_file)
            + "."
            + metric
            + "."
            + harmonizartion_method
            + "."
            + robust_text(robust)
            + "."
            + rwp_text(rwp)
            + ".csv"
        )
    combat_quick_apply.apply(mov_data_file, model_filename, output_filename)
    return output_filename

def visualize_harmonization(f, new_f, ref_data_file, directory, all_bundles = False):
    cmd = (
        "scripts/combat_visualize_harmonization.py"
        + " "
        + ref_data_file
        + " "
        + f
        + " "
        + new_f
        + " --out_dir "
        + directory
        + " -f"
    )
    if all_bundles:
        cmd += " --bundles all"
    subprocess.call(cmd, shell=True)

def QC(ref_data, output_filename, output_model_filename):
    return combat_quick_QC.QC(ref_data, output_filename, output_model_filename)
    
def compare_with_compilation(df, compilation_file):
    # Charger le DataFrame COMPILATION
    compilation_df = pd.read_csv(compilation_file)

    # Filtrer les patients de COMPILATION qui sont dans df en utilisant les sid
    common_sids = df['sid'].unique()
    filtered_compilation_df = compilation_df[compilation_df['sid'].isin(common_sids)]

    # Initialiser une liste pour stocker les résultats
    comparison_df = pd.DataFrame()

    # Comparer la différence absolue de la colonne mean par bundle
    for bundle in df['bundle'].unique():
        df_bundle = df[df['bundle'] == bundle]
        compilation_bundle = filtered_compilation_df[filtered_compilation_df['bundle'] == bundle]
        
        # Fusionner les deux DataFrames sur les colonnes 'sid' et 'bundle'
        merged_df = pd.merge(df_bundle, compilation_bundle, on=['sid', 'bundle'], suffixes=('_df', '_compilation'))
        
        # Calculer la différence absolue de la colonne mean
        merged_df['abs_diff_mean'] = (merged_df['mean_df'] - merged_df['mean_compilation']).abs()
        # Calculer la somme des différences absolues pour le bundle
        comparison_df[bundle] = merged_df['abs_diff_mean']
            
    # Ajouter le site au DataFrame
    mean_df = pd.DataFrame(comparison_df.mean()).transpose()

    return mean_df


def create_presentation(directory, method):
    # Create a presentation object
    prs = Presentation()
    
    # Define the subdirectories
    subdirs = ["hc", "NoRobust", "robust", "robust_rwp"]
    # Get the list of images
    images = [img for img in os.listdir(os.path.join(directory, subdirs[0])) if method in img and img.endswith('.png')]
    
    for img in images:
        slide_layout = prs.slide_layouts[5]  # Use a blank slide layout
        slide = prs.slides.add_slide(slide_layout)
        
        for i, subdir in enumerate(subdirs):
            img_path = os.path.join(directory, subdir, img)
            left = Inches(0.5 + (i % 2) * 4.5)  # Positioning images in two columns
            top = Inches(0.2 + (i // 2) * 3.5)  # Positioning images in two rows with more space between rows
            
            # Add text above the image
            text_box = slide.shapes.add_textbox(left, top, width=Inches(4), height=Inches(0.5))
            text_frame = text_box.text_frame
            text_frame.text = subdir
            
            # Add the image
            slide.shapes.add_picture(img_path, left, top + Inches(0.5), width=Inches(4))
    
    # Save the presentation
    prs.save(os.path.join(directory, 'harmonization_results.pptx'))


def compare_distances(directory, site, hc_dists, no_robust_dists, robust_dists, robust_rwp_dists):
    # compare les distances de 4 methodes de harmonization
    comparison_results = {
        "hc_vs_no_robust": (np.array(hc_dists) - np.array(no_robust_dists))/np.array(no_robust_dists)*100,
        "robust_vs_no_robust": (np.array(robust_dists) - np.array(no_robust_dists))/np.array(no_robust_dists)*100,
        "robust_rwp_vs_no_robust": (np.array(robust_rwp_dists) - np.array(no_robust_dists))/np.array(no_robust_dists)*100
    }
    df = pd.DataFrame(comparison_results)
    
    # Calculer le nombre de comparaisons négatives et positives, et les moyennes et médianes
    results = []
    for method in comparison_results.keys():
        negative_values = df[method][df[method] < 0]
        positive_values = df[method][df[method] >= 0]
        
        num_negative = len(negative_values)
        num_positive = len(positive_values)
        
        mean_negative = negative_values.mean() if num_negative > 0 else 0
        mean_positive = positive_values.mean() if num_positive > 0 else 0
        
        median_negative = negative_values.median() if num_negative > 0 else 0
        median_positive = positive_values.median() if num_positive > 0 else 0
        
        mean_difference = df[method].mean()
        
        results.append({
            "site": site,
            "comparaison": method,
            "Nb comp. nég.": num_negative,
            "Nb comp. pos.": num_positive,
            "Moy. tot.": mean_difference,
            "Moy. val. nég.": mean_negative,
            "Moy. val. pos.": mean_positive,
            "Méd. val. nég.": median_negative,
            "Méd. val. pos.": median_positive
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(directory, f"{site}_comparison_results.csv"), index=False)
    return results_df
