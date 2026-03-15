# import libraries
import argparse
import yaml
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import palantir
import celloracle as co
import os
import logging

co.check_python_requirements()



# load config
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

# make save folder

save_folder = config["save_folder"]

# In case the save folder doesn't exist, create directory here

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Created save folder: {save_folder}")
else:
    print(f"Save folder already exists: {save_folder}")

# configure logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{save_folder}/pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# load data
adata = sc.read_h5ad(config["input_data"])
adata.var = pd.DataFrame(
        {'variable_gene': adata.var['vst.variable'].astype(bool)},
        index=adata.var.index
    )

# process data
di = {"Donor": "Donor", "NRpre": "preLVAD HF", "Rpre":"preLVAD HF", "NRpost":"NRpost",
      "Rpost":"Rpost"}
adata.obs = adata.obs.replace({"condition": di})
log.info(adata.obs["condition"].value_counts())

# visualize data
celltype = config["celltype_colname"]
target_gene = config["target_gene"]
sc.pl.umap(adata, color=[celltype,'condition', target_gene], s=30, cmap="viridis", ncols=2, use_raw=False, show=False)
plt.savefig(f"{save_folder}/celltype_condition_targetGene_umap.png")
plt.close()

adata.obs["condition_and_celltype"] = adata.obs["condition"].astype("str") + "_" + adata.obs[celltype].astype("str")
sc.tl.embedding_density(adata, basis='umap', groupby='condition')

for i in adata.obs.condition.unique():
    sc.pl.embedding_density(
        adata, basis='umap', key='umap_density_condition', group=i
    )
    plt.savefig(f"{save_folder}/{i}_umap_density.png")
    plt.close()

sc.pl.violin(adata, keys=[target_gene], groupby="condition")
plt.savefig(f"{save_folder}/{target_gene}_violinplot.png")
plt.close()

# Downsample data
n_cells_downsample = config["n_cells_downsample"] 
if adata.shape[0] > n_cells_downsample:
    sc.pp.subsample(adata, n_obs=n_cells_downsample, random_state=123)

log.info(f"Cell number is :{adata.shape[0]}")
var_gene = adata.var.index[adata.var.variable_gene].values
adata = adata[:, var_gene]
log.info(f"adata shape : {adata.X.shape}")

# load base-GRN data
model = config['base_GRN_model']
if model == "mouse":
    base_GRN = co.data.load_mouse_scATAC_atlas_base_GRN()
elif model == "human":
    base_GRN = co.data.load_human_promoter_base_GRN()

# make oracle object
oracle = co.Oracle()

adata.X = adata.layers["raw_count"].toarray()
sc.pp.normalize_per_cell(adata)

# instantiate oracle object
oracle.import_anndata_as_raw_count(adata=adata,
                                   cluster_column_name=celltype,
                                   embedding_name="X_umap")

# load base GRN data into oracle object
oracle.import_TF_data(TF_info_matrix=base_GRN)
log.info(oracle.adata.var.loc[target_gene])

# KNN imputation
## Perform PCA
oracle.perform_PCA()

## Select important PCs
plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])

# Try to define n_comps and default to 50 if it fails
try:
    n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>config["n_comps_variance_threshold"])[0][0])
except Exception as e:
    log.info("IndexError, defaulting to 50 for n_comps")
    n_comps = config["n_comps_default"]

plt.axvline(n_comps, c="k")
plt.savefig(f"{save_folder}/Cumulative_variance_plot.png")
plt.close()
log.info(f"n_comps : {n_comps}")
n_comps = min(n_comps, config["n_comps_max"])

## KNN imputation
n_cell = oracle.adata.shape[0]
k = int(config["knn_fraction"]*n_cell)
oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*config["knn_b_sight_multiplier"],
                      b_maxl=k*config["knn_b_maxl_multiplier"], n_jobs=config["knn_n_jobs"])
sc.pl.umap(oracle.adata, layer="raw_count",
           color=[target_gene], s=30, cmap="viridis")
plt.savefig(f"{save_folder}/raw_count_{target_gene}.png")
plt.close()

sc.pl.umap(oracle.adata, layer="imputed_count",
           color=[target_gene], s=30, cmap="viridis")
plt.savefig(f"{save_folder}/imputed_count_{target_gene}.png")
plt.close()

# save oracle object
oracle_object_path = f"./{config['oracle_file_name']}.oracle"
oracle.to_hdf5(oracle_object_path)

# get GRNs
unit = celltype
links = oracle.get_links(cluster_name_for_GRN_unit=unit, alpha=config["grn_alpha"], verbose_level=config["grn_verbose_level"])


links.filter_links(p=config["links_p_threshold"],
                   weight=config["links_weight"],
                   threshold_number=config["links_threshold_number"])
links.get_network_score()

links_object_path = f"./{config['grn_links_file_name']}_{unit}.celloracle.links"
links.to_hdf5(file_path= links_object_path)

################################################################
# load data
# oracle = co.load_hdf5(oracle_object_path)
# links = co.load_hdf5(links_object_path)

# make predictive models for simulation
links.filter_links()
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, 
                              use_cluster_specific_TFdict=True)

# check gene expression
goi = target_gene
sc.pl.umap(oracle.adata, color=[goi, oracle.cluster_column_name], s=config["condition_map"],
                 layer="imputed_count", use_raw=False, cmap="viridis")
plt.savefig(f"{save_folder}/imputed_{target_gene}_and_cluster_umap.png")
plt.close()

sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.savefig(f"{save_folder}/imputed_count_histogram.png")
plt.close()

# calculate future gene expression after perturbation
oracle.simulate_shift(perturb_condition={goi: config["perturb_value"]},
                      n_propagation=config["n_propagation"])

# get transition probability
oracle.estimate_transition_prob(n_neighbors=config["n_neighbors_transition"],
                                knn_random=config["knn_random"],
                                sampled_fraction=config["sampled_fraction"])

# calculate embedding 
oracle.calculate_embedding_shift(sigma_corr=config["sigma_corr"])

# visualize results
fig, ax = plt.subplots(1, 2,  figsize=[13, 6])
scale = config["quiver_scale"]

## quiver plot
oracle.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")
oracle.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")
plt.savefig(f"{save_folder}/quiver_plot.png")
plt.close()

## vector field graph
n_grid = config["n_grid"]
min_mass = config["min_mass"]
mass_smooth = config["mass_smooth"]
n_mass_neighbors = config["n_mass_neighbors"]
scale_simulation = config["scale_simulation"]

oracle.calculate_p_mass(smooth=mass_smooth, n_grid=n_grid, n_neighbors=n_mass_neighbors)
oracle.suggest_mass_thresholds(n_suggestion=config["n_mass_suggestions"])
plt.savefig(f"{save_folder}/vector_field_graph.png")
plt.close()

min_mass = config['min_mass']
oracle.calculate_mass_filter(min_mass=min_mass, plot=True)
plt.savefig(f"{save_folder}/grid_points_selected.png")
plt.close()

## KO simulation vector field graph
fig, ax = plt.subplots(1, 2,  figsize=[13, 6])
scale_simulation = config["scale_simulation"]

oracle.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")
oracle.plot_simulation_flow_random_on_grid(scale=scale_simulation, ax=ax[1])
ax[1].set_title(f"Random negative control vector")
plt.savefig(f"{save_folder}/{target_gene}_KO_simulation_vector_field.png")
plt.close()


fig, ax = plt.subplots(figsize=[8, 8])
oracle.plot_cluster_whole(ax=ax, s=config["cluster_point_size"])
oracle.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax, show_background=False)
plt.savefig(f"{save_folder}/{target_gene}_KO_simulation_vector_field_on_cluster_color.png")
plt.close()

# pseudotime simulation
adata = oracle.adata
sc.pp.normalize_per_cell(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=config["n_top_genes_pseudotime"],
                            flavor=config["hvg_flavor"])
sc.pp.pca(adata)
pca_projections = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names)
dm_res = palantir.utils.run_diffusion_maps(pca_projections)
ms_data = palantir.utils.determine_multiscale_space(dm_res)
adata.layers['MAGIC_imputed_data'] = palantir.utils.run_magic_imputation(adata, dm_res)

if config['use_manual_start'] == True:
    start_cell = config['start_cell']
else:
    start_cell = palantir.utils.early_cell(adata, config['start_celltype'], celltype)
pr_res = palantir.core.run_palantir(ms_data, start_cell, num_waypoints=config["num_waypoints"])

lst = []
for i in list(pr_res.branch_probs.columns):
    lst.append(adata.obs[celltype][i])

pr_res.branch_probs.columns = lst
pr_res.branch_probs = pr_res.branch_probs.loc[:, lst]
oracle.adata.obs = pd.DataFrame(pr_res.pseudotime)

# Visualize pseudotime
from celloracle.applications import Gradient_calculator, Oracle_development_module

gradient = Gradient_calculator(oracle_object=oracle, pseudotime_key=0)
gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=n_mass_neighbors)
gradient.calculate_mass_filter(min_mass=min_mass, plot=True)
gradient.transfer_data_into_grid(args={"method": config["magic_smooth_method"],
                                       "n_poly": config["magic_n_poly"]})
gradient.calculate_gradient()
scale_dev = 40
gradient.visualize_results(scale=scale_dev, s=5)
plt.savefig(f"{save_folder}/pseudotime_plots.png")
plt.close()

fig, ax = plt.subplots(figsize=[6, 6])
gradient.plot_dev_flow_on_grid(scale=scale_dev, ax=ax)
plt.savefig(f"{save_folder}/pseudotime_vector_plot.png")
plt.close()

# Save gradient
dev_gradient_file_name = config['dev_gradient_file_name']
gradient.to_hdf5(f"./{dev_gradient_file_name}.gradient")
#gradient = co.load_hdf5("cardiomyocyte.developmentGradient.celloracle.gradient")

# Developmental module
dev = Oracle_development_module()
dev.load_differentiation_reference_data(gradient_object=gradient)
dev.load_perturb_simulation_data(oracle_object=oracle)
dev.calculate_inner_product()
dev.calculate_digitized_ip(n_bins=config["n_digitized_bins"])

# Visualize developmental module
dev.visualize_development_module_layout_0(s=5,
                                          scale_for_simulation=config["scale_simulation"],
                                          s_grid=config["grid_point_size"],
                                          scale_for_pseudotime=scale_dev,
                                          vm=config["vm_layout"])
plt.savefig(f"{save_folder}/summary_plot.png")
plt.close()

# perturbation scores with vector field
dev.plot_inner_product_on_grid(vm=config["vm_inner_product"], s=config["grid_point_size"], ax=ax)
dev.plot_simulation_flow_on_grid(scale=config["scale_simulation"], show_background=False, ax=ax)
plt.savefig(f"{save_folder}/{target_gene}_KO_simulation_vector_field_dot_product.png")
plt.close()
