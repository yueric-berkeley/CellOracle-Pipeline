# import libraries
import argparse
import yaml
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import palantir
import celloracle as co
co.check_python_requirements()

# load config
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

# make save folder
save_folder = config["save_folder"]

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
print(adata.obs["condition"].value_counts())

# visualize data
celltype = config["celltype_colname"]
target_gene = config["target_gene"]
sc.pl.umap(adata, color=[celltype,'condition', target_gene], s=30, cmap="viridis", ncols=2, use_raw=False, show=False)
plt.savefig(f"{save_folder}/celltype_condition_targetGene_umap.png")

adata.obs["condition_and_celltype"] = adata.obs["condition"].astype("str") + "_" + adata.obs[celltype].astype("str")
sc.tl.embedding_density(adata, basis='umap', groupby='condition')

for i in adata.obs.condition.unique():
    sc.pl.embedding_density(
        adata, basis='umap', key='umap_density_condition', group=i
    )
    plt.savefig(f"{save_folder}/{i}_umap_density.png")

sc.pl.violin(adata, keys=[target_gene], groupby="condition")
plt.savefig(f"{save_folder}/{target_gene}_violinplot.png")

# Downsample data
n_cells_downsample = config["n_cells_downsample"] 
if adata.shape[0] > n_cells_downsample:
    sc.pp.subsample(adata, n_obs=n_cells_downsample, random_state=123)

print(f"Cell number is :{adata.shape[0]}")
var_gene = adata.var.index[adata.var.variable_gene].values
adata = adata[:, var_gene]
print(f"adata shape : {adata.X.shape}")

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
print(oracle.adata.var.loc[target_gene])

# KNN imputation
## Perform PCA
oracle.perform_PCA()

## Select important PCs
plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
plt.axvline(n_comps, c="k")
plt.savefig(f"{save_folder}/Cumulative_variance_plot.png")
print("n_comps : ", n_comps)
n_comps = min(n_comps, 50)

## KNN imputation
n_cell = oracle.adata.shape[0]
k = int(0.025*n_cell)
oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                      b_maxl=k*4, n_jobs=4)
sc.pl.umap(oracle.adata, layer="raw_count",
           color=[target_gene], s=30, cmap="viridis")
plt.savefig(f"{save_folder}/raw_count_{target_gene}.png")

sc.pl.umap(oracle.adata, layer="imputed_count",
           color=[target_gene], s=30, cmap="viridis")
plt.savefig(f"{save_folder}/imputed_count_{target_gene}.png")

# save oracle object
oracle_object_path = f"./{config['oracle_file_name']}.oracle"
oracle.to_hdf5(oracle_object_path)

# get GRNs
unit = celltype
links = oracle.get_links(cluster_name_for_GRN_unit=unit, alpha=10, verbose_level=10)

links.filter_links(p=0.001, weight="coef_abs", threshold_number=2000)
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
sc.pl.umap(oracle.adata, color=[goi, oracle.cluster_column_name], s=30,
                 layer="imputed_count", use_raw=False, cmap="viridis")
plt.savefig(f"{save_folder}/imputed_{target_gene}_and_cluster_umap.png")

sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.savefig(f"{save_folder}/imputed_count_histogram.png")

# calculate future gene expression after perturbation
oracle.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)

# get transition probability
oracle.estimate_transition_prob(n_neighbors=200,
                                knn_random=True, 
                                sampled_fraction=0.2)

# calculate embedding 
oracle.calculate_embedding_shift(sigma_corr=0.05)

# visualize results
fig, ax = plt.subplots(1, 2,  figsize=[13, 6])
scale = 25

## quiver plot
oracle.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")
oracle.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")
plt.savefig(f"{save_folder}/quiver_plot.png")

## vector field graph
n_grid = 40 
oracle.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
oracle.suggest_mass_thresholds(n_suggestion=12)
plt.savefig(f"{save_folder}/vector_field_graph.png")

min_mass = config['min_mass']
oracle.calculate_mass_filter(min_mass=min_mass, plot=True)
plt.savefig(f"{save_folder}/grid_points_selected.png")

## KO simulation vector field graph
fig, ax = plt.subplots(1, 2,  figsize=[13, 6])
scale_simulation = 7

oracle.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")
oracle.plot_simulation_flow_random_on_grid(scale=scale_simulation, ax=ax[1])
ax[1].set_title(f"Random negative control vector")
plt.savefig(f"{save_folder}/{target_gene}_KO_simulation_vector_field.png")


fig, ax = plt.subplots(figsize=[8, 8])
oracle.plot_cluster_whole(ax=ax, s=10)
oracle.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax, show_background=False)
plt.savefig(f"{save_folder}/{target_gene}_KO_simulation_vector_field_on_cluster_color.png")

# pseudotime simulation
adata = oracle.adata
sc.pp.normalize_per_cell(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1500, flavor='cell_ranger')
sc.pp.pca(adata)
pca_projections = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names)
dm_res = palantir.utils.run_diffusion_maps(pca_projections)
ms_data = palantir.utils.determine_multiscale_space(dm_res)
adata.layers['MAGIC_imputed_data'] = palantir.utils.run_magic_imputation(adata, dm_res)

if config['use_manual_start'] == True:
    start_cell = config['start_cell']
else:
    start_cell = palantir.utils.early_cell(adata, config['start_celltype'], celltype)
pr_res = palantir.core.run_palantir(ms_data, start_cell, num_waypoints=600)

lst = []
for i in list(pr_res.branch_probs.columns):
    lst.append(adata.obs[celltype][i])

pr_res.branch_probs.columns = lst
pr_res.branch_probs = pr_res.branch_probs.loc[:, lst]
oracle.adata.obs = pd.DataFrame(pr_res.pseudotime)

# Visualize pseudotime
from co.applications import Gradient_calculator, Oracle_development_module
gradient = Gradient_calculator(oracle_object=oracle, pseudotime_key=0)
gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
gradient.calculate_mass_filter(min_mass=min_mass, plot=True)
gradient.transfer_data_into_grid(args={"method": "polynomial", "n_poly":3})
gradient.calculate_gradient()
scale_dev = 40
gradient.visualize_results(scale=scale_dev, s=5)
plt.savefig(f"{save_folder}/pseudotime_plots.png")

fig, ax = plt.subplots(figsize=[6, 6])
gradient.plot_dev_flow_on_grid(scale=scale_dev, ax=ax)
plt.savefig(f"{save_folder}/pseudotime_vector_plot.png")

# Save gradient
dev_gradient_file_name = config['dev_gradient_file_name']
gradient.to_hdf5(f"./{dev_gradient_file_name}.gradient")
#gradient = co.load_hdf5("cardiomyocyte.developmentGradient.celloracle.gradient")

# Developmental module
dev = Oracle_development_module()
dev.load_differentiation_reference_data(gradient_object=gradient)
dev.load_perturb_simulation_data(oracle_object=oracle)
dev.calculate_inner_product()
dev.calculate_digitized_ip(n_bins=10)

# Visualize developmental module
scale_simulation = 7
dev.visualize_development_module_layout_0(s=5,
                                          scale_for_simulation=scale_simulation,
                                          s_grid=50,
                                          scale_for_pseudotime=scale_dev,
                                          vm=0.02)
plt.savefig(f"{save_folder}/summary_plot.png")

# perturbation scores with vector field
dev.plot_inner_product_on_grid(vm=0.05, s=50, ax=ax)
dev.plot_simulation_flow_on_grid(scale=scale_simulation, show_background=False, ax=ax)
plt.savefig(f"./{target_gene}_KO_simulation_vector_field_dot_product.png")