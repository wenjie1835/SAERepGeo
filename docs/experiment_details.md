Experiment Details
This document provides detailed information about the experimental setup, hyperparameters, and data preparation for the SAERepGeo project, as described in the paper "Sparsification and Reconstruction from the Perspective of Representation Geometry" (Appendix F).
Models and Datasets
Models

GPT2-Small: Layer 11 residual stream, d_model=768, using SAE from gpt2-small-resid-post-v5-32k.
Pythia-70M: Layer 5 residual stream, d_model=512, using SAE from pythia-70m-deduped-res-sm.
Gemma-2-2B: Layer 19 residual stream, d_model=2304, using SAE from sae_bench_gemma-2-2b_topk_width-2pow16_date-1109.
Source: SAEs are loaded via sae-lens library.

Datasets

Concepts: Tested concepts include months, days, elements, color, planets, number, alphabet, phonetic_symbol, and constellations.
Prompts: Stored in data/concept_prompts.json. Prompts are designed to be diverse and information-dense, capturing complete representations (see Table 5 in Appendix B.1 of the paper).
Preprocessing: Tokens irrelevant to concepts (e.g., prepositions, <|endoftext|>) are filtered using concept-specific token lists (see SAEManifoldAnalyzer.concept_filter_tokens).

Experimental Setup
Case 1: Stratified Manifold Analysis

Objective: Analyze the stratified manifold structure of SAE latent representations.
Setup:
Noise levels: Currently implemented with zero noise (0.0). The paper mentions multiple levels (0.0 to 10.0), which can be extended.
Batch size: 10.
Activation threshold: Dynamically set to the 50th percentile of absolute latent activations, with a minimum of 0.01.


Outputs: Cached in outputs/case1_stratified_manifold/ as results_zero_noise.pkl and tokens_dict_zero_noise.pkl.

Case 2: Representation Structure Analysis

Objective: Compare representation structures before and after sparse encoding.
Setup:
Concepts: months, days, elements.
Dimensionality reduction: UMAP with n_components=50, random_state=42, n_neighbors=15, min_dist=0.1, metric='euclidean'.
Clustering: HDBSCAN with min_cluster_size=10, metric='euclidean'.
Metrics: Intrinsic dimensionality (TwoNN), Minimum Spanning Tree Weight (MSTW), Procrustes Disparity.


Outputs: Cached in outputs/case2_representation_structure/ (e.g., residual_cache.pkl, clustering_analysis_results.csv).

Case 3: Intervention Analysis

Objective: Investigate the impact of geometric interventions on reconstruction performance.
Setup:
Concepts: months, days, elements.
Intervention: Gromov-Wasserstein distance (d_GW) and AEDP^-1 optimization.
Hyperparameters:
Scale factors (alpha): [0.5, 0.8, 1.0, 1.2, 1.5].
Number of iterations: 10.
Lambda MSE (lambda_mse): 1.0.


Metrics: Mean Squared Error (MSE), Average Euclidean Distance between Pairs (AEDP).


Outputs: Saved in outputs/case3_intervention_analysis/ as intervention_analysis_results_icd_monge.csv.

Hyperparameter Selection

Batch Size (10): Chosen to balance memory usage and computational efficiency.
UMAP Parameters: Selected based on standard practices for preserving representation structure (see UMAP documentation).
HDBSCAN min_cluster_size (10): Set to ensure meaningful clusters without excessive noise.
Scale Factors: Range chosen to explore a variety of intervention strengths.
Lambda MSE (1.0): Balances reconstruction loss and geometric objectives, as per preliminary experiments.

Data Splits

No explicit train/test splits are used, as the experiments focus on analyzing pre-trained SAEs with fixed prompts.
All prompts in concept_prompts.json are processed in a single pass for each concept and model.

Reproducibility

Random Seeds: Set to 42 for torch.manual_seed and torch.cuda.manual_seed_all to ensure consistent results.
Dependencies: Listed in requirements.txt with pinned versions.
Environment: Reproducible via environment.yml for Conda users.
Instructions: See README.md for setup and execution steps.

Notes

Noise Levels: The current implementation only uses zero noise for Case 1. To match the paper, extend generate_case1_caches to include noise levels [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0].
LLM Usage: Prompts were generated using an LLM (unspecified in the paper), as noted in Section 16 of the paper. Details are in data/README.md.

