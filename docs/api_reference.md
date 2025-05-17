API Reference
This document provides detailed API references for the key functions and classes in the SAERepGeo project, organized by module.
Module: sae_manifold_analyzer.py
Class: SAEManifoldAnalyzer
Description: Analyzes the manifold structure of Sparse Autoencoder (SAE) representations.
Initializer:
SAEManifoldAnalyzer(model, sae, target_hook: str, model_name: str)


Parameters:
model (HookedSAETransformer): The language model.
sae (SAE): The sparse autoencoder.
target_hook (str): The hook point in the model (e.g., "blocks.11.hook_resid_post").
model_name (str): Name of the model for logging.


Attributes:
device: Device used ("cuda" or "cpu").
d_model: Model's hidden dimension.
d_sae: SAE's feature dimension.



Method: extract_common_activations
extract_common_activations(tokens_list: List[torch.Tensor], concept_name: str, activation_threshold=0.01, batch_size=10)


Description: Extracts common activations from tokens for a given concept.
Parameters:
tokens_list (List[torch.Tensor]): List of tokenized prompts.
concept_name (str): Concept name (e.g., "months").
activation_threshold (float): Threshold for active features (default: 0.01).
batch_size (int): Batch size for processing (default: 10).


Returns:
all_resid_posts (torch.Tensor or None): Residual stream activations.
common_indices (torch.Tensor or None): Indices of active features.
all_masks (torch.Tensor or None): Mask for filtered tokens.



Module: case1_stratified_manifold.py
Function: generate_case1_caches
generate_case1_caches(model_configs, concept_prompts, case1_output_dir, device)


Description: Generates caches for Case 1 (stratified manifold analysis).
Parameters:
model_configs (dict): Model configurations.
concept_prompts (dict): Concept prompts.
case1_output_dir (str): Output directory for Case 1.
device (str): Device ("cuda" or "cpu").


Returns:
results (dict): Cached latent representations.
tokens_dict (dict): Tokenized prompts.



Module: case2_representation_structure.py
Function: generate_case2_cache
generate_case2_cache(model_configs, case2_3_concepts, concept_prompts, case2_output_dir, device)


Description: Generates residual caches for Case 2.
Parameters:
model_configs (dict): Model configurations.
case2_3_concepts (list): Concepts for Case 2 and 3.
concept_prompts (dict): Concept prompts.
case2_output_dir (str): Output directory for Case 2.
device (str): Device ("cuda" or "cpu").


Returns:
residual_cache (dict): Cached residual representations.



Function: analyze_representation_structure
analyze_representation_structure(model_configs, case2_3_concepts, case1_output_dir, case2_output_dir)


Description: Analyzes representation structure changes for Case 2.
Parameters:
model_configs (dict): Model configurations.
case2_3_concepts (list): Concepts for Case 2 and 3.
case1_output_dir (str): Case 1 output directory.
case2_output_dir (str): Case 2 output directory.


Returns:
analysis_results (list): Analysis results (clusters, dimensions, etc.).
residual_cache (dict): Residual cache.



Module: case3_intervention_analysis.py
Class: CustomSAE
Description: Custom SAE class overriding decoding behavior.Inherits: SAEMethods:

run_time_activation_ln_out(acts): Returns activations unchanged.
decode(latents): Decodes latent representations.

Function: run_intervention_analysis
run_intervention_analysis(model_configs, case2_3_concepts, case1_output_dir, case2_output_dir, case3_output_dir, tokens_dict, lambda_mse=1.0, device)


Description: Performs intervention analysis for Case 3.
Parameters:
model_configs (dict): Model configurations.
case2_3_concepts (list): Concepts for Case 2 and 3.
case1_output_dir (str): Case 1 output directory.
case2_output_dir (str): Case 2 output directory.
case3_output_dir (str): Case 3 output directory.
tokens_dict (dict): Tokenized prompts from Case 1.
lambda_mse (float): Weight for MSE loss (default: 1.0).
device (str): Device ("cuda" or "cpu").


Returns:
analysis_results (list): Intervention analysis results.



Module: utils.py
Function: setup_logging
setup_logging()


Description: Configures logging.
Returns:
logger (logging.Logger): Configured logger.



Function: setup_device
setup_device()


Description: Sets up the computation device.
Returns:
device (str): "cuda" or "cpu".



Function: create_output_dirs
create_output_dirs(base_output_dir)


Description: Creates output directories for all cases.
Parameters:
base_output_dir (str): Base output directory.


Returns:
base_output_dir (str): Base output directory.
case1_dir (str): Case 1 output directory.
case2_dir (str): Case 2 output directory.
case3_dir (str): Case 3 output directory.



