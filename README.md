## [Sparsification and Reconstruction from the perspective of Representation Geometry](https://arxiv.org/abs/2505.22506)

This repository contains code for the paper, it's implements experiments analyzing Sparse Autoencoder (SAE) representation geometry in language models via three cases:

Sparse Autoencoders (SAEs) have emerged as a predominant tool in mechanistic interpretability, aiming to identify interpretable monosemantic features. However, how does sparse encoding organize the representations of activation vector from language models? What is the relationship between this organizational paradigm and feature disentanglement as well as reconstruction performance? To address these questions, we propose the SAEMA, which validates the stratified structure of the representation by observing the variability of the rank of the symmetric semipositive definite (SSPD) matrix corresponding to the modal tensor unfolded along the latent tensor with the level of noise added to the residual stream. To systematically investigate how sparse encoding alters representational structures, we define local and global representations, demonstrating that they amplify inter-feature distinctions by merging similar semantic features and introducing additional degrees of freedom. Furthermore, we intervene the global representation from an optimization perspective, proving a significant causal relationship between their separability and the reconstruction performance. This study explains the principles of sparsity from the perspective of representational geometry and demonstrates the impact of changes in representational structure on reconstruction performance. Particularly emphasizes the necessity of understanding representations and incorporating representational constraints, providing empirical references for developing new interpretable tools and improving SAEs.


1. **Case 1**: Stratified Manifold Analysis
2. **Case 2**: Representation Structure Analysis
3. **Case 3**: Intervention Analysis


## Reproducibility Guide

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SAERepGeo/SAERepGeo.git
   cd SAERepGeo
   ```

2. Set up the Conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate saerepgeo
   ```

3. Install the project:

   ```bash
   pip install .
   ```


```bash
pip install torch==2.5.1+cu121
```

### Data Preparation

Prepare `data/concept_prompts.json` with prompts for concepts (`months`, `days`, `elements`, etc.):

- **Generate**: Use an LLM to create diverse prompts (see `data/README.md` for format).

- **Example**:

  ```json
  {
    "months": ["January is the first month.", ...],
    "days": ["Monday is the first workday.", ...],
    "elements": ["Hydrogen is the first element.", ...]
  }
  ```


### Running Experiments

Run all experiments (Case 1, 2, 3):

```bash
python src/main.py
```

- **Outputs** (in `outputs/`):
  - Case 1: `case1_stratified_manifold/{results,tokens_dict}_zero_noise.pkl`
  - Case 2: `case2_representation_structure/{residual_cache.pkl, clustering_analysis_results.csv, ...}`
  - Case 3: `case3_intervention_analysis/intervention_analysis_results_icd_monge.csv`

### Verifying Results

- **Case 1**: Check cached latents in `results_zero_noise.pkl` (zero noise; extend for multi-noise, see paper Page 5).

- **Case 2**: Compare `clustering_analysis_results.csv` with paper Table 1 (Page 6).

- **Case 3**: Verify `intervention_analysis_results_icd_monge.csv` for MSE/AEDP correlations (Page 8).

- Run tests:

  ```bash
  python -m unittest discover tests
  ```

### Troubleshooting

- **Missing** `concept_prompts.json`: Generate or download (see Data Preparation).
- **Dependency Issues**: Recreate environment (`conda env remove -n saerepgeo; conda env create -f environment.yml`).
- **GPU Errors**: Use CPU (`device = "cpu"` in `utils.py`) or install CUDA 12.1.
- **Contact**: Open GitHub issue or contact authors via NeurIPS portal.

## Directory Structure

- `src/`: Source code (`main.py`, experiment modules).
- `data/`: Input data (`concept_prompts.json`).
- `outputs/`: Experiment results.
- `tests/`: Unit tests.
- `docs/`: Documentation (`experiment_details.md`, `api_reference.md`).

## License

MIT License (see `LICENSE`).

### Citation

If you find this work useful, please consider citing our paper:

@misc{sun2025sparsificationreconstructionperspectiverepresentation,\
title={Sparsification and Reconstruction from the Perspective of Representation Geometry},\
author={Wenjie Sun and Bingzhe Wu and Zhile Yang and Chengke Wu},\
year={2025},\
eprint={2505.22506},\
archivePrefix={arXiv},\
primaryClass={cs.LG},\
url={https://arxiv.org/abs/2505.22506},

}
