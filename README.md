Sparse Autoencoders (SAEs) have emerged as a predominant tool in mechanistic interpretability, aiming to identify interpretable monosemantic features. However, how does sparse encoding organize the representations of activation vector from language models? What is the relationship between this organizational paradigm and feature disentanglement as well as reconstruction performance? To address these questions, we propose the SAEMA, which validates the stratified structure of the representation by observing the variability of the rank of the symmetric semipositive definite (SSPD) matrix corresponding to the modal tensor unfolded along the latent tensor with the level of noise added to the residual stream. To systematically investigate how sparse encoding alters representational structures, we define local and global representations, demonstrating that they amplify inter-feature distinctions by merging similar semantic features and introducing additional degrees of freedom. Furthermore, we intervene the global representation from an optimization perspective, proving a significant causal relationship between their separability and the reconstruction performance. This study explains the principles of sparsity from the perspective of representational geometry and demonstrates the impact of changes in representational structure on reconstruction performance. Particularly emphasizes the necessity of understanding representations and incorporating representational constraints, providing empirical references for developing new interpretable tools and improving SAEs.


Case 1: Stratified Manifold Analysis - Analyzes the manifold structure of SAE latent representations.

Case 2: Representation Structure Analysis - Examines changes in local and global representation structures.

Case 3: Intervention Analysis - Investigates the impact of geometric interventions on reconstruction performance.




Install dependencies:
pip install -r requirements.txt


(Optional) Set up a Conda environment:
conda env create -f environment.yml
conda activate saerepgeo



Usage

Prepare the concept_prompts.json file in the data/ directory (see data/README.md for format).
Run the main script:python src/main.py



This will execute all experiments sequentially, saving outputs to the outputs/ directory.
Directory Structure

src/: Source code for experiments and utilities.
data/: Input data (e.g., concept_prompts.json).
outputs/: Experiment results and caches.
tests/: Unit tests for each case.
docs/: Detailed documentation and API references.

Reproducibility
The code is designed to be fully reproducible, as detailed in Appendix F of the paper. Ensure you have:

A compatible GPU (CUDA-enabled) or CPU.
The concept_prompts.json file with prompts for concepts like months, days, and elements.
Sufficient disk space for output caches.

License
This project is licensed under the MIT License. See LICENSE for details.
Contact
For issues or questions, please open an issue on GitHub or contact the anonymous authors via the NeurIPS submission portal.
