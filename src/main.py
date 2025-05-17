import json
import logging
from utils import setup_logging, setup_device, create_output_dirs
from case1_stratified_manifold import generate_case1_caches
from case2_representation_structure import analyze_representation_structure, generate_case2_cache
from case3_intervention_analysis import run_intervention_analysis
import sys
logger = setup_logging()


def load_concept_prompts():
    try:
        with open("data/concept_prompts.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("concept_prompts.json 文件未找到，请确保文件存在")
        sys.exit(1)


def main():
    device = setup_device()
    output_dir, case3_output_dir = create_output_dirs("case1(优化后的分层流形)")

    model_configs = {
        "GPT2-Small Layer 11": {
            "model_name": "gpt2-small",
            "release": "gpt2-small-resid-post-v5-32k",
            "sae_id": "blocks.11.hook_resid_post",
            "hook": "blocks.11.hook_resid_post",
            "token": None,
            "d_model": 768
        },
        "Pythia-70M Layer 5": {
            "model_name": "EleutherAI/pythia-70m-deduped",
            "release": "pythia-70m-deduped-res-sm",
            "sae_id": "blocks.5.hook_resid_post",
            "hook": "blocks.5.hook_resid_post",
            "token": None,
            "d_model": 512
        },
        "Gemma-2-2B Layer 19": {
            "model_name": "gemma-2-2b",
            "release": "sae_bench_gemma-2-2b_topk_width-2pow16_date-1109",
            "sae_id": "blocks.19.hook_resid_post__trainer_0",
            "hook": "blocks.19.hook_resid_post",
            "token": "", #Enter your own token
            "d_model": 2304
        }
    }

    case2_3_concepts = ['months', 'days', 'elements']
    concept_prompts = load_concept_prompts()

    logger.info("Runing Case 1 analysis")
    results, tokens_dict = generate_case1_caches(model_configs, concept_prompts, output_dir, device)

    logger.info("Generating Case 2 cache")
    generate_case2_cache(model_configs, case2_3_concepts, concept_prompts, output_dir, device)

    logger.info("Runing Case 2 analysis")
    analysis_results_case2, residual_cache = analyze_representation_structure(model_configs, case2_3_concepts,
                                                                              output_dir)

    logger.info("Runing Case 3 analysis")
    analysis_results_case3 = run_intervention_analysis(model_configs, case2_3_concepts, output_dir, tokens_dict,
                                                       lambda_mse=1.0)


if __name__ == "__main__":
    main()

    import json
    import logging
    from utils import setup_logging, setup_device, create_output_dirs
    from case1_stratified_manifold import generate_case1_caches
    from case2_representation_structure import generate_case2_cache, analyze_representation_structure
    from case3_intervention_analysis import run_intervention_analysis

    logger = setup_logging()


    def load_concept_prompts():
        try:
            with open("data/concept_prompts.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Can't find concept_prompts.json")
            sys.exit(1)


    def main():
        device = setup_device()
        base_output_dir, case1_output_dir, case2_output_dir, case3_output_dir = create_output_dirs("outputs")

        model_configs = {
            "GPT2-Small Layer 11": {
                "model_name": "gpt2-small",
                "release": "gpt2-small-resid-post-v5-32k",
                "sae_id": "blocks.11.hook_resid_post",
                "hook": "blocks.11.hook_resid_post",
                "token": None,
                "d_model": 768
            },
            "Pythia-70M Layer 5": {
                "model_name": "EleutherAI/pythia-70m-deduped",
                "release": "pythia-70m-deduped-res-sm",
                "sae_id": "blocks.5.hook_resid_post",
                "hook": "blocks.5.hook_resid_post",
                "token": None,
                "d_model": 512
            },
            "Gemma-2-2B Layer 19": {
                "model_name": "gemma-2-2b",
                "release": "sae_bench_gemma-2-2b_topk_width-2pow16_date-1109",
                "sae_id": "blocks.19.hook_resid_post__trainer_0",
                "hook": "blocks.19.hook_resid_post",
                "token": "", #Enter your own token
                "d_model": 2304
            }
        }

        case2_3_concepts = ['months', 'days', 'elements']
        concept_prompts = load_concept_prompts()

        logger.info("Running Case 1's analyze")
        results, tokens_dict = generate_case1_caches(model_configs, concept_prompts, case1_output_dir, device)

        logger.info("Generating Case 2's cache")
        residual_cache = generate_case2_cache(model_configs, case2_3_concepts, concept_prompts, case2_output_dir,
                                              device)

        logger.info("Running Case 2's analyze")
        analysis_results_case2, residual_cache = analyze_representation_structure(model_configs, case2_3_concepts,
                                                                                  case1_output_dir, case2_output_dir)

        logger.info("Running Case 3's analyze")
        analysis_results_case3 = run_intervention_analysis(model_configs, case2_3_concepts, case1_output_dir,
                                                           case2_output_dir, case3_output_dir, tokens_dict,
                                                           lambda_mse=1.0, device=device)


    if __name__ == "__main__":
        main()