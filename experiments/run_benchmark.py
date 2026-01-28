#!/usr/bin/env python3
"""
Main benchmark execution script for MARAG
Runs comprehensive evaluation across all models and languages
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from src.marag.core.router import AdaptiveRouter
from src.marag.indices.index_factory import IndexFactory
from src.marag.evaluation.benchmark_evaluator import BenchmarkEvaluator
from src.utils.logger import setup_logger
from src.utils.data_loader import MuLRDataLoader

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run MARAG benchmark evaluation")
    parser.add_argument("--config", type=str, default="configs/marag_full.yaml",
                       help="Configuration file path")
    parser.add_argument("--languages", type=str, nargs="+",
                       default=["sw", "bn", "am", "es"],
                       help="Languages to evaluate")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("MARAG_Benchmark", log_level)
    
    logger.info("=" * 60)
    logger.info("MARAG Benchmark Evaluation")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Languages: {', '.join(args.languages)}")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    logger.info("Initializing MARAG components...")
    
    # Load language profiles
    language_profiles = {}
    for lang_code in args.languages:
        profile_path = Path(f"data/language_profiles/{lang_code}_profile.json")
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                language_profiles[lang_code] = json.load(f)
        else:
            logger.warning(f"Profile not found for {lang_code}, using defaults")
    
    # Initialize router
    router = AdaptiveRouter(
        input_dim=config['router']['input_features'],
        hidden_dims=config['router']['hidden_layers']
    )
    
    # Load pretrained router weights if available
    router_weights_path = Path("models/pretrained/router_model/model_weights.bin")
    if router_weights_path.exists():
        router.load_state_dict(torch.load(router_weights_path))
        logger.info("Loaded pretrained router weights")
    
    # Initialize index factory
    index_factory = IndexFactory(config['retrieval'])
    
    # Load benchmark data
    logger.info("Loading MuLR benchmark data...")
    data_loader = MuLRDataLoader()
    benchmark_data = {}
    
    for lang_code in args.languages:
        benchmark_data[lang_code] = data_loader.load_language_data(lang_code, split="test")
        logger.info(f"  Loaded {len(benchmark_data[lang_code])} samples for {lang_code}")
    
    # Initialize evaluator
    evaluator = BenchmarkEvaluator(
        router=router,
        index_factory=index_factory,
        language_profiles=language_profiles,
        config=config
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_all(benchmark_data)
    
    # Save results
    results_file = output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Generate summary report
    summary = evaluator.generate_summary_report(results)
    print("\n" + "=" * 60)
    print("MARAG BENCHMARK SUMMARY")
    print("=" * 60)
    print(summary)
    
    # Save summary
    summary_file = output_dir / "benchmark_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    logger.info(f"Summary saved to {summary_file}")
    logger.info("Benchmark evaluation completed successfully!")

if __name__ == "__main__":
    main()
