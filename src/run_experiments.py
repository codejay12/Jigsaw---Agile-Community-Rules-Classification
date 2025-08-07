#!/usr/bin/env python3
"""
Main pipeline to run all experiments and generate final ensemble predictions
"""

import os
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from experiment_config import create_experiment_configs, ExperimentTracker
from data_augmentation import DataAugmentor

def setup_environment():
    """Setup directories and initial data"""
    print("Setting up experiment environment...")
    
    # Create directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiment_configs", exist_ok=True)
    
    # Create experiment configurations
    configs = create_experiment_configs()
    for config in configs:
        config.save(f"experiment_configs/{config.experiment_name}.json")
    
    print(f"Created {len(configs)} experiment configurations")
    return configs

def prepare_data():
    """Prepare augmented data and CV splits"""
    print("\nPreparing data...")
    
    if not os.path.exists("train_augmented.csv") or not os.path.exists("cv_splits.json"):
        print("Running data augmentation...")
        subprocess.run(["python", "data_augmentation.py"], check=True)
    else:
        print("Augmented data already exists")
    
    # Load and print data statistics
    df_orig = pd.read_csv("train.csv")
    df_aug = pd.read_csv("train_augmented.csv")
    
    print(f"Original data: {len(df_orig)} samples")
    print(f"Augmented data: {len(df_aug)} samples ({len(df_aug)/len(df_orig):.1f}x increase)")

def run_single_experiment(config_name: str):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {config_name}")
    print(f"{'='*60}")
    
    try:
        subprocess.run([
            "python", "improved_training.py", 
            "--config", config_name
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment {config_name} failed: {e}")
        return False

def run_all_experiments(configs, skip_existing=True):
    """Run all experiments"""
    print("\nRunning experiments...")
    
    # Load existing results to skip completed experiments
    tracker = ExperimentTracker()
    completed_experiments = {result['experiment_name'] for result in tracker.results}
    
    successful_experiments = []
    failed_experiments = []
    
    for config in configs:
        if skip_existing and config.experiment_name in completed_experiments:
            print(f"Skipping {config.experiment_name} (already completed)")
            successful_experiments.append(config.experiment_name)
            continue
        
        success = run_single_experiment(config.experiment_name)
        if success:
            successful_experiments.append(config.experiment_name)
        else:
            failed_experiments.append(config.experiment_name)
    
    print(f"\nExperiment Summary:")
    print(f"Successful: {len(successful_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print(f"Failed experiments: {failed_experiments}")
    
    return successful_experiments, failed_experiments

def generate_ensemble_predictions():
    """Generate final ensemble predictions"""
    print("\n" + "="*60)
    print("GENERATING ENSEMBLE PREDICTIONS")
    print("="*60)
    
    # Show leaderboard
    tracker = ExperimentTracker()
    tracker.print_leaderboard(top_k=10)
    
    # Generate ensemble predictions on test set
    print("\nGenerating ensemble predictions...")
    subprocess.run([
        "python", "ensemble_inference.py",
        "--test_file", "test.csv",
        "--auto_ensemble",
        "--top_k", "3",
        "--use_tta",
        "--output", "final_submission.csv"
    ])
    
    print("Final submission created: final_submission.csv")

def analyze_results():
    """Analyze and summarize all experiment results"""
    print("\n" + "="*60)
    print("EXPERIMENT ANALYSIS")
    print("="*60)
    
    tracker = ExperimentTracker()
    
    if not tracker.results:
        print("No experiment results found!")
        return
    
    # Basic statistics
    scores = [result['cv_mean'] for result in tracker.results]
    print(f"Total experiments: {len(tracker.results)}")
    print(f"Best score: {max(scores):.4f}")
    print(f"Worst score: {min(scores):.4f}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Score std: {np.std(scores):.4f}")
    
    # Show top performers
    tracker.print_leaderboard(top_k=5)
    
    # Analyze what works
    print("\nTop performing configurations:")
    sorted_results = sorted(tracker.results, key=lambda x: x['cv_mean'], reverse=True)
    
    for i, result in enumerate(sorted_results[:3]):
        print(f"\n{i+1}. {result['experiment_name']} (CV: {result['cv_mean']:.4f})")
        config = result['config']
        print(f"   - LoRA rank: {config['lora_r']}")
        print(f"   - Learning rate: {config['learning_rate']}")
        print(f"   - Max steps: {config['max_steps']}")
        print(f"   - Augmented data: {config['use_augmented_data']}")

def main():
    """Main execution pipeline"""
    print("JIGSAW COMPETITION EXPERIMENT PIPELINE")
    print("="*60)
    
    # Setup
    configs = setup_environment()
    
    # Prepare data
    prepare_data()
    
    # Run experiments
    successful, failed = run_all_experiments(configs, skip_existing=True)
    
    if not successful:
        print("No successful experiments to analyze!")
        return
    
    # Analyze results
    analyze_results()
    
    # Generate final ensemble
    generate_ensemble_predictions()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED!")
    print("="*60)
    print("Next steps:")
    print("1. Check 'final_submission.csv' for submission")
    print("2. Review experiment results in experiments/results.json")
    print("3. Consider running additional experiments based on insights")

if __name__ == "__main__":
    main()