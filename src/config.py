import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path

@dataclass
class ExperimentConfig:
    # Model configuration
    model_name: str = "unsloth/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    
    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = None
    
    # Training configuration
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 500
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    
    
    # Data configuration
    use_augmented_data: bool = True
    augmentation_factor: int = 3
    
    # Prompt configuration
    prompt_template: str = "template_v2"
    
    # Experiment metadata
    experiment_name: str = "baseline"
    description: str = ""
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                 "gate_proj", "up_proj", "down_proj"]
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

class ExperimentTracker:
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        
        self.results_file = self.experiments_dir / "results.json"
        self.results = self.load_results()
    
    def load_results(self) -> List[Dict]:
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_results(self):
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def log_experiment(self, config: ExperimentConfig, cv_scores: List[float], 
                      test_score: float = None, additional_metrics: Dict = None):
        experiment_result = {
            "experiment_name": config.experiment_name,
            "description": config.description,
            "config": asdict(config),
            "cv_scores": cv_scores,
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "test_score": test_score,
            "additional_metrics": additional_metrics or {},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        self.results.append(experiment_result)
        self.save_results()
        
        print(f"Logged experiment: {config.experiment_name}")
        print(f"CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        if test_score:
            print(f"Test Score: {test_score:.4f}")
    
    def get_best_experiment(self) -> Dict:
        if not self.results:
            return None
        return max(self.results, key=lambda x: x['cv_mean'])
    
    def print_leaderboard(self, top_k: int = 10):
        sorted_results = sorted(self.results, key=lambda x: x['cv_mean'], reverse=True)
        
        print("\n" + "="*80)
        print("EXPERIMENT LEADERBOARD")
        print("="*80)
        print(f"{'Rank':<5} {'Experiment':<25} {'CV Score':<12} {'Test Score':<12} {'Description':<20}")
        print("-"*80)
        
        for i, result in enumerate(sorted_results[:top_k]):
            rank = i + 1
            name = result['experiment_name'][:24]
            cv_score = f"{result['cv_mean']:.4f}±{result['cv_std']:.3f}"
            test_score = f"{result.get('test_score', 'N/A')}"
            if test_score != 'N/A':
                test_score = f"{float(test_score):.4f}"
            desc = result['description'][:19]
            
            print(f"{rank:<5} {name:<25} {cv_score:<12} {test_score:<12} {desc:<20}")

def create_experiment_configs() -> List[ExperimentConfig]:
    """Create all experiment configurations"""
    configs = []
    
    # Baseline experiment
    configs.append(ExperimentConfig(
        experiment_name="baseline_v1",
        description="Current approach",
        max_steps=60,
        lora_r=16,
        use_augmented_data=False
    ))
    
    # Improved baseline with more training
    configs.append(ExperimentConfig(
        experiment_name="baseline_v2", 
        description="More training steps",
        max_steps=500,
        lora_r=16,
        use_augmented_data=False
    ))
    
    # With data augmentation
    configs.append(ExperimentConfig(
        experiment_name="augmented_v1",
        description="With 3x data augmentation", 
        max_steps=500,
        lora_r=32,
        use_augmented_data=True,
        augmentation_factor=3
    ))
    
    # Higher LoRA rank
    configs.append(ExperimentConfig(
        experiment_name="high_rank_v1",
        description="LoRA rank 64",
        max_steps=500,
        lora_r=64,
        lora_alpha=64,
        use_augmented_data=True
    ))
    
    # Different learning rates
    for lr in [1e-4, 5e-4]:
        configs.append(ExperimentConfig(
            experiment_name=f"lr_{lr:.0e}",
            description=f"Learning rate {lr}",
            learning_rate=lr,
            max_steps=500,
            use_augmented_data=True
        ))
    
    # Longer training
    configs.append(ExperimentConfig(
        experiment_name="long_training",
        description="1000 training steps",
        max_steps=1000,
        use_augmented_data=True,
        learning_rate=1e-4  # Lower LR for longer training
    ))
    
    return configs

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Create experiment configs
    configs = create_experiment_configs()
    
    # Save configs
    config_dir = Path("experiment_configs")
    config_dir.mkdir(exist_ok=True)
    
    for config in configs:
        config.save(config_dir / f"{config.experiment_name}.json")
    
    print(f"Created {len(configs)} experiment configurations")
    
    # Initialize tracker
    tracker = ExperimentTracker()
    
    # Example of logging a result
    # tracker.log_experiment(
    #     configs[0], 
    #     cv_scores=[0.55, 0.57, 0.56, 0.58, 0.55],
    #     test_score=0.60
    # )