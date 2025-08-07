import os
import pandas as pd
import numpy as np
import vllm
import torch
from typing import List, Dict, Tuple
import json
from pathlib import Path
import argparse
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor

class EnsembleInference:
    def __init__(self, model_paths: List[str], weights: List[float] = None):
        self.model_paths = model_paths
        self.weights = weights or [1.0] * len(model_paths)
        self.weights = np.array(self.weights) / np.sum(self.weights)  # Normalize
        
        print(f"Initializing ensemble with {len(model_paths)} models")
        print(f"Weights: {self.weights}")
        
    def get_prompt_templates(self):
        """Get different prompt templates for diversity"""
        return {
            "template_v1": {
                "system": "You are a Reddit comment classifier. Determine if comments violate specific subreddit rules.",
                "user": """Rule: {rule}
Subreddit: r/{subreddit}

Positive examples (VIOLATE the rule):
1. {positive_example_1}
2. {positive_example_2}

Negative examples (DO NOT violate the rule):  
1. {negative_example_1}
2. {negative_example_2}

Classify this comment: {body}
Answer: """
            },
            
            "template_v2": {
                "system": "Classify Reddit comments as rule violations. Answer only 'Yes' or 'No'.",
                "user": """r/{subreddit} Rule: {rule}

Examples that violate:
• {positive_example_1}
• {positive_example_2}

Examples that don't violate:
• {negative_example_1} 
• {negative_example_2}

Comment: {body}
Violates rule? """
            },
            
            "template_v3": {
                "system": "You are an expert content moderator. Classify if Reddit comments violate subreddit rules.",
                "user": """SUBREDDIT: r/{subreddit}
RULE: {rule}

VIOLATIONS (Yes):
1) {positive_example_1}
2) {positive_example_2}

NON-VIOLATIONS (No):
1) {negative_example_1}
2) {negative_example_2}

COMMENT TO CLASSIFY: {body}

CLASSIFICATION: """
            }
        }
    
    def create_paraphrases(self, comment: str, n_paraphrases: int = 2) -> List[str]:
        """Create simple paraphrases for test-time augmentation"""
        paraphrases = [comment]  # Original
        
        # Simple transformations
        if len(comment.split()) > 3:
            # Shuffle sentence order if multiple sentences
            sentences = comment.split('. ')
            if len(sentences) > 1:
                shuffled = sentences[1:] + [sentences[0]]
                paraphrases.append('. '.join(shuffled))
        
        # Case variations
        if comment.islower():
            paraphrases.append(comment.capitalize())
        elif comment.isupper():
            paraphrases.append(comment.lower())
            
        # Return up to n_paraphrases unique paraphrases
        unique_paraphrases = list(dict.fromkeys(paraphrases))[:n_paraphrases+1]
        return unique_paraphrases
    
    def predict_single_model(self, model_path: str, test_df: pd.DataFrame, 
                           template_name: str = "template_v2", 
                           use_tta: bool = True) -> np.ndarray:
        """Get predictions from a single model"""
        print(f"Loading model: {model_path}")
        
        try:
            llm = vllm.LLM(
                model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                trust_remote_code=True,
                dtype="half",
                enforce_eager=True,
                max_model_len=2048,
                disable_log_stats=True,
            )
            
            tokenizer = llm.get_tokenizer()
            templates = self.get_prompt_templates()
            template = templates[template_name]
            
            # Setup logits processor
            choices = ["No", "Yes"]
            choice_tokens = []
            for choice in choices:
                token_id = tokenizer.encode(choice, add_special_tokens=False)[0]
                choice_tokens.append(token_id)
            
            logits_processor = MultipleChoiceLogitsProcessor(choice_tokens)
            
            all_predictions = []
            
            for idx, row in test_df.iterrows():
                if idx % 100 == 0:
                    print(f"Processing {idx}/{len(test_df)}")
                
                # Get paraphrases for TTA
                if use_tta:
                    comments = self.create_paraphrases(row.body, n_paraphrases=2)
                else:
                    comments = [row.body]
                
                comment_predictions = []
                
                for comment in comments:
                    user_prompt = template["user"].format(
                        rule=row.rule,
                        subreddit=row.subreddit,
                        body=comment,
                        positive_example_1=row.positive_example_1,
                        positive_example_2=row.positive_example_2,
                        negative_example_1=row.negative_example_1,
                        negative_example_2=row.negative_example_2
                    )
                    
                    messages = [
                        {"role": "system", "content": template["system"]},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    # Generate prediction
                    response = llm.generate(
                        [prompt],
                        vllm.SamplingParams(
                            temperature=0,
                            max_tokens=1,
                            logprobs=2,
                            logits_processors=[logits_processor]
                        )
                    )[0]
                    
                    # Extract probability
                    try:
                        logprobs = response.outputs[0].logprobs[0]
                        yes_prob = 0.5  # default
                        
                        if choice_tokens[1] in logprobs:  # "Yes" token
                            yes_logprob = logprobs[choice_tokens[1]].logprob
                            no_logprob = logprobs.get(choice_tokens[0], {"logprob": -10}).logprob
                            
                            # Convert to probabilities
                            yes_prob = np.exp(yes_logprob) / (np.exp(yes_logprob) + np.exp(no_logprob))
                        
                        comment_predictions.append(yes_prob)
                    except:
                        comment_predictions.append(0.5)
                
                # Average across paraphrases
                all_predictions.append(np.mean(comment_predictions))
            
            return np.array(all_predictions)
            
        except Exception as e:
            print(f"Error with model {model_path}: {e}")
            return np.full(len(test_df), 0.5)  # fallback
    
    def predict_ensemble(self, test_df: pd.DataFrame, use_tta: bool = True) -> np.ndarray:
        """Get ensemble predictions"""
        templates = ["template_v2", "template_v1", "template_v3"]
        
        all_model_predictions = []
        
        for i, model_path in enumerate(self.model_paths):
            # Use different templates for diversity
            template = templates[i % len(templates)]
            
            print(f"\nModel {i+1}/{len(self.model_paths)}: {model_path}")
            print(f"Using template: {template}")
            
            predictions = self.predict_single_model(
                model_path, test_df, template_name=template, use_tta=use_tta
            )
            all_model_predictions.append(predictions)
        
        # Weighted ensemble
        ensemble_predictions = np.zeros(len(test_df))
        for i, predictions in enumerate(all_model_predictions):
            ensemble_predictions += self.weights[i] * predictions
        
        return ensemble_predictions
    
    def create_submission(self, test_df: pd.DataFrame, predictions: np.ndarray, 
                         output_file: str = "ensemble_submission.csv"):
        """Create submission file"""
        if 'row_id' not in test_df.columns:
            # Create row_id if it doesn't exist
            test_df = test_df.reset_index()
            test_df['row_id'] = test_df.index
        
        submission = pd.DataFrame({
            'row_id': test_df['row_id'],
            'rule_violation': predictions
        })
        
        submission.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
        
        return submission

def find_best_models(experiments_dir: str = "experiments", top_k: int = 5) -> List[Tuple[str, float]]:
    """Find the best performing models from experiments"""
    results_file = Path(experiments_dir) / "results.json"
    
    if not results_file.exists():
        print("No experiment results found!")
        return []
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Sort by CV score
    sorted_results = sorted(results, key=lambda x: x['cv_mean'], reverse=True)
    
    best_models = []
    for result in sorted_results[:top_k]:
        experiment_name = result['experiment_name']
        cv_score = result['cv_mean']
        
        # Find model paths (assuming they exist)
        model_base_path = f"models/{experiment_name}"
        if os.path.exists(model_base_path + "_fold_0"):
            # Use fold 0 model for simplicity (could ensemble across folds too)
            best_models.append((model_base_path + "_fold_0", cv_score))
    
    return best_models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="test.csv", help="Test data file")
    parser.add_argument("--model_paths", nargs="+", help="Paths to trained models")
    parser.add_argument("--weights", nargs="+", type=float, help="Model weights")
    parser.add_argument("--auto_ensemble", action="store_true", help="Auto-select best models")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top models for auto ensemble")
    parser.add_argument("--use_tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--output", default="ensemble_submission.csv", help="Output file")
    args = parser.parse_args()
    
    # Load test data
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    else:
        test_df = pd.read_csv(args.test_file)
    
    print(f"Loaded test data: {len(test_df)} samples")
    
    # Determine model paths and weights
    if args.auto_ensemble:
        print("Auto-selecting best models...")
        best_models = find_best_models(top_k=args.top_k)
        
        if not best_models:
            print("No trained models found! Please train some models first.")
            return
        
        model_paths = [path for path, score in best_models]
        weights = [score for path, score in best_models]
        
        print("Selected models:")
        for path, score in best_models:
            print(f"  {path}: {score:.4f}")
    else:
        model_paths = args.model_paths
        weights = args.weights
        
        if not model_paths:
            print("Please specify model paths or use --auto_ensemble")
            return
    
    # Create ensemble
    ensemble = EnsembleInference(model_paths, weights)
    
    # Generate predictions
    print("\nGenerating ensemble predictions...")
    predictions = ensemble.predict_ensemble(test_df, use_tta=args.use_tta)
    
    # Evaluate if we have labels (validation mode)
    if 'rule_violation' in test_df.columns:
        auc_score = roc_auc_score(test_df['rule_violation'], predictions)
        print(f"\nEnsemble AUC Score: {auc_score:.4f}")
    
    # Create submission
    submission = ensemble.create_submission(test_df, predictions, args.output)
    
    print(f"\nPrediction statistics:")
    print(f"Mean: {np.mean(predictions):.4f}")
    print(f"Std: {np.std(predictions):.4f}")
    print(f"Min: {np.min(predictions):.4f}")
    print(f"Max: {np.max(predictions):.4f}")

if __name__ == "__main__":
    main()