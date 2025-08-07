from unsloth import FastLanguageModel
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import json
import os
from pathlib import Path
import argparse
from experiment_config import ExperimentConfig, ExperimentTracker

class ImprovedTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load and configure the model"""
        print(f"Loading model: {self.config.model_name}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
        )
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
    
    def get_prompt_template(self):
        """Get the appropriate prompt template"""
        templates = {
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
        
        return templates.get(self.config.prompt_template, templates["template_v2"])
    
    def prepare_dataset(self, df: pd.DataFrame):
        """Convert dataframe to training format"""
        template = self.get_prompt_template()
        
        instructions = []
        inputs = []
        outputs = []
        
        for _, row in df.iterrows():
            user_prompt = template["user"].format(
                rule=row.rule,
                subreddit=row.subreddit,
                body=row.body,
                positive_example_1=row.positive_example_1,
                positive_example_2=row.positive_example_2,
                negative_example_1=row.negative_example_1,
                negative_example_2=row.negative_example_2
            )
            
            instructions.append(template["system"])
            inputs.append(user_prompt)
            outputs.append("Yes" if row.rule_violation == 1 else "No")
        
        dataset_dict = {
            "instruction": instructions,
            "input": inputs,
            "output": outputs,
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def formatting_prompts_func(self, examples):
        """Format prompts for training"""
        EOS_TOKEN = self.tokenizer.eos_token
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = f"### System:\n{instruction}\n\n### User:\n{input_text}\n\n### Assistant:\n{output}{EOS_TOKEN}"
            texts.append(text)
        return {"text": texts}
    
    def train_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame, fold: int):
        """Train on a single fold"""
        print(f"\nTraining fold {fold}")
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df)
        train_dataset = train_dataset.map(self.formatting_prompts_func, batched=True)
        
        # Setup trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=20,
                optim="adamw_8bit",
                weight_decay=self.config.weight_decay,
                lr_scheduler_type=self.config.lr_scheduler_type,
                seed=3407,
                output_dir=f"outputs_fold_{fold}",
                report_to="none",
                save_steps=self.config.max_steps // 4,  # Save 4 times during training
                save_total_limit=2,
            ),
        )
        
        # Train
        trainer_stats = trainer.train()
        
        # Save model for this fold
        model_path = f"models/{self.config.experiment_name}_fold_{fold}"
        os.makedirs("models", exist_ok=True)
        self.model.save_pretrained_merged(model_path, self.tokenizer)
        
        return trainer_stats
    
    def evaluate_on_validation(self, val_df: pd.DataFrame, model_path: str):
        """Evaluate model on validation set using VLLM"""
        try:
            import vllm
            from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
            
            # Load model for inference
            llm = vllm.LLM(
                model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.7,
                trust_remote_code=True,
                dtype="half",
                max_model_len=self.config.max_seq_length,
            )
            
            tokenizer = llm.get_tokenizer()
            
            # Prepare validation prompts
            template = self.get_prompt_template()
            prompts = []
            
            for _, row in val_df.iterrows():
                user_prompt = template["user"].format(
                    rule=row.rule,
                    subreddit=row.subreddit,
                    body=row.body,
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
                prompts.append(prompt)
            
            # Set up logits processor for Yes/No
            choices = ["No", "Yes"]
            choice_tokens = []
            for choice in choices:
                token_id = tokenizer.encode(choice, add_special_tokens=False)[0]
                choice_tokens.append(token_id)
            
            logits_processor = MultipleChoiceLogitsProcessor(choice_tokens)
            
            # Generate predictions
            responses = llm.generate(
                prompts,
                vllm.SamplingParams(
                    temperature=0,
                    max_tokens=1,
                    logprobs=2,
                    logits_processors=[logits_processor]
                ),
                use_tqdm=True
            )
            
            # Extract probabilities
            probabilities = []
            for response in responses:
                try:
                    logprobs = response.outputs[0].logprobs[0]
                    yes_prob = 0.5  # default
                    
                    if choice_tokens[1] in logprobs:  # "Yes" token
                        yes_logprob = logprobs[choice_tokens[1]].logprob
                        no_logprob = logprobs.get(choice_tokens[0], {"logprob": -10}).logprob
                        
                        # Convert to probabilities
                        yes_prob = np.exp(yes_logprob) / (np.exp(yes_logprob) + np.exp(no_logprob))
                    
                    probabilities.append(yes_prob)
                except:
                    probabilities.append(0.5)  # fallback
            
            # Calculate AUC
            auc_score = roc_auc_score(val_df['rule_violation'], probabilities)
            print(f"Validation AUC: {auc_score:.4f}")
            
            return auc_score
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 0.5  # fallback score

def run_experiment(config_name: str):
    """Run a full experiment with cross-validation"""
    print(f"Running experiment: {config_name}")
    
    # Load config
    config = ExperimentConfig.load(f"experiment_configs/{config_name}.json")
    
    # Load data
    if config.use_augmented_data and os.path.exists("train_augmented.csv"):
        df = pd.read_csv("train_augmented.csv")
        print(f"Using augmented data: {len(df)} samples")
    else:
        df = pd.read_csv("train.csv")
        print(f"Using original data: {len(df)} samples")
    
    # Load CV splits
    with open("cv_splits.json", 'r') as f:
        cv_splits = json.load(f)
    
    # Initialize trainer
    trainer = ImprovedTrainer(config)
    trainer.load_model()
    
    # Run cross-validation
    cv_scores = []
    
    for fold_info in cv_splits:
        fold = fold_info['fold']
        train_indices = fold_info['train_indices']
        val_indices = fold_info['val_indices']
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        
        # Train fold
        trainer.train_fold(train_df, val_df, fold)
        
        # Evaluate fold
        model_path = f"models/{config.experiment_name}_fold_{fold}"
        auc_score = trainer.evaluate_on_validation(val_df, model_path)
        cv_scores.append(auc_score)
        
        print(f"Fold {fold} AUC: {auc_score:.4f}")
    
    # Log results
    tracker = ExperimentTracker()
    tracker.log_experiment(config, cv_scores)
    
    print(f"\nExperiment {config_name} completed!")
    print(f"CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment config name")
    args = parser.parse_args()
    
    run_experiment(args.config)