import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import openai
import time
import json
from typing import List, Dict
import random

class DataAugmentor:
    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key
        
    def paraphrase_comment(self, comment: str, rule: str) -> List[str]:
        """Generate paraphrases of a comment while maintaining its rule violation status"""
        prompt = f"""
        Paraphrase the following Reddit comment in 3 different ways. Keep the same meaning and tone, but vary the wording:

        Original comment: "{comment}"
        Context rule: "{rule}"

        Return 3 paraphrases as a JSON list:
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            
            paraphrases = json.loads(response.choices[0].message.content)
            return paraphrases[:3]  # Ensure we get exactly 3
        except:
            # Fallback: simple transformations
            return [
                comment.replace("I think", "I believe"),
                comment.replace("really", "very"),
                comment.capitalize() if not comment[0].isupper() else comment.lower()
            ][:3]
    
    def generate_hard_negatives(self, rule: str, subreddit: str, positive_examples: List[str]) -> List[str]:
        """Generate comments that are close to violations but don't cross the line"""
        prompt = f"""
        Rule: "{rule}"
        Subreddit: r/{subreddit}
        
        These comments VIOLATE the rule:
        {chr(10).join(f"- {ex}" for ex in positive_examples)}
        
        Generate 3 comments that are borderline but DO NOT violate this rule. They should be close to the line but clearly acceptable.
        Return as JSON list:
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=400
            )
            
            hard_negatives = json.loads(response.choices[0].message.content)
            return hard_negatives[:3]
        except:
            # Fallback: modified positive examples
            return [ex.replace("buy", "consider") for ex in positive_examples[:3]]
    
    def augment_underrepresented_rules(self, df: pd.DataFrame, target_samples_per_rule: int = 1000) -> pd.DataFrame:
        """Generate more examples for rules with fewer samples"""
        rule_counts = df['rule'].value_counts()
        augmented_rows = []
        
        for rule in rule_counts.index:
            if rule_counts[rule] < target_samples_per_rule:
                needed = target_samples_per_rule - rule_counts[rule]
                rule_data = df[df['rule'] == rule]
                
                # Sample existing examples to augment
                samples_to_augment = rule_data.sample(min(needed, len(rule_data)), replace=True)
                
                for _, row in samples_to_augment.iterrows():
                    # Paraphrase the main comment
                    paraphrases = self.paraphrase_comment(row['body'], row['rule'])
                    
                    for paraphrase in paraphrases:
                        new_row = row.copy()
                        new_row['body'] = paraphrase
                        new_row['row_id'] = f"aug_{len(augmented_rows)}"
                        augmented_rows.append(new_row)
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            return pd.concat([df, augmented_df], ignore_index=True)
        return df
    
    def create_cv_splits(self, df: pd.DataFrame, n_splits: int = 5) -> List[Dict]:
        """Create stratified cross-validation splits"""
        # Create stratification key combining rule and subreddit
        df['strat_key'] = df['rule'].astype(str) + '_' + df['subreddit'].astype(str) + '_' + df['rule_violation'].astype(str)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['strat_key'])):
            splits.append({
                'fold': fold,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            })
        
        return splits
    
    def augment_full_dataset(self, df: pd.DataFrame, augment_factor: int = 3) -> pd.DataFrame:
        """Comprehensive data augmentation pipeline"""
        print(f"Original dataset size: {len(df)}")
        
        augmented_data = []
        
        for idx, row in df.iterrows():
            # Keep original
            augmented_data.append(row)
            
            # Add paraphrases
            paraphrases = self.paraphrase_comment(row['body'], row['rule'])
            for i, paraphrase in enumerate(paraphrases[:augment_factor-1]):
                new_row = row.copy()
                new_row['body'] = paraphrase
                new_row['row_id'] = f"{row['row_id']}_para_{i}"
                augmented_data.append(new_row)
            
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(df)} rows")
                time.sleep(0.1)  # Rate limiting
        
        augmented_df = pd.DataFrame(augmented_data)
        print(f"Augmented dataset size: {len(augmented_df)}")
        
        return augmented_df

def main():
    # Load data
    df = pd.read_csv('train.csv')
    
    # Initialize augmentor
    augmentor = DataAugmentor()  # Add API key if using OpenAI
    
    # Create CV splits first
    print("Creating cross-validation splits...")
    cv_splits = augmentor.create_cv_splits(df)
    
    # Save CV splits
    with open('cv_splits.json', 'w') as f:
        json.dump(cv_splits, f, indent=2)
    
    print(f"Created {len(cv_splits)} CV folds")
    for i, split in enumerate(cv_splits):
        print(f"Fold {i}: Train={split['train_size']}, Val={split['val_size']}")
    
    # Augment data (start with factor 2 to test)
    print("\nAugmenting dataset...")
    augmented_df = augmentor.augment_full_dataset(df, augment_factor=2)
    
    # Save augmented data
    augmented_df.to_csv('train_augmented.csv', index=False)
    print(f"Saved augmented dataset with {len(augmented_df)} samples")
    
    # Print statistics
    print("\nAugmented dataset statistics:")
    print(f"Rule distribution:\n{augmented_df['rule'].value_counts()}")
    print(f"Violation distribution:\n{augmented_df['rule_violation'].value_counts()}")

if __name__ == "__main__":
    main()