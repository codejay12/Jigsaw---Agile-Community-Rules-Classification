import os, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import pandas as pd
import numpy as np

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv', index_col='row_id')
else:
    test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    sub = test[['row_id']].copy()
    
import vllm
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import torch
import vllm
import numpy as np
from vllm.lora.request import LoRARequest
import argparse
from scipy.special import softmax

llm = vllm.LLM(
    "/qwen-2.5-7b-instruct-instruct-jigsaw",
    tensor_parallel_size=2, 
    gpu_memory_utilization=0.95, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    max_model_len=2048,
    disable_log_stats=True,
    enable_prefix_caching=True,
    
)

tokenizer = llm.get_tokenizer()
sys_prompt = '''You are given a comment on reddit and a rule. Your task is to classify whether the comment violates the rule. Only respond Yes/No.'''
def formatting(dataset):
    texts = []
    for i in range(len(dataset)):
        texts.append(tokenizer.apply_chat_template(dataset[i], tokenize=False, add_generation_prompt=False))
    return texts
  
template = """
Subreddit: r/{subreddit}
Rule: {rule}
Examples:
1) {positive_example_1}
Violation: Yes

2) {negative_example_1}
Violation: No

3) {negative_example_2}
Violation: No

4) {positive_example_2}
Violation: Yes
Comment:
{body}
Violation: """

from typing import Any, Dict, List
from transformers import LogitsProcessor
import torch

choices = ["No", "Yes"]

KEEP = []
for x in choices:
    c = tokenizer.encode(x,add_special_tokens=False)[0]
    KEEP.append(c)
print(f"Force predictions to be tokens {KEEP} which are {choices}.")

class DigitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.allowed_ids = KEEP
        
    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        scores[self.allowed_ids] += 100
        return scores
      
dataset = []

for index,row in test.iterrows():
    
    formatted_sample = [
        {
        "role": "system",
        "content": sys_prompt
    },
       {
           "role": "user",
           "content": template.format(
               rule = row.rule,
               subreddit = row.subreddit,
               body = row.body,
               positive_example_1 = row.positive_example_1,
               negative_example_1 = row.negative_example_1,
               positive_example_2 = row.positive_example_2,
               negative_example_2 = row.negative_example_2
           )
       }]
    
    dataset.append( formatted_sample )
all_prompts = formatting(dataset)
logits_processors = [DigitLogitsProcessor(tokenizer)]
responses = llm.generate(
    all_prompts,
    vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness the sampling
        seed=777, # Seed for reprodicibility
        skip_special_tokens=True,  # Whether to skip special tokens in the output.
        max_tokens=1,  # Maximum number of tokens to generate per output sequence.
        logits_processors=logits_processors,
        logprobs = 2
    ),
    use_tqdm = True
)
results = []
errors = 0

for i,response in enumerate(responses):
    try:
        x = response.outputs[0].logprobs[0]
        logprobs = []
        for k in KEEP:
            if k in x:
                logprobs.append( math.exp(x[k].logprob) )
            else:
                logprobs.append( 0 )
                print(f"bad logits {i}")
        logprobs = np.array( logprobs )
        logprobs /= logprobs.sum()
        results.append( logprobs )
    except:
        #print(f"error {i}")
        results.append( np.array([1/2., 1/2.]) )
        errors += 1
        
print(f"There were {errors} inference errors out of {i+1} inferences")
results = np.vstack(results)

probs = [x[1] for x in results]
from sklearn.metrics import roc_auc_score
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print(roc_auc_score(test['rule_violation'], probs))
    
sub['rule_violation'] = probs
sub.to_csv('submission.csv')