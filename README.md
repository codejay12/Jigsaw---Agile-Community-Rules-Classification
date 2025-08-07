# Jigsaw Competition - Reddit Comment Rule Violation Prediction

A machine learning solution for predicting Reddit comment rule violations using fine-tuned large language models. This project implements a comprehensive pipeline for data augmentation, model training, and ensemble inference to achieve high performance on the Jigsaw Agile Community Rules competition.

## Project Overview

This solution addresses the challenge of building models that can predict whether Reddit comments violate specific subreddit rules. The key challenge is developing a flexible model capable of generalizing to rules not present in the training data.

### Key Features

- Fine-tuning of Qwen2.5-7B-Instruct model using Unsloth
- Data augmentation pipeline for 3x training data increase
- Multi-model ensemble with test-time augmentation
- Comprehensive experiment tracking and configuration management
- Cross-validation strategy respecting rule distribution

## Performance

- Initial baseline: 0.81 AUC
- Current leaderboard score: 0.867 AUC
- Target performance: 0.90+ AUC

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended: 16GB+ VRAM)- kaggle notebooks(2X T4 gpus's)
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jigsaw-competition.git
cd jigsaw-competition
```

2. Run the setup script:
```bash
chmod +x run.sh
./run.sh setup
```

This will:
- Create a virtual environment
- Install all dependencies
- Create necessary directories

## Usage

### Quick Start

Run the full experiment pipeline:
```bash
./run.sh experiment
```

### Individual Components

1. **Data Augmentation**:
```bash
./run.sh augment
```

2. **Training**:
```bash
./run.sh train --config baseline_v2 --gpu 0
```

3. **Inference**:
```bash
./run.sh inference --gpu 0
```

4. **Ensemble**:
```bash
./run.sh ensemble
```

5. **Run Tests**:
```bash
./run.sh test
```

## Project Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── src/
│   ├── __init__.py
│   ├── training.py             # Basic training script
│   ├── inference.py            # Inference pipeline
│   ├── improved_training.py    # Enhanced training with CV
│   ├── data_augmentation.py    # Data preprocessing
│   ├── ensemble_inference.py   # Multi-model ensemble
│   ├── config.py              # Configuration management
│   └── run_experiments.py     # Full pipeline orchestration
├── test/
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_model_utils.py
│   └── test_config.py
├── run.sh                      # Main execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

Edit `src/config.py` to customize training parameters:

```python
ExperimentConfig:
    model_name: str = "unsloth/Qwen2.5-7B-Instruct"
    lora_r: int = 32                    # LoRA rank
    max_steps: int = 500               # Training steps  
    learning_rate: float = 2e-4        # Learning rate
    use_augmented_data: bool = True    # Use augmented data
    prompt_template: str = "template_v2" # Prompt version
```

## Data Format

### Training Data (train.csv)
- `row_id`: Unique identifier
- `body`: Comment text
- `rule`: Rule being evaluated
- `subreddit`: Source subreddit
- `positive_example_1`, `positive_example_2`: Examples of rule violations
- `negative_example_1`, `negative_example_2`: Examples of non-violations
- `rule_violation`: Binary target (1 = violation, 0 = no violation)

### Test Data (test.csv)
Same format as training data but without `rule_violation` column.

## Model Architecture

- **Base Model**: Qwen2.5-7B-Instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit for memory efficiency
- **Max Sequence Length**: 2048 tokens
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Training Strategy

1. **Cross-Validation**: 5-fold stratified by rule+subreddit
2. **Data Augmentation**: Paraphrasing to triple dataset size
3. **Hyperparameter Optimization**: Grid search over LoRA rank, learning rate
4. **Extended Training**: 500+ steps vs baseline 60 steps
5. **Prompt Engineering**: Multiple template variations

## Inference Pipeline

1. **Model Loading**: Load fine-tuned checkpoints
2. **Prompt Formatting**: Apply chat template with examples
3. **Constrained Generation**: Force Yes/No outputs
4. **Probability Extraction**: Get logprobs for binary classification
5. **Ensemble Averaging**: Combine multiple model predictions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Unsloth team for the efficient fine-tuning framework
- Hugging Face for model hosting and tools
- Kaggle for hosting the competition
- Reddit communities for the data

## Contact

For questions or collaboration, please open an issue on GitHub.
