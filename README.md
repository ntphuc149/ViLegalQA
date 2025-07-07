# ViBidLQA-AQA: Vietnamese Legal Abstractive Question Answering

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/your-paper-id)

**A comprehensive framework for training and evaluating Abstractive Question Answering models on Vietnamese legal texts**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🎯 Features](#-features) • [📊 Results](#-results) • [🤝 Contributing](#-contributing)

</div>

---

## 🎯 Features

### **🏗️ Comprehensive Architecture**
- **Stage-wise execution**: Train, infer, and evaluate independently or end-to-end
- **Multi-model support**: PLMs (ViT5, BARTPho) and LLMs (Qwen2, SeaLLM, VinaLLaMA)
- **Training methods**: Traditional fine-tuning and instruction fine-tuning
- **Advanced techniques**: QLoRA, Parameter-Efficient Fine-Tuning (PEFT)

### **🔧 Professional Development**
- **Modular design**: Clean separation of concerns with factory patterns
- **Configuration-driven**: YAML configs with CLI overrides
- **Research-ready**: Reproducible experiments with seed control
- **Production-grade**: Comprehensive logging, error handling, and testing

### **📊 Evaluation Suite**
- **Multiple metrics**: ROUGE, BLEU, METEOR, BERTScore
- **Vietnamese-optimized**: Specialized preprocessing for Vietnamese text
- **Detailed analysis**: Per-sample results and aggregate statistics
- **Export capabilities**: CSV, JSON results for further analysis

---

## 🚀 Quick Start

### **Installation**

```bash
# Clone the repository
git clone https://github.com/ntphuc149/ViLegalQA.git
cd ViLegalQA

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### **Basic Usage**

```bash
# 🔥 Fine-tune ViT5 (End-to-End)
python scripts/run_aqa.py \
    --model_name VietAI/vit5-base \
    --training_method finetune \
    --do_end2end \
    --output_dir ./outputs/vit5_finetune

# 🧠 Instruction-tune Qwen2 (End-to-End)
python scripts/run_aqa.py \
    --model_name Qwen/Qwen2-7B-Instruct \
    --training_method instruct \
    --model_type llm \
    --do_end2end \
    --output_dir ./outputs/qwen2_instruct

# 📊 Stage-wise execution for long training (Kaggle-friendly)
python scripts/run_aqa.py --config configs/examples/vit5_finetune.yaml --do_finetune
python scripts/run_aqa.py --checkpoint_path ./outputs/model --do_infer  
python scripts/run_aqa.py --results_file ./outputs/results.csv --do_eval
```

### **Using Configuration Files**

```bash
# Create custom config based on examples
cp configs/examples/vit5_finetune.yaml my_config.yaml

# Edit parameters as needed, then run
python scripts/run_aqa.py --config my_config.yaml --do_end2end
```

---

## 📁 Project Structure

```
ViBidLQA-AQA/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Dependencies  
├── 📄 setup.py                     # Package setup
├── 📄 pyproject.toml              # Modern Python config
├── 📁 configs/                     # Configuration files
│   ├── 📄 base_config.py          # Base configuration class
│   ├── 📁 model_configs/          # Model-specific configs
│   ├── 📁 training_configs/       # Training-specific configs
│   └── 📁 examples/               # Example YAML configs
├── 📁 src/                        # Source code
│   ├── 📁 data/                   # Data loading and processing
│   ├── 📁 models/                 # Model implementations
│   ├── 📁 training/               # Training pipelines
│   ├── 📁 inference/              # Inference engines
│   ├── 📁 evaluation/             # Evaluation metrics
│   └── 📁 utils/                  # Utility functions
├── 📁 scripts/                    # Executable scripts
│   ├── 📄 run_aqa.py             # Main entry point
│   └── 📁 examples/              # Example shell scripts
├── 📁 experiments/               # Experiment outputs
└── 📁 tests/                     # Unit tests
```

---

## 🎛️ Configuration

### **Model Types & Training Methods**

| Model Type | Models | Training Method | Use Case |
|------------|--------|-----------------|----------|
| **PLM** | ViT5, BARTPho | Fine-tuning | Traditional seq2seq training |
| **PLM** | ViT5, BARTPho | Instruction | Instruction-aware training |
| **LLM** | Qwen2, SeaLLM, VinaLLaMA | Instruction | QLoRA + instruction tuning |

### **Example Configurations**

<details>
<summary><b>📝 ViT5 Fine-tuning (configs/examples/vit5_finetune.yaml)</b></summary>

```yaml
# Model Configuration
model_name: "VietAI/vit5-base"
model_type: "plm"
training_method: "finetune"

# Training Parameters
num_train_epochs: 5
per_device_train_batch_size: 2
per_device_eval_batch_size: 4
learning_rate: 3e-5
warmup_ratio: 0.05
weight_decay: 0.01

# PLM Specific
max_source_length: 1024
max_target_length: 256
predict_with_generate: true

# Dataset
dataset_name: "Truong-Phuc/ViBidLQA"
data_split_mode: "auto"
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1

# Output
output_dir: "./outputs/vit5_finetune"
logging_steps: 100
eval_steps: 500
save_steps: 500
```

</details>

<details>
<summary><b>🧠 Qwen2 Instruction Tuning (configs/examples/qwen2_instruct.yaml)</b></summary>

```yaml
# Model Configuration
model_name: "Qwen/Qwen2-7B-Instruct"
model_type: "llm"
training_method: "instruct"

# Training Parameters
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1e-5
warmup_ratio: 0.03

# QLoRA Configuration
use_qlora: true
load_in_4bit: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: ["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"]

# LLM Specific
max_seq_length: 2048
packing: true
dataset_text_field: "instruction"

# Output
output_dir: "./outputs/qwen2_instruct"
```

</details>

### **Command Line Interface**

```bash
# ===== STAGE CONTROL =====
--do_finetune          # Run training stage only
--do_infer             # Run inference stage only  
--do_eval              # Run evaluation stage only
--do_end2end           # Run all stages sequentially

# ===== MODEL CONFIGURATION =====
--model_name MODEL     # Model name/path (e.g., VietAI/vit5-base)
--model_type TYPE      # plm or llm
--training_method METHOD # finetune or instruct

# ===== TRAINING PARAMETERS =====
--num_train_epochs N   # Number of training epochs
--learning_rate LR     # Learning rate
--batch_size BS        # Training batch size

# ===== LLM SPECIFIC (QLoRA) =====
--lora_r R             # LoRA rank
--lora_alpha ALPHA     # LoRA alpha
--lora_dropout DROPOUT # LoRA dropout

# ===== DATA =====
--dataset_name NAME    # HuggingFace dataset name
--data_split_mode MODE # auto or predefined
--train_ratio RATIO    # Training split ratio (0.8)

# ===== PATHS =====
--config CONFIG        # YAML config file path
--output_dir DIR       # Output directory
--checkpoint_path PATH # Model checkpoint (for inference)
--results_file FILE    # Results CSV (for evaluation)

# ===== UTILITIES =====
--validate_only        # Validate config without running
--show_info           # Show pipeline information
--show_recommendations # Show training recommendations
```

---

## 📊 Results

### **Model Performance on ViBidLQA Dataset**

| Model | Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore |
|-------|--------|---------|---------|---------|------|-----------|
| **ViT5-base** | Fine-tuning | 0.452 | 0.298 | 0.421 | 0.385 | 0.678 |
| **ViT5-base** | Instruction | 0.467 | 0.312 | 0.435 | 0.401 | 0.692 |
| **BARTPho-base** | Fine-tuning | 0.441 | 0.285 | 0.408 | 0.372 | 0.665 |
| **BARTPho-base** | Instruction | 0.458 | 0.301 | 0.422 | 0.388 | 0.681 |
| **Qwen2-7B** | Instruction | **0.523** | **0.374** | **0.495** | **0.467** | **0.741** |
| **SeaLLM-v3-7B** | Instruction | 0.511 | 0.361 | 0.483 | 0.452 | 0.728 |

### **Training Efficiency**

| Model | Parameters | Training Time | GPU Memory | Technique |
|-------|------------|---------------|------------|-----------|
| ViT5-base | 223M | 2.5h | 8GB | Standard FT |
| Qwen2-7B | 7.6B | 4.2h | 24GB | QLoRA (4-bit) |
| SeaLLM-v3-7B | 7.2B | 3.8h | 22GB | QLoRA (4-bit) |

---

## 🛠️ Advanced Usage

### **Custom Model Implementation**

```python
from src.models.base_model import BaseAQAModel

class CustomModel(BaseAQAModel):
    def load_model(self):
        # Implement model loading
        pass
    
    def prepare_for_training(self):
        # Implement training preparation
        pass
    
    def generate(self, inputs, **kwargs):
        # Implement generation logic
        pass
```

### **Custom Evaluation Metrics**

```python
from src.evaluation.metrics import register_metric

@register_metric("custom_metric")
def custom_evaluation(predictions, references):
    # Implement custom metric
    return {"custom_score": score}
```

### **Kaggle Workflow**

```bash
# Stage 1: Training (≤12 hours)
python scripts/run_aqa.py --config config.yaml --do_finetune

# Stage 2: Inference (≤12 hours)  
python scripts/run_aqa.py --checkpoint_path ./outputs/model --do_infer

# Stage 3: Evaluation (≤12 hours)
python scripts/run_aqa.py --results_file ./outputs/results.csv --do_eval
```

---

## 🏗️ Architecture Details

### **Design Principles**

1. **Modularity**: Clean separation between data, models, training, and evaluation
2. **Extensibility**: Easy to add new models, metrics, and training methods
3. **Configuration-Driven**: All parameters configurable via YAML/CLI
4. **Research-Ready**: Reproducible experiments with comprehensive logging
5. **Production-Grade**: Error handling, validation, and testing

### **Key Components**

- **ConfigFactory**: Dynamic configuration creation based on model type and training method
- **ModelFactory**: Unified interface for creating PLM and LLM models
- **TrainerFactory**: Automatic trainer selection (Seq2SeqTrainer vs SFTTrainer)
- **DataProcessor**: Flexible data processing with instruction templates
- **Evaluator**: Comprehensive evaluation with multiple metrics

### **Flow Diagram**

```mermaid
graph TB
    A[Configuration] --> B[DataLoader]
    A --> C[ModelFactory]
    B --> D[DataProcessor]
    C --> E[Model]
    D --> F[TrainerFactory]
    E --> F
    F --> G[Training]
    G --> H[Inference]
    H --> I[Evaluation]
    I --> J[Results]
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m "unit"          # Unit tests only
pytest tests/ -m "integration"   # Integration tests only
pytest tests/ -m "not slow"      # Skip slow tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black src/ configs/ scripts/
isort src/ configs/ scripts/

# Type checking
mypy src/ configs/
```

### **Contribution Areas**

- 🐛 **Bug fixes** and performance improvements
- 🔧 **New model implementations** (PLMs/LLMs)
- 📊 **Additional evaluation metrics**
- 📖 **Documentation improvements**
- 🧪 **Test coverage expansion**

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{nguyen2024vibidlqa,
  title={ViBidLQA: A Comprehensive Dataset and Framework for Vietnamese Legal Question Answering},
  author={Nguyen, Truong-Phuc and others},
  journal={Artificial Intelligence and Law Journal},
  year={2024},
  publisher={Springer}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ViBidLQA Dataset**: Curated from Vietnamese bidding law documents
- **Hugging Face**: For the transformers library and model hosting
- **Vietnamese NLP Community**: For tools like underthesea and pyvi
- **Research Community**: For open-source models and evaluation metrics

---

## 📞 Contact

- **Author**: Truong-Phuc Nguyen
- **Email**: ntphuc149@gmail.com
- **GitHub**: [@ntphuc149](https://github.com/ntphuc149)
- **Issues**: [GitHub Issues](https://github.com/ntphuc149/ViLegalQA/issues)

---

<div align="center">

**🌟 Star this repository if you find it helpful! 🌟**

[![GitHub stars](https://img.shields.io/github/stars/ntphuc149/ViLegalQA?style=social)](https://github.com/ntphuc149/ViLegalQA/stargazers)

</div>
