"""
Configuration for instruction fine-tuning of both PLMs and LLMs
Supports instruction-based training with various chat templates and formats.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from ..base_config import BaseConfig


@dataclass
class InstructConfig(BaseConfig):
    """
    Configuration for instruction fine-tuning of both PLM and LLM models.
    
    This config handles instruction formatting, chat templates, and 
    specialized training parameters for instruction-following models.
    """
    
    # Override training method
    training_method: str = "instruct"
    
    # ===== INSTRUCTION FORMATTING =====
    chat_template: str = "chatml"
    """Chat template format: 'chatml', 'vicuna', 'alpaca', 'llama2', etc."""
    
    system_message: str = "Bạn là chuyên gia về lĩnh vực pháp luật tại Việt Nam."
    """System message for instruction formatting."""
    
    instruction_template_name: str = "vietnamese_legal"
    """Template name for instruction formatting."""
    
    # ===== PLM INSTRUCTION SETTINGS =====
    plm_instruction_prefix: str = "Dựa vào nội dung văn bản pháp luật sau:"
    """Instruction prefix for PLM models."""
    
    plm_question_prefix: str = "Bạn hãy đưa ra câu trả lời cho câu hỏi:"
    """Question prefix for PLM models."""
    
    # ===== LLM SPECIFIC SETTINGS (inherited from LLMConfig when model_type='llm') =====
    
    # QLoRA settings
    use_qlora: bool = True
    """Whether to use QLoRA for LLM training."""
    
    load_in_4bit: bool = True
    """Whether to load LLM in 4-bit precision."""
    
    bnb_4bit_quant_type: str = "nf4"
    """4-bit quantization type."""
    
    bnb_4bit_compute_dtype: str = "float16"
    """Compute dtype for 4-bit quantization."""
    
    bnb_4bit_use_double_quant: bool = True
    """Whether to use double quantization."""
    
    # LoRA settings
    lora_r: int = 16
    """LoRA rank."""
    
    lora_alpha: int = 32
    """LoRA alpha parameter."""
    
    lora_dropout: float = 0.05
    """LoRA dropout rate."""
    
    lora_target_modules: list = field(default_factory=lambda: [
        'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'
    ])
    """Target modules for LoRA adaptation."""
    
    lora_bias: str = "none"
    """LoRA bias setting."""
    
    lora_task_type: str = "CAUSAL_LM"
    """LoRA task type."""
    
    # ===== SEQUENCE SETTINGS =====
    max_seq_length: int = 2048
    """Maximum sequence length (for LLMs) or max_source_length (for PLMs)."""
    
    max_target_length: int = 256
    """Maximum target length (mainly for PLMs)."""
    
    # ===== TRAINING SETTINGS =====
    packing: bool = True
    """Whether to use sequence packing (mainly for LLMs)."""
    
    dataset_text_field: str = "instruction"
    """Field name containing instruction text."""
    
    # ===== OPTIMIZER SETTINGS =====
    optim: str = "paged_adamw_32bit"
    """Optimizer type for LLM training."""
    
    # ===== TEMPLATE CONFIGURATIONS =====
    template_configs: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "chatml": {
            "system_start": "<|im_start|>system",
            "system_end": "<|im_end|>",
            "user_start": "<|im_start|>user",
            "user_end": "<|im_end|>",
            "assistant_start": "<|im_start|>assistant",
            "assistant_end": "<|im_end|>"
        },
        "vicuna": {
            "system_start": "### System:\n",
            "system_end": "\n\n",
            "user_start": "### Human:\n",
            "user_end": "\n\n",
            "assistant_start": "### Assistant:\n",
            "assistant_end": "\n\n"
        },
        "alpaca": {
            "system_start": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
            "system_end": "",
            "user_start": "### Instruction:\n",
            "user_end": "\n\n",
            "assistant_start": "### Response:\n",
            "assistant_end": ""
        }
    })
    """Template configurations for different chat formats."""
    
    def __post_init__(self):
        """Post-initialization validation and setup for instruction tuning."""
        super().__post_init__()
        
        # Validate chat template
        if self.chat_template not in self.template_configs:
            raise ValueError(f"Unsupported chat template: {self.chat_template}")
        
        # Set model-type specific defaults
        if self.model_type == "llm":
            self._setup_llm_defaults()
        elif self.model_type == "plm":
            self._setup_plm_defaults()
        
        # Validate LoRA settings for LLMs
        if self.model_type == "llm" and self.use_qlora:
            self._validate_lora_settings()
    
    def _setup_llm_defaults(self):
        """Setup default values for LLM instruction tuning."""
        # Adjust batch size for LLM training
        if self.per_device_train_batch_size > 1:
            print("Adjusting batch size to 1 for LLM instruction tuning")
            self.per_device_train_batch_size = 1
        
        # Ensure gradient accumulation for effective batch size
        if self.gradient_accumulation_steps < 2:
            self.gradient_accumulation_steps = 2
        
        # Set appropriate learning rate for LLMs
        if self.learning_rate > 1e-4:
            print(f"Adjusting learning rate from {self.learning_rate} to 1e-5 for LLM")
            self.learning_rate = 1e-5
    
    def _setup_plm_defaults(self):
        """Setup default values for PLM instruction tuning."""
        # PLMs can handle larger batch sizes
        if self.per_device_train_batch_size < 2:
            self.per_device_train_batch_size = 2
        
        # Set max_source_length from max_seq_length for PLMs
        self.max_source_length = self.max_seq_length
        
        # Disable LLM-specific settings for PLMs
        self.use_qlora = False
        self.load_in_4bit = False
        self.packing = False
    
    def _validate_lora_settings(self):
        """Validate LoRA configuration for LLMs."""
        if self.lora_alpha < self.lora_r:
            print(f"Warning: lora_alpha ({self.lora_alpha}) < lora_r ({self.lora_r})")
        
        if self.lora_dropout < 0 or self.lora_dropout > 1:
            raise ValueError(f"lora_dropout must be between 0 and 1, got {self.lora_dropout}")
    
    def get_chat_template(self) -> Dict[str, str]:
        """Get the chat template configuration."""
        return self.template_configs[self.chat_template]
    
    def format_instruction(self, context: str, question: str, answer: str = "") -> str:
        """Format instruction based on the selected template and model type."""
        template = self.get_chat_template()
        
        if self.model_type == "llm":
            return self._format_llm_instruction(context, question, answer, template)
        else:
            return self._format_plm_instruction(context, question, answer)
    
    def _format_llm_instruction(self, context: str, question: str, answer: str, template: Dict[str, str]) -> str:
        """Format instruction for LLM models using chat templates."""
        instruction_parts = []
        
        # System message
        if self.system_message:
            instruction_parts.extend([
                template["system_start"],
                self.system_message,
                template["system_end"]
            ])
        
        # User message
        user_message = f"{self.plm_instruction_prefix}\n{context}\n{self.plm_question_prefix}\n{question}"
        instruction_parts.extend([
            template["user_start"],
            user_message,
            template["user_end"]
        ])
        
        # Assistant message (for training)
        if answer:
            instruction_parts.extend([
                template["assistant_start"],
                answer,
                template["assistant_end"]
            ])
        else:
            # For inference, just add the start token
            instruction_parts.append(template["assistant_start"])
        
        return "".join(instruction_parts)
    
    def _format_plm_instruction(self, context: str, question: str, answer: str = "") -> str:
        """Format instruction for PLM models (simpler format)."""
        instruction = f"{self.plm_instruction_prefix}\n{context}\n{self.plm_question_prefix} {question}"
        return instruction
    
    def get_training_args_dict(self) -> Dict[str, Any]:
        """Get training arguments based on model type."""
        base_args = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "evaluation_strategy": self.evaluation_strategy,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "seed": self.seed,
            "report_to": "wandb" if self.use_wandb else "none"
        }
        
        # Add model-type specific arguments
        if self.model_type == "llm":
            base_args.update({
                "optim": self.optim,
                "packing": self.packing,
                "dataset_text_field": self.dataset_text_field,
                "max_seq_length": self.max_seq_length
            })
        else:  # PLM
            base_args.update({
                "predict_with_generate": True,
                "generation_max_length": self.max_target_length,
                "generation_num_beams": 4
            })
        
        return base_args
    
    def get_lora_config_dict(self) -> Dict[str, Any]:
        """Get LoRA configuration for LLM training."""
        if self.model_type != "llm" or not self.use_qlora:
            return {}
        
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.lora_target_modules,
            "bias": self.lora_bias,
            "task_type": self.lora_task_type
        }