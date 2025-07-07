"""
Instruction templates for different models and training methods.
Provides consistent formatting for instruction-based training.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class InstructionTemplate(ABC):
    """
    Abstract base class for instruction templates.
    """
    
    def __init__(self, system_message: str = ""):
        """
        Initialize instruction template.
        
        Args:
            system_message: System message for the model
        """
        self.system_message = system_message
    
    @abstractmethod
    def format_instruction(
        self,
        context: str,
        question: str,
        answer: Optional[str] = None
    ) -> str:
        """
        Format instruction for training or inference.
        
        Args:
            context: Legal document context
            question: User question
            answer: Expected answer (for training)
            
        Returns:
            Formatted instruction string
        """
        pass
    
    @abstractmethod
    def format_inference_input(self, context: str, question: str) -> str:
        """
        Format input for inference only.
        
        Args:
            context: Legal document context
            question: User question
            
        Returns:
            Formatted input string
        """
        pass


class VietnameseLegalTemplate(InstructionTemplate):
    """
    Template for Vietnamese legal question answering.
    """
    
    def __init__(self, system_message: str = "Bạn là chuyên gia về lĩnh vực pháp luật tại Việt Nam."):
        super().__init__(system_message)
    
    def format_instruction(
        self,
        context: str,
        question: str,
        answer: Optional[str] = None
    ) -> str:
        """Format instruction for Vietnamese legal QA."""
        instruction = f"""Dựa vào nội dung văn bản pháp luật sau:
{context}

Bạn hãy đưa ra câu trả lời cho câu hỏi: {question}"""
        
        if answer is not None:
            instruction += f"\n\nCâu trả lời: {answer}"
        
        return instruction
    
    def format_inference_input(self, context: str, question: str) -> str:
        """Format input for inference."""
        return self.format_instruction(context, question, answer=None)


class ChatMLTemplate(InstructionTemplate):
    """
    ChatML format template for instruction tuning (used by Qwen, etc.).
    """
    
    def __init__(self, system_message: str = "Bạn là chuyên gia về lĩnh vực pháp luật tại Việt Nam."):
        super().__init__(system_message)
    
    def format_instruction(
        self,
        context: str,
        question: str,
        answer: Optional[str] = None
    ) -> str:
        """Format instruction using ChatML format."""
        messages = [
            f"<|im_start|>system\n{self.system_message}<|im_end|>",
            f"<|im_start|>user\nDựa vào nội dung văn bản pháp luật sau:\n{context}\nBạn hãy đưa ra câu trả lời cho câu hỏi:\n{question}<|im_end|>"
        ]
        
        if answer is not None:
            messages.append(f"<|im_start|>assistant\n{answer}<|im_end|>")
        
        return "\n".join(messages)
    
    def format_inference_input(self, context: str, question: str) -> str:
        """Format input for inference."""
        messages = [
            f"<|im_start|>system\n{self.system_message}<|im_end|>",
            f"<|im_start|>user\nDựa vào nội dung văn bản pháp luật sau:\n{context}\nBạn hãy đưa ra câu trả lời cho câu hỏi:\n{question}<|im_end|>",
            "<|im_start|>assistant\n"
        ]
        
        return "\n".join(messages)


class AlpacaTemplate(InstructionTemplate):
    """
    Alpaca format template for instruction tuning.
    """
    
    def __init__(self, system_message: str = ""):
        super().__init__(system_message)
    
    def format_instruction(
        self,
        context: str,
        question: str,
        answer: Optional[str] = None
    ) -> str:
        """Format instruction using Alpaca format."""
        instruction_text = "Bạn là chuyên gia pháp luật. Dựa vào văn bản pháp luật được cung cấp, hãy trả lời câu hỏi một cách chính xác."
        
        input_text = f"Văn bản pháp luật: {context}\n\nCâu hỏi: {question}"
        
        formatted = f"### Instruction:\n{instruction_text}\n\n### Input:\n{input_text}\n\n### Response:\n"
        
        if answer is not None:
            formatted += answer
        
        return formatted
    
    def format_inference_input(self, context: str, question: str) -> str:
        """Format input for inference."""
        return self.format_instruction(context, question, answer=None)


class SimpleTemplate(InstructionTemplate):
    """
    Simple template without special tokens (for PLMs like ViT5, BARTPho).
    """
    
    def __init__(self, system_message: str = ""):
        super().__init__(system_message)
    
    def format_instruction(
        self,
        context: str,
        question: str,
        answer: Optional[str] = None
    ) -> str:
        """Format instruction simply."""
        if self.system_message:
            formatted = f"{self.system_message}\n\n"
        else:
            formatted = ""
        
        formatted += f"Văn bản: {context}\nCâu hỏi: {question}\nTrả lời:"
        
        if answer is not None:
            formatted += f" {answer}"
        
        return formatted
    
    def format_inference_input(self, context: str, question: str) -> str:
        """Format input for inference."""
        if self.system_message:
            formatted = f"{self.system_message}\n\n"
        else:
            formatted = ""
        
        formatted += f"Văn bản: {context}\nCâu hỏi: {question}\nTrả lời:"
        
        return formatted


class LlamaTemplate(InstructionTemplate):
    """
    LLaMA-style template for instruction tuning.
    """
    
    def __init__(self, system_message: str = "Bạn là chuyên gia về lĩnh vực pháp luật tại Việt Nam."):
        super().__init__(system_message)
    
    def format_instruction(
        self,
        context: str,
        question: str,
        answer: Optional[str] = None
    ) -> str:
        """Format instruction using LLaMA format."""
        messages = [
            f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n",
            f"Dựa vào nội dung văn bản pháp luật sau:\n{context}\n\nBạn hãy đưa ra câu trả lời cho câu hỏi:\n{question} [/INST]"
        ]
        
        if answer is not None:
            messages.append(f" {answer} </s>")
        
        return "".join(messages)
    
    def format_inference_input(self, context: str, question: str) -> str:
        """Format input for inference."""
        messages = [
            f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n",
            f"Dựa vào nội dung văn bản pháp luật sau:\n{context}\n\nBạn hãy đưa ra câu trả lời cho câu hỏi:\n{question} [/INST] "
        ]
        
        return "".join(messages)


# Template registry
INSTRUCTION_TEMPLATES = {
    "vietnamese_legal": VietnameseLegalTemplate,
    "chatml": ChatMLTemplate,
    "alpaca": AlpacaTemplate,
    "simple": SimpleTemplate,
    "llama": LlamaTemplate,
}


def get_instruction_template(
    template_name: str,
    system_message: Optional[str] = None
) -> InstructionTemplate:
    """
    Get instruction template by name.
    
    Args:
        template_name: Name of the template
        system_message: Optional custom system message
        
    Returns:
        InstructionTemplate instance
        
    Raises:
        ValueError: If template name is not found
    """
    if template_name not in INSTRUCTION_TEMPLATES:
        available_templates = list(INSTRUCTION_TEMPLATES.keys())
        raise ValueError(
            f"Unknown template '{template_name}'. "
            f"Available templates: {available_templates}"
        )
    
    template_class = INSTRUCTION_TEMPLATES[template_name]
    
    if system_message is not None:
        return template_class(system_message=system_message)
    else:
        return template_class()


def auto_select_template(model_name: str) -> str:
    """
    Automatically select appropriate template based on model name.
    
    Args:
        model_name: Model name
        
    Returns:
        Template name
    """
    model_name_lower = model_name.lower()
    
    # Qwen models use ChatML
    if "qwen" in model_name_lower:
        return "chatml"
    
    # LLaMA-based models
    if any(model in model_name_lower for model in ["llama", "vicuna", "alpaca"]):
        return "llama"
    
    # SeaLLM models (LLaMA-based but may prefer ChatML)
    if "sea" in model_name_lower:
        return "chatml"
    
    # VinaLLaMA models
    if "vina" in model_name_lower:
        return "llama"
    
    # PLM models (ViT5, BARTPho)
    if any(model in model_name_lower for model in ["vit5", "bartpho", "t5", "bart"]):
        return "simple"
    
    # Default fallback
    logger.warning(f"Unknown model {model_name}, using default template")
    return "vietnamese_legal"


def format_dataset_with_template(
    dataset: Dict[str, Any],
    template_name: str,
    system_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format entire dataset with instruction template.
    
    Args:
        dataset: Dataset with context, question, answer fields
        template_name: Template name to use
        system_message: Optional system message
        
    Returns:
        Dataset with formatted instruction field
    """
    template = get_instruction_template(template_name, system_message)
    
    def format_sample(sample):
        formatted_instruction = template.format_instruction(
            context=sample["context"],
            question=sample["question"],
            answer=sample.get("answer", sample.get("abstractive_answer"))
        )
        
        sample["instruction"] = formatted_instruction
        return sample
    
    # Apply formatting based on dataset type
    if hasattr(dataset, 'map'):
        # HuggingFace Dataset
        return dataset.map(format_sample)
    elif isinstance(dataset, dict):
        # Dictionary format
        formatted_samples = []
        for sample in dataset.get("data", []):
            formatted_samples.append(format_sample(sample))
        dataset["data"] = formatted_samples
        return dataset
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")


# Template validation
def validate_template_output(
    template: InstructionTemplate,
    sample_context: str = "Luật Đấu thầu số 22/2023/QH15",
    sample_question: str = "Luật nào quy định về đấu thầu?",
    sample_answer: str = "Luật Đấu thầu số 22/2023/QH15"
) -> bool:
    """
    Validate template output format.
    
    Args:
        template: Template to validate
        sample_context: Sample context for testing
        sample_question: Sample question for testing
        sample_answer: Sample answer for testing
        
    Returns:
        True if template output is valid
    """
    try:
        # Test training format
        training_output = template.format_instruction(
            sample_context, sample_question, sample_answer
        )
        
        # Test inference format
        inference_output = template.format_inference_input(
            sample_context, sample_question
        )
        
        # Basic validation
        assert len(training_output) > 0, "Training output is empty"
        assert len(inference_output) > 0, "Inference output is empty"
        assert sample_context in training_output, "Context not found in training output"
        assert sample_question in training_output, "Question not found in training output"
        assert sample_answer in training_output, "Answer not found in training output"
        assert sample_context in inference_output, "Context not found in inference output"
        assert sample_question in inference_output, "Question not found in inference output"
        
        logger.info(f"Template validation passed for {template.__class__.__name__}")
        return True
        
    except Exception as e:
        logger.error(f"Template validation failed for {template.__class__.__name__}: {e}")
        return False


# Test all templates
def test_all_templates() -> Dict[str, bool]:
    """
    Test all registered templates.
    
    Returns:
        Dictionary mapping template names to validation results
    """
    results = {}
    
    for template_name in INSTRUCTION_TEMPLATES.keys():
        try:
            template = get_instruction_template(template_name)
            results[template_name] = validate_template_output(template)
        except Exception as e:
            logger.error(f"Failed to test template {template_name}: {e}")
            results[template_name] = False
    
    return results