"""
Instruction templates and chat formatting for Vietnamese Legal QA.
Supports multiple chat formats and instruction styles.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from abc import ABC, abstractmethod

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ChatTemplate:
    """Chat template configuration for different LLM formats."""
    
    name: str
    system_start: str
    system_end: str
    user_start: str
    user_end: str
    assistant_start: str
    assistant_end: str
    description: str = ""
    
    def format_message(
        self,
        system_msg: Optional[str] = None,
        user_msg: Optional[str] = None,
        assistant_msg: Optional[str] = None,
        include_assistant_start: bool = True
    ) -> str:
        """Format a complete chat message."""
        parts = []
        
        # Add system message
        if system_msg:
            parts.extend([self.system_start, system_msg, self.system_end])
        
        # Add user message
        if user_msg:
            parts.extend([self.user_start, user_msg, self.user_end])
        
        # Add assistant message
        if assistant_msg:
            parts.extend([self.assistant_start, assistant_msg, self.assistant_end])
        elif include_assistant_start:
            parts.append(self.assistant_start)
        
        return "".join(parts)


class InstructionTemplateManager:
    """Manager for instruction templates and chat formats."""
    
    # Predefined chat templates
    CHAT_TEMPLATES = {
        "chatml": ChatTemplate(
            name="chatml",
            system_start="<|im_start|>system\n",
            system_end="<|im_end|>\n",
            user_start="<|im_start|>user\n",
            user_end="<|im_end|>\n",
            assistant_start="<|im_start|>assistant\n",
            assistant_end="<|im_end|>",
            description="ChatML format used by OpenAI and Qwen models"
        ),
        
        "vicuna": ChatTemplate(
            name="vicuna",
            system_start="### System:\n",
            system_end="\n\n",
            user_start="### Human:\n",
            user_end="\n\n",
            assistant_start="### Assistant:\n",
            assistant_end="\n\n",
            description="Vicuna-style conversation format"
        ),
        
        "alpaca": ChatTemplate(
            name="alpaca",
            system_start="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
            system_end="",
            user_start="### Instruction:\n",
            user_end="\n\n",
            assistant_start="### Response:\n",
            assistant_end="",
            description="Alpaca instruction format"
        ),
        
        "llama2": ChatTemplate(
            name="llama2",
            system_start="<s>[INST] <<SYS>>\n",
            system_end="\n<</SYS>>\n\n",
            user_start="",
            user_end=" [/INST] ",
            assistant_start="",
            assistant_end=" </s><s>[INST] ",
            description="Llama-2 chat format"
        ),
        
        "seallm": ChatTemplate(
            name="seallm",
            system_start="<|im_start|>system\n",
            system_end="<|im_end|>\n",
            user_start="<|im_start|>user\n",
            user_end="<|im_end|>\n",
            assistant_start="<|im_start|>assistant\n",
            assistant_end="<|im_end|>",
            description="SeaLLM chat format (similar to ChatML)"
        )
    }
    
    # Vietnamese legal instruction templates
    INSTRUCTION_TEMPLATES = {
        "vietnamese_legal": {
            "system_message": "Bạn là chuyên gia về lĩnh vực pháp luật tại Việt Nam.",
            "context_prefix": "Dựa vào nội dung văn bản pháp luật sau:",
            "question_prefix": "Bạn hãy đưa ra câu trả lời cho câu hỏi:",
            "answer_prefix": "",
            "description": "Standard Vietnamese legal QA format"
        },
        
        "vietnamese_legal_formal": {
            "system_message": "Bạn là một luật sư chuyên nghiệp với kinh nghiệm sâu rộng về pháp luật Việt Nam.",
            "context_prefix": "Căn cứ vào các quy định pháp luật sau đây:",
            "question_prefix": "Vui lòng phân tích và đưa ra câu trả lời pháp lý cho vấn đề:",
            "answer_prefix": "Theo quy định của pháp luật Việt Nam:",
            "description": "Formal legal language format"
        },
        
        "vietnamese_legal_simple": {
            "system_message": "Bạn là chuyên gia pháp luật, hãy trả lời câu hỏi một cách dễ hiểu.",
            "context_prefix": "Dựa vào văn bản pháp luật:",
            "question_prefix": "Câu hỏi:",
            "answer_prefix": "Trả lời:",
            "description": "Simplified format for accessibility"
        }
    }
    
    def __init__(self):
        """Initialize the template manager."""
        self._custom_templates = {}
        logger.info(f"Initialized InstructionTemplateManager with {len(self.CHAT_TEMPLATES)} chat templates")
    
    def get_chat_template(self, template_name: str) -> ChatTemplate:
        """Get a chat template by name."""
        if template_name not in self.CHAT_TEMPLATES:
            available = list(self.CHAT_TEMPLATES.keys())
            raise ValueError(f"Unknown chat template: {template_name}. Available: {available}")
        
        return self.CHAT_TEMPLATES[template_name]
    
    def get_instruction_template(self, template_name: str) -> Dict[str, str]:
        """Get an instruction template by name."""
        if template_name in self._custom_templates:
            return self._custom_templates[template_name]
        
        if template_name not in self.INSTRUCTION_TEMPLATES:
            available = list(self.INSTRUCTION_TEMPLATES.keys())
            raise ValueError(f"Unknown instruction template: {template_name}. Available: {available}")
        
        return self.INSTRUCTION_TEMPLATES[template_name]
    
    def register_custom_template(self, name: str, template: Dict[str, str]) -> None:
        """Register a custom instruction template."""
        required_keys = ["system_message", "context_prefix", "question_prefix"]
        missing_keys = [key for key in required_keys if key not in template]
        
        if missing_keys:
            raise ValueError(f"Missing required keys in template: {missing_keys}")
        
        self._custom_templates[name] = template
        logger.info(f"Registered custom template: {name}")
    
    def list_templates(self) -> Dict[str, List[str]]:
        """List all available templates."""
        return {
            "chat_templates": list(self.CHAT_TEMPLATES.keys()),
            "instruction_templates": list(self.INSTRUCTION_TEMPLATES.keys()) + list(self._custom_templates.keys())
        }
    
    def format_for_plm(
        self,
        context: str,
        question: str,
        answer: str = "",
        template_name: str = "vietnamese_legal"
    ) -> Dict[str, str]:
        """
        Format input for PLM (encoder-decoder) models.
        
        Args:
            context: Legal document context
            question: Question to answer
            answer: Target answer (for training)
            template_name: Instruction template to use
            
        Returns:
            Dictionary with 'input_text' and 'target_text'
        """
        template = self.get_instruction_template(template_name)
        
        # Build input text
        input_parts = []
        
        if template.get("context_prefix"):
            input_parts.append(template["context_prefix"])
            input_parts.append(context)
        
        if template.get("question_prefix"):
            input_parts.append(template["question_prefix"])
            input_parts.append(question)
        
        input_text = "\n".join(input_parts)
        
        # Target text is the answer
        target_text = answer if answer else ""
        
        return {
            "input_text": input_text,
            "target_text": target_text
        }
    
    def format_for_llm(
        self,
        context: str,
        question: str,
        answer: str = "",
        chat_template_name: str = "chatml",
        instruction_template_name: str = "vietnamese_legal",
        include_assistant_start: bool = True
    ) -> str:
        """
        Format input for LLM (decoder-only) models using chat templates.
        
        Args:
            context: Legal document context
            question: Question to answer
            answer: Target answer (for training)
            chat_template_name: Chat template to use
            instruction_template_name: Instruction template to use
            include_assistant_start: Whether to include assistant start token
            
        Returns:
            Formatted instruction string
        """
        chat_template = self.get_chat_template(chat_template_name)
        instruction_template = self.get_instruction_template(instruction_template_name)
        
        # Build user message
        user_message_parts = []
        
        if instruction_template.get("context_prefix"):
            user_message_parts.append(instruction_template["context_prefix"])
            user_message_parts.append(context)
        
        if instruction_template.get("question_prefix"):
            user_message_parts.append(instruction_template["question_prefix"])
            user_message_parts.append(question)
        
        user_message = "\n".join(user_message_parts)
        
        # Format complete instruction
        return chat_template.format_message(
            system_msg=instruction_template.get("system_message"),
            user_msg=user_message,
            assistant_msg=answer if answer else None,
            include_assistant_start=include_assistant_start
        )
    
    def format_for_inference(
        self,
        context: str,
        question: str,
        model_type: str = "llm",
        chat_template_name: str = "chatml",
        instruction_template_name: str = "vietnamese_legal"
    ) -> str:
        """
        Format input for inference (no target answer).
        
        Args:
            context: Legal document context
            question: Question to answer
            model_type: 'plm' or 'llm'
            chat_template_name: Chat template for LLMs
            instruction_template_name: Instruction template
            
        Returns:
            Formatted input for inference
        """
        if model_type == "plm":
            result = self.format_for_plm(context, question, "", instruction_template_name)
            return result["input_text"]
        else:
            return self.format_for_llm(
                context, question, "",
                chat_template_name, instruction_template_name,
                include_assistant_start=True
            )
    
    def extract_answer_from_response(
        self,
        response: str,
        chat_template_name: str = "chatml"
    ) -> str:
        """
        Extract answer from LLM response by removing chat formatting.
        
        Args:
            response: Raw response from LLM
            chat_template_name: Chat template used for generation
            
        Returns:
            Cleaned answer text
        """
        chat_template = self.get_chat_template(chat_template_name)
        
        # Remove assistant tokens
        cleaned = response
        
        # Remove assistant start/end tokens
        if chat_template.assistant_start in cleaned:
            cleaned = cleaned.split(chat_template.assistant_start)[-1]
        
        if chat_template.assistant_end in cleaned:
            cleaned = cleaned.split(chat_template.assistant_end)[0]
        
        # Remove other chat tokens that might appear
        tokens_to_remove = [
            chat_template.system_start, chat_template.system_end,
            chat_template.user_start, chat_template.user_end,
            "<|im_start|>", "<|im_end|>", "### ", "[INST]", "[/INST]"
        ]
        
        for token in tokens_to_remove:
            cleaned = cleaned.replace(token, "")
        
        # Clean up whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def validate_template_compatibility(
        self,
        model_name: str,
        chat_template_name: str
    ) -> bool:
        """
        Validate if a chat template is compatible with a model.
        
        Args:
            model_name: Name of the model
            chat_template_name: Name of the chat template
            
        Returns:
            True if compatible, False otherwise
        """
        model_lower = model_name.lower()
        
        # Model-specific compatibility checks
        compatibility_map = {
            "qwen": ["chatml", "seallm"],
            "seallm": ["chatml", "seallm"],
            "vinallama": ["vicuna", "alpaca", "llama2"],
            "llama": ["llama2", "vicuna", "alpaca"],
            "vicuna": ["vicuna"],
            "alpaca": ["alpaca"]
        }
        
        for model_family, compatible_templates in compatibility_map.items():
            if model_family in model_lower:
                if chat_template_name in compatible_templates:
                    return True
                else:
                    logger.warning(
                        f"Template '{chat_template_name}' may not be optimal for {model_name}. "
                        f"Recommended: {compatible_templates}"
                    )
                    return False
        
        # Default to True for unknown models
        logger.info(f"Unknown model family for {model_name}, assuming template compatibility")
        return True
    
    def get_recommended_template(self, model_name: str) -> str:
        """
        Get recommended chat template for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Recommended chat template name
        """
        model_lower = model_name.lower()
        
        # Model-specific recommendations
        if "qwen" in model_lower:
            return "chatml"
        elif "seallm" in model_lower:
            return "chatml"  # SeaLLM uses ChatML-like format
        elif "vinallama" in model_lower or "llama" in model_lower:
            return "llama2" if "llama-2" in model_lower else "vicuna"
        elif "vicuna" in model_lower:
            return "vicuna"
        elif "alpaca" in model_lower:
            return "alpaca"
        else:
            # Default recommendation
            logger.info(f"No specific recommendation for {model_name}, using ChatML")
            return "chatml"