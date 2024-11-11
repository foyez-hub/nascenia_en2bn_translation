from abc import ABC, abstractmethod
from translator import Translation
import os
from dotenv import load_dotenv
from typing import Optional
from enum import Enum

class Environment(Enum):
    LOCAL = "local"
    COLAB = "colab"

class BaseTranslationConfig(ABC):
    """Abstract base class for translation configurations"""
    
    @abstractmethod
    def get_repo_id(self) -> str:
        pass
    
    @abstractmethod
    def get_base_dir(self) -> Optional[str]:
        pass
    
    @abstractmethod
    def get_token(self) -> str:
        pass

class LocalTranslationConfig(BaseTranslationConfig):
    """Configuration for local environment"""
    
    def __init__(self):
        load_dotenv()
        
    def get_repo_id(self) -> str:
        return os.getenv('HUGGINGFACE_REPO_ID', "nascenia/bn2en_base")
    
    def get_base_dir(self) -> Optional[str]:
        return os.getenv('MODEL_BASE_DIR')
    
    def get_token(self) -> str:
        token = os.getenv('HUGGINGFACE_TOKEN')
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN is required in environment variables")
        return token

class ColabTranslationConfig(BaseTranslationConfig):
    """Configuration for Google Colab environment"""
    
    def __init__(self, hf_token: str, base_dir: str, repo_id: str):
        self.hf_token = hf_token
        self.base_dir = base_dir
        self.repo_id = repo_id
    
    def get_repo_id(self) -> str:
        return self.repo_id
    
    def get_base_dir(self) -> str:
        return self.base_dir
    
    def get_token(self) -> str:
        return self.hf_token

class TranslationSetup:
    """Main translation setup class"""
    
    def __init__(self, config: BaseTranslationConfig):
        """
        Initialize translation setup with configuration
        
        Args:
            config (BaseTranslationConfig): Configuration instance for the desired environment
        """
        self.config = config
        self.translation_system = None
    
    def initialize_system(self) -> bool:
        """Initialize the translation system"""
        try:
            self.translation_system = Translation(
                repo_id=self.config.get_repo_id(),
                base_dir=self.config.get_base_dir(),
                hf_token=self.config.get_token()
            )
            return self.translation_system.setup()
        except Exception as e:
            print(f"Error initializing translation system: {e}")
            return False
    
    def translate_text(self, text: str) -> Optional[str]:
        """
        Translate the given text
        
        Args:
            text (str): Text to translate
            
        Returns:
            Optional[str]: Translated text or None if translation fails
        """
        try:
            if not self.translation_system:
                if not self.initialize_system():
                    raise ValueError("Failed to initialize translation system")
                    
            return self.translation_system.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return None

def create_translator(env: Environment, **kwargs) -> TranslationSetup:
    """
    Factory function to create appropriate translator setup
    
    Args:
        env (Environment): Environment type (LOCAL or COLAB)
        **kwargs: Additional arguments for specific environments
            - hf_token (str): Required for COLAB environment
            - base_dir (str): Required for COLAB environment
            - repo_id (str): Required for COLAB environment
    
    Returns:
        TranslationSetup: Configured translation setup
    """
    if env == Environment.LOCAL:
        config = LocalTranslationConfig()
    elif env == Environment.COLAB:
        required_params = ['hf_token', 'base_dir', 'repo_id']
        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters for Colab setup: {', '.join(missing_params)}")
            
        config = ColabTranslationConfig(
            hf_token=kwargs['hf_token'],
            base_dir=kwargs['base_dir'],
            repo_id=kwargs['repo_id']
        )
    else:
        raise ValueError(f"Unsupported environment: {env}")
    
    return TranslationSetup(config)

# Usage examples
def get_local_translation(text: str) -> Optional[str]:
    """
    Translate text using local setup
    
    Args:
        text (str): Text to translate
        
    Returns:
        Optional[str]: Translated text or None if translation fails
    """
    try:
        translator = create_translator(Environment.LOCAL)
        return translator.translate_text(text)
    except Exception as e:
        print(f"Local translation failed: {e}")
        return None

def get_colab_translation(
    text: str,
    hf_token: str,
    base_dir: str,
    repo_id: str = "nascenia/bn2en_base"
) -> Optional[str]:
    """
    Translate text using Colab setup
    
    Args:
        text (str): Text to translate
        hf_token (str): Hugging Face token
        base_dir (str): Directory path for storing models
        repo_id (str): Hugging Face model repository ID
        
    Returns:
        Optional[str]: Translated text or None if translation fails
    """
    try:
        translator = create_translator(
            Environment.COLAB,
            hf_token=hf_token,
            base_dir=base_dir,
            repo_id=repo_id
        )
        return translator.translate_text(text)
    except Exception as e:
        print(f"Colab translation failed: {e}")
        return None