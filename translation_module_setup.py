from translator import Translation
import os
from dotenv import load_dotenv
from typing import Optional

class TranslationSetup:
    def __init__(self):
        """Initialize translation setup with configuration from environment"""
        self.load_config()
        self.translation_system = None

    def load_config(self) -> None:
        """Load configuration from environment variables"""
        load_dotenv()
        
        self.repo_id = os.getenv('HUGGINGFACE_REPO_ID', "nascenia/bn2en_base")
        self.base_dir = os.getenv('MODEL_BASE_DIR', None)
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')

        if not self.hf_token:
            raise ValueError("HUGGINGFACE_TOKEN is required in environment variables")

    def initialize_system(self) -> bool:
        """Initialize the translation system"""
        try:
            self.translation_system = Translation(
                repo_id=self.repo_id,
                base_dir=self.base_dir,
                hf_token=self.hf_token
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

# Usage example
def get_translation(text: str) -> Optional[str]:
    """
    Main function to translate text
    
    Args:
        text (str): Text to translate
        
    Returns:
        Optional[str]: Translated text or None if translation fails
    """
    try:
        translator = TranslationSetup()
        return translator.translate_text(text)
    except Exception as e:
        print(f"Translation failed: {e}")
        return None

