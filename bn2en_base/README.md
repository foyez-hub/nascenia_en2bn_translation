---
pipeline_tag: translation
---




## Installation

1. Install required packages:
```bash
pip install ctranslate2 sentencepiece huggingface_hub
```

2. Clone and use the complete implementation:
```python
from huggingface_hub import snapshot_download
import os
import ctranslate2
import sentencepiece as spm
import torch

class Translation:
    def __init__(self, repo_id, base_dir=None, device=None, hf_token=None):
        self.repo_id = repo_id
        self.base_dir = base_dir
        self.translator = None
        self.sp_source = None
        self.sp_target = None
        self.hf_token=hf_token

        # Extract repo name from repo_id (part after the '/')
        self.repo_name = repo_id.split('/')[-1]

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _get_model_directory(self):
        """Determine the model directory path based on configuration"""
        if self.base_dir:
            # If base_dir is provided, combine it with repo_name
            model_dir = os.path.join(self.base_dir, self.repo_name)
        else:
            # If no base_dir provided, use repo_name directly
            model_dir = self.repo_name

        return model_dir

    def setup(self):
        """Initialize the translation system"""
        try:
            # Download models
            model_path = self._download_models()
            if not model_path:
                return False

            # Setup paths
            ct_model_path = os.path.join(model_path, "")
            sp_source_path = os.path.join(model_path, "bn.model")
            sp_target_path = os.path.join(model_path, "en.model")

            # Initialize models
            self._initialize_models(ct_model_path, sp_source_path, sp_target_path)

            return True

        except Exception as e:
            print(f"Error setting up translation system: {e}")
            return False

    def _download_models(self):
        """Download models from Hugging Face"""
        try:
            print("Downloading models from Hugging Face...")
            model_dir = self._get_model_directory()

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_dir), exist_ok=True)

            model_path = snapshot_download(
                repo_id=self.repo_id,
                local_dir=model_dir,
                token=self.hf_token
            )
            print(f"Models downloaded successfully to: {model_path}")
            return model_path
        except Exception as e:
            print(f"Error downloading models: {e}")
            return None

    def _initialize_models(self, ct_model_path, sp_source_path, sp_target_path):
        """Initialize translation models"""
        # Initialize SentencePiece
        self.sp_source = spm.SentencePieceProcessor()
        self.sp_target = spm.SentencePieceProcessor()

        # Load SentencePiece models
        self.sp_source.load(sp_source_path)
        self.sp_target.load(sp_target_path)

        # Initialize CTranslate2
        self.translator = ctranslate2.Translator(ct_model_path, device=self.device)

    def translate(self, text):
        """Translate a single text"""
        try:
            # Tokenize
            source_tokens = self.sp_source.encode_as_pieces(text)

            # Translate
            translations = self.translator.translate_batch(
                [source_tokens],
                batch_type="tokens",
                max_batch_size=4096
            )

            # Get translation
            translated_tokens = translations[0].hypotheses[0]

            # Detokenize
            translation = self.sp_target.decode_pieces(translated_tokens)

            return translation

        except Exception as e:
            print(f"Error during translation: {e}")
            return None

# Usage example
if __name__ == "__main__":
    # Initialize translation system with optional base directory
    repo_id = "nascenia/bn2en_base"
    base_dir = "provide_your_base_dir"
    Hf_token="your_huggingface_token"
    
    
    # If no base directory provided, pass None
    base_dir = base_dir if base_dir else None
    
    translation_system = Translation(repo_id, base_dir=base_dir,hf_token=Hf_token)
    
    if translation_system.setup():
        print("Translation system ready")
        
        # Interactive translation
        while True:
            text = input("Enter text to translate (or 'q' to quit): ")
            if text.lower() == 'q':
                break
                
            translation = translation_system.translate(text)
            if translation:
                print(f"Translation: {translation}")
            else:
                print("Translation failed")
    else:
        print("Failed to setup translation system")

```

## Model Components

The repository contains the following components:
- `bn.model`: SentencePiece model for Bangla tokenization
- `en.model`: SentencePiece model for English tokenization
- CTranslate2 model files in the root directory

## Usage Examples

### Basic Translation
```python
# Initialize
repo_id = "nascenia/bn2en_base"
base_dir = "provide_your_base_dir"
translation_system = Translation(repo_id, base_dir=base_dir)
translation_system.setup()

# Single sentence translation
bengali_text = "আমি বাংলায় কথা বলি।"
english_translation = translation_system.translate(bengali_text)
print(f"Translation: {english_translation}")
```


## Model Configuration

- **Max Batch Size**: 4096 tokens
- **Device**: CPU (can be modified to use GPU)
- **Tokenization**: SentencePiece subword tokenization