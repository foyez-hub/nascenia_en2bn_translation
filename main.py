from translation_module_setup import get_local_translation
# Create a .env file with your configurations
text = "আমার সোনার বাংলা"
translated_text = get_local_translation(text)
print(translated_text)