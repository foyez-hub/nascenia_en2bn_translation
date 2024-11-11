from translation_module_setup import get_translation
print("hello")
text = "আপনার বাংলা টেক্সট"
translated_text = get_translation(text)
if translated_text:
    print(translated_text)
