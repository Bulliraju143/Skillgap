import unicodedata
import re

def normalize_text(text):
    # Normalize Unicode (accents → ASCII)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove tabs and carriage returns, replace with a single space
    text = re.sub(r'[\t\r]+', ' ', text)
    
    # Collapse multiple horizontal spaces (but preserve newlines \n)
    text = re.sub(r'[ ]{2,}', ' ', text).strip()
    
    # It is important NOT to use r'\s+' here, as it includes '\n'
    return text