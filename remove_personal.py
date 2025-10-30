import re

def remove_personal(text):
    # Remove emails
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    # Remove phone numbers (basic pattern)
    text = re.sub(r'\+?\d[\d\-\s]{7,}\d', '[PHONE]', text)
    
    # Remove dates with formats like mm/yyyy, mm/yyyy-mm/yyyy, yyyy-mm-dd, Month dd, yyyy, etc.
    date_pattern = r'(\b\d{1,2}[/\-]\d{2,4}\b|\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b)'
    text = re.sub(date_pattern, '[DATE]', text, flags=re.IGNORECASE)

    return text
