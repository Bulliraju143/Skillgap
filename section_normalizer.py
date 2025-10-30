import re
from src.txt_cleaner import normalize_text as basic_normalize

def standardize_sections(text):
    """
    with the '=== HEADER ===' format.
    """
    def replace_if_line_alone(text, section_names, replacement):
        # The pattern matches only when the section header is on a line by itself,
        # possibly followed by a colon or dot, and surrounded by whitespace.
        # It uses the MULTILINE flag (re.MULTILINE or re.M) for ^ and $ to match start/end of a line.
        pattern = r'^\s*(' + '|'.join(section_names) + r')\s*[:.]?\s*$'
        return re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.MULTILINE)

    # Standardize the headers using the line-alone constraint
    text = replace_if_line_alone(text, ['EDUCATION'], '=== EDUCATION ===')
    text = replace_if_line_alone(text, ['SKILLS', 'CORE COMPETENCIES', 'AREAS OF EXPERTISE'], '=== SKILLS ===')
    text = replace_if_line_alone(text, ['PROJECTS', 'RELEVANT PROJECTS', 'PORTFOLIO'], '=== PROJECTS ===')
    text = replace_if_line_alone(text, ['EXPERIENCE', 'WORK HISTORY', 'PROFESSIONAL EXPERIENCE', 'INTERNSHIP'], '=== EXPERIENCE ===')

    # Final cleanup: Collapse multiple horizontal spaces (preserves newlines)
    text = re.sub(r'[ ]{2,}', ' ', text)
    return text.strip()


def normalize_text(text):
    """
    The main wrapper function for text and section normalization.
    """
    # 1. Basic cleaning (replaces tabs, cleans unicode, preserves newlines)
    text = basic_normalize(text)

    # 2. Standardize sections (only replaces headers on their own line)
    text = standardize_sections(text)

    # 3. Clean up any extra empty lines that resulted from section replacement
    text = re.sub(r'\n{2,}', '\n', text)

    return text
