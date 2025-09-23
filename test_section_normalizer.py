from src.section_normalizer import standardize_sections

# Test cases
test_texts = [
    "Here are my SKILLS: Python, Java",
    "CORE COMPETENCIES: Leadership",
    "AREAS OF EXPERTISE: Data Science",
    "PROJECTS: Built a website",
    "RELEVANT PROJECTS: AI model",
    "EXPERIENCE: 5 years",
    "WORK HISTORY: Various jobs",
    "PROFESSIONAL EXPERIENCE: Tech roles",
    "my SKILLS are good",  # Should not replace
    "gaining SKILLS in ML",  # Should not replace
]

for text in test_texts:
    result = standardize_sections(text)
    print(f"Input: {text}")
    print(f"Output: {result}")
