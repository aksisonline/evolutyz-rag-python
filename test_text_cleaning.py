#!/usr/bin/env python3
"""
Simple test script to verify text cleaning functionality
"""

import re

def clean_text(text: str) -> str:
    """Clean and normalize text content more aggressively"""
    
    if not text:
        return ""
    
    # First, normalize the text
    text = text.strip()
    
    # Remove excessive brackets, quotes, and special characters
    text = re.sub(r'["\'""`''""'']+', '', text)  # Remove all types of quotes
    text = re.sub(r'\[+[^\]]*\]+', '', text)     # Remove content in square brackets
    text = re.sub(r'\{+[^}]*\}+', '', text)      # Remove content in curly brackets
    text = re.sub(r'\(+[^)]*\)+', ' ', text)     # Replace parentheses content with space
    
    # Remove asterisks and stars (often used for emphasis in raw text)
    text = re.sub(r'\*+', ' ', text)
    
    # Clean up commas and normalize punctuation
    text = re.sub(r'\s*,\s*', ', ', text)        # Normalize commas
    text = re.sub(r'\s*\.\s*', '. ', text)       # Normalize periods
    text = re.sub(r'\s*;\s*', '; ', text)        # Normalize semicolons
    text = re.sub(r'\s*:\s*', ': ', text)        # Normalize colons
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)             # Multiple spaces to single space
    text = re.sub(r'\n\s*\n', '\n', text)        # Multiple newlines to single
    
    # Remove common PDF/document artifacts
    text = re.sub(r'(?:page \d+|Page \d+|pg\. \d+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:figure \d+|table \d+|chart \d+|fig\. \d+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:see page|see fig|see table)', '', text, flags=re.IGNORECASE)
    
    # Remove multiple consecutive punctuation
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[,]{2,}', ',', text)
    text = re.sub(r'[-]{2,}', '-', text)
    
    # Remove leading/trailing punctuation fragments
    text = re.sub(r'^[,.\-;:\s]+', '', text)
    text = re.sub(r'[,.\-;:\s]+$', '', text)
    
    # Ensure sentences start with capital letters
    sentences = text.split('. ')
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 2:
            sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
            cleaned_sentences.append(sentence)
    
    result = '. '.join(cleaned_sentences)
    
    # Final cleanup
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

# Test with sample messy text
if __name__ == "__main__":
    sample_text = """
    [DOC 1] "Abhiram Kanna" is a creative professional passionate about problem-solving, continuous learning, and collaborative development. He is pursuing a *Bachelor's* in Artificial Intelligence and Machine Learning from Gandhi Institute of Technology and Management (2021-2025). **Databases**: MySQL, MongoDB, PostgreSQL **Media**, enhancing content strategies and brand visibility.
    """
    
    print("Original text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    print("Cleaned text:")
    print(clean_text(sample_text))
