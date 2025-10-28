import re
from typing import List, Tuple
import docx
import PyPDF2

def extract_text_from_docx(file):
    """DOCX dosyasından metin çıkar"""
    doc = docx.Document(file)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_pdf(file):
    """PDF dosyasından metin çıkar"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_into_sentences(text: str) -> List[str]:
    """Metni cümlelere ayır"""
    # Almanca cümle sonu işaretleri
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def find_keyword_contexts(sentences: List[str], keywords: List[str], 
                         context_before: int = 3, context_after: int = 3) -> List[dict]:
    """
    Anahtar kelimelerin geçtiği cümleleri ve context'lerini bul
    
    Args:
        sentences: Cümle listesi
        keywords: Arama yapılacak anahtar kelimeler
        context_before: Önceki kaç cümle
        context_after: Sonraki kaç cümle
    
    Returns:
        Her match için dict listesi
    """
    matches = []
    
    for i, sentence in enumerate(sentences):
        # Anahtar kelime kontrolü (case-insensitive)
        for keyword in keywords:
            if keyword.lower() in sentence.lower():
                # Context penceresi
                start_idx = max(0, i - context_before)
                end_idx = min(len(sentences), i + context_after + 1)
                
                context_sentences = sentences[start_idx:end_idx]
                
                matches.append({
                    'keyword': keyword,
                    'sentence_index': i,
                    'target_sentence': sentence,
                    'context': ' '.join(context_sentences),
                    'context_sentences': context_sentences,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
                break  # Her cümle için sadece bir match
    
    return matches

def clean_text(text: str) -> str:
    """Metni temizle"""
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
