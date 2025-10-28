from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from germansentiment import SentimentModel
import streamlit as st

@st.cache_resource
def load_model_1():
    """Model 1: Hızlı Pilot - oliverguhr/german-sentiment-bert"""
    model = SentimentModel()
    return model

@st.cache_resource
def load_model_2():
    """Model 2: Haber Metinleri - mdraw/german-news-sentiment-bert"""
    model = SentimentModel('mdraw/german-news-sentiment-bert')
    return model

@st.cache_resource
def load_model_3():
    """Model 3: Detaylı - GoEmotions (27 duygu)"""
    tokenizer = AutoTokenizer.from_pretrained("SchuylerH/bert-multilingual-go-emtions")
    model = AutoModelForSequenceClassification.from_pretrained("SchuylerH/bert-multilingual-go-emtions")
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

def analyze_with_model_1(text: str, model) -> dict:
    """Model 1 ile analiz"""
    result = model.predict_sentiment([text])
    sentiment = result[0]
    
    return {
        'model': 'Hızlı Pilot (Guhr et al. 2020)',
        'sentiment': sentiment,
        'categories': {'positive': 0, 'negative': 0, 'neutral': 0}
    }

def analyze_with_model_2(text: str, model) -> dict:
    """Model 2 ile analiz"""
    result = model.predict_sentiment([text])
    sentiment = result[0]
    
    return {
        'model': 'Haber Metinleri (mdraw)',
        'sentiment': sentiment,
        'categories': {'positive': 0, 'negative': 0, 'neutral': 0}
    }

def analyze_with_model_3(text: str, pipeline_model) -> dict:
    """Model 3 ile analiz (27 duygu)"""
    results = pipeline_model(text[:512])[0]  # Token limiti
    
    # En yüksek skorlu 5 duyguyu al
    top_emotions = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
    
    return {
        'model': 'Detaylı GoEmotions (27 duygu)',
        'top_emotions': top_emotions,
        'all_emotions': results
    }

def analyze_text_with_all_models(text: str) -> dict:
    """Tüm modellerle analiz yap"""
    
    # Modelleri yükle
    model_1 = load_model_1()
    model_2 = load_model_2()
    model_3 = load_model_3()
    
    # Analizler
    result_1 = analyze_with_model_1(text, model_1)
    result_2 = analyze_with_model_2(text, model_2)
    result_3 = analyze_with_model_3(text, model_3)
    
    return {
        'model_1': result_1,
        'model_2': result_2,
        'model_3': result_3
    }
