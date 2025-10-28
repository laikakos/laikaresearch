import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_sentiment_comparison_chart(results: dict):
    """3 model için sentiment karşılaştırma grafiği"""
    
    models = ['Model 1: Hızlı', 'Model 2: Haber', 'Model 3: Detaylı']
    sentiments_1 = results['model_1']['sentiment']
    sentiments_2 = results['model_2']['sentiment']
    
    # Basit bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Model 1',
        x=[sentiments_1],
        y=['Sentiment'],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Model Karşılaştırması',
        barmode='group',
        height=300
    )
    
    return fig

def create_emotion_radar_chart(emotions: list):
    """Duygu radar grafiği (Model 3 için)"""
    
    if not emotions:
        return None
    
    labels = [e['label'] for e in emotions[:8]]
    values = [e['score'] for e in emotions[:8]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Duygular'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title='Duygu Dağılımı (Model 3)',
        height=400
    )
    
    return fig

def create_results_dataframe(all_results: list) -> pd.DataFrame:
    """Tüm sonuçları DataFrame'e dönüştür"""
    
    data = []
    for result in all_results:
        # Model 3 top emotion güvenli şekilde al
        model_3_top = ''
        model_3_score = ''
        if result.get('model_3', {}).get('top_emotions'):
            top_emotion = result['model_3']['top_emotions'][0]
            model_3_top = top_emotion.get('label', '')
            model_3_score = f"{top_emotion.get('score', 0):.2%}"
        
        data.append({
            'Dosya': result.get('filename', 'N/A'),  # YENİ: Dosya adı eklendi
            'Anahtar Kelime': result.get('keyword', ''),
            'Cümle No': result.get('sentence_index', ''),  # YENİ: Cümle numarası
            'Hedef Cümle': result.get('target_sentence', '')[:100] + '...' if len(result.get('target_sentence', '')) > 100 else result.get('target_sentence', ''),
            'Context (İlk 100 kar)': result.get('context', '')[:100] + '...' if len(result.get('context', '')) > 100 else result.get('context', ''),
            'Model 1 (Pilot)': result.get('model_1', {}).get('sentiment', ''),
            'Model 2 (Haber)': result.get('model_2', {}).get('sentiment', ''),
            'Model 3 (Top Duygu)': model_3_top,
            'Model 3 (Skor)': model_3_score
        })
    
    return pd.DataFrame(data)

def create_sentiment_distribution_chart(all_results: list):
    """Tüm sonuçlar için sentiment dağılımı grafiği"""
    
    if not all_results:
        return None
    
    # Model 1 dağılımı
    model_1_sentiments = [r.get('model_1', {}).get('sentiment', '') for r in all_results]
    sentiment_counts_1 = pd.Series(model_1_sentiments).value_counts()
    
    # Model 2 dağılımı
    model_2_sentiments = [r.get('model_2', {}).get('sentiment', '') for r in all_results]
    sentiment_counts_2 = pd.Series(model_2_sentiments).value_counts()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Model 1 (Pilot)',
        x=sentiment_counts_1.index,
        y=sentiment_counts_1.values,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Model 2 (Haber)',
        x=sentiment_counts_2.index,
        y=sentiment_counts_2.values,
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Sentiment Dağılımı (Tüm Sonuçlar)',
        xaxis_title='Sentiment',
        yaxis_title='Sayı',
        barmode='group',
        height=400
    )
    
    return fig

def create_file_summary_chart(all_results: list):
    """Dosya bazlı özet grafiği"""
    
    if not all_results:
        return None
    
    # Dosya başına eşleşme sayısı
    file_counts = pd.Series([r.get('filename', 'N/A') for r in all_results]).value_counts()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=file_counts.index[:20],  # İlk 20 dosya
        y=file_counts.values[:20],
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Dosya Başına Eşleşme Sayısı (İlk 20)',
        xaxis_title='Dosya',
        yaxis_title='Eşleşme Sayısı',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_keyword_summary_chart(all_results: list):
    """Anahtar kelime bazlı özet grafiği"""
    
    if not all_results:
        return None
    
    # Anahtar kelime başına eşleşme sayısı
    keyword_counts = pd.Series([r.get('keyword', '') for r in all_results]).value_counts()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=keyword_counts.index,
        y=keyword_counts.values,
        marker_color='orange'
    ))
    
    fig.update_layout(
        title='Anahtar Kelime Başına Eşleşme Sayısı',
        xaxis_title='Anahtar Kelime',
        yaxis_title='Eşleşme Sayısı',
        height=400
    )
    
    return fig
