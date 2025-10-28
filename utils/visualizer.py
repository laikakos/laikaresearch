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
        data.append({
            'Anahtar Kelime': result.get('keyword', ''),
            'Cümle': result.get('target_sentence', ''),
            'Model 1': result.get('model_1', {}).get('sentiment', ''),
            'Model 2': result.get('model_2', {}).get('sentiment', ''),
            'Model 3 (Top)': result.get('model_3', {}).get('top_emotions', [{}])[0].get('label', '') if result.get('model_3', {}).get('top_emotions') else ''
        })
    
    return pd.DataFrame(data)
