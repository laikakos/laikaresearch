import streamlit as st
import pandas as pd
from utils.text_processor import (
    extract_text_from_docx,
    extract_text_from_pdf,
    split_into_sentences,
    find_keyword_contexts,
    clean_text
)
from utils.models import analyze_text_with_all_models
from utils.visualizer import (
    create_emotion_radar_chart,
    create_results_dataframe,
    create_sentiment_distribution_chart,
    create_file_summary_chart,
    create_keyword_summary_chart
)
import os

# Sayfa ayarları
st.set_page_config(
    page_title="Qatar Sentiment Analysis",
    page_icon="🏆",
    layout="wide"
)

# Başlık
st.title("🏆 Qatar Dünya Kupası Duygu Analizi")
st.markdown("### Almanca Haber Metinleri için Çoklu Model Karşılaştırması")

st.markdown("""
Bu uygulama 3 farklı model ile Almanca metinlerde duygu analizi yapar:
- **Model 1:** Hızlı Pilot (Guhr et al. LREC 2020)
- **Model 2:** Haber Metinleri (mdraw - özelleştirilmiş)
- **Model 3:** Detaylı Akademik (GoEmotions - 27 duygu)
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # Anahtar kelimeler
    st.subheader("Anahtar Kelimeler")
    keywords_input = st.text_area(
        "Her satıra bir kelime",
        value="Qatar\nKatar\nWeltmeisterschaft\nFußball\nWM",
        height=150
    )
    keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
    
    # Context window
    st.subheader("Context Window")
    context_before = st.slider("Önceki cümle sayısı", 0, 5, 3)
    context_after = st.slider("Sonraki cümle sayısı", 0, 5, 3)
    
    # Batch processing ayarı
    st.subheader("Toplu İşleme")
    batch_size = st.number_input("Batch boyutu", min_value=10, max_value=500, value=100, step=10)
    st.info(f"Her {batch_size} dosya için ilerleme gösterilecek")
    
    st.markdown("---")
    st.markdown("**Geliştirici:** laikaresearch")

# Dosya tipini kontrol eden yardımcı fonksiyon
def get_file_extension(filename):
    """Dosya uzantısını güvenli şekilde al"""
    # Son noktadan sonrasını al (uzantı)
    if '.' in filename:
        return filename.rsplit('.', 1)[-1].lower()
    return ''

# Ana içerik
tab1, tab2, tab3, tab4 = st.tabs(["📄 Dosya Yükle", "📊 Sonuçlar", "📈 İstatistikler", "ℹ️ Hakkında"])

with tab1:
    st.header("Dosya Yükleme")
    
    uploaded_files = st.file_uploader(
        "Almanca haber dosyalarınızı yükleyin (.txt, .docx veya .pdf)",
        type=['txt', 'docx', 'pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.warning(f"⚠️ {len(uploaded_files)} dosya yüklendi. Büyük dosya sayısı için işlem uzun sürebilir.")
        
        # Tüm dosyaları birleştir
        all_texts = []
        
        # Dosya yükleme progress bar
        file_progress = st.progress(0)
        file_status = st.empty()
        
        failed_files = []
        
        for file_idx, uploaded_file in enumerate(uploaded_files):
            try:
                file_status.text(f"Dosya okunuyor: {uploaded_file.name} ({file_idx+1}/{len(uploaded_files)})")
                
                # Dosya uzantısını güvenli şekilde al
                file_extension = get_file_extension(uploaded_file.name)
                
                # Dosya tipine göre okuma
                if file_extension == 'docx':
                    text = extract_text_from_docx(uploaded_file)
                elif file_extension == 'pdf':
                    text = extract_text_from_pdf(uploaded_file)
                elif file_extension == 'txt':
                    # TXT dosyaları için encoding denemesi
                    try:
                        text = uploaded_file.read().decode('utf-8')
                    except UnicodeDecodeError:
                        # UTF-8 başarısız olursa latin-1 dene
                        uploaded_file.seek(0)  # Dosya pozisyonunu başa al
                        try:
                            text = uploaded_file.read().decode('latin-1')
                        except:
                            uploaded_file.seek(0)
                            text = uploaded_file.read().decode('cp1252')  # Windows encoding
                else:
                    failed_files.append((uploaded_file.name, f"Desteklenmeyen dosya tipi: .{file_extension}"))
                    continue
                
                # Temizlenmiş metni kaydet
                cleaned_text = clean_text(text)
                
                if len(cleaned_text.strip()) > 0:
                    all_texts.append({
                        'filename': uploaded_file.name,
                        'text': cleaned_text
                    })
                else:
                    failed_files.append((uploaded_file.name, "Boş dosya"))
                
            except Exception as e:
                failed_files.append((uploaded_file.name, str(e)))
                st.error(f"❌ Hata: {uploaded_file.name} - {str(e)[:100]}")
            
            # Progress güncelle
            file_progress.progress((file_idx + 1) / len(uploaded_files))
        
        file_status.empty()
        file_progress.empty()
        
        # Sonuç özeti
        if all_texts:
            st.success(f"✅ {len(all_texts)} dosya başarıyla yüklendi")
        
        if failed_files:
            st.error(f"❌ {len(failed_files)} dosya yüklenemedi")
            with st.expander("Başarısız Dosyalar"):
                for fname, error in failed_files:
                    st.write(f"- **{fname}**: {error}")
        
        if not all_texts:
            st.error("❌ Hiçbir dosya başarıyla yüklenemedi!")
        else:
            # Toplam istatistikler
            total_chars = sum(len(t['text']) for t in all_texts)
            st.info(f"📝 Toplam karakter: {total_chars:,}")
            
            # Dosya listesi (ilk 20 dosya)
            with st.expander(f"📂 Yüklenen Dosyalar (İlk 20/{len(all_texts)})"):
                for i, item in enumerate(all_texts[:20]):
                    st.write(f"{i+1}. **{item['filename']}** - {len(item['text']):,} karakter")
                if len(all_texts) > 20:
                    st.write(f"... ve {len(all_texts) - 20} dosya daha")
            
            # Önizleme
            with st.expander("📄 İlk Dosya Önizleme"):
                preview_text = all_texts[0]['text']
                st.text(preview_text[:1000] + "..." if len(preview_text) > 1000 else preview_text)
            
            # Analiz butonu
            if st.button("🚀 Analizi Başlat", type="primary"):
                with st.spinner("Analiz yapılıyor..."):
                    
                    all_results = []
                    
                    # Genel progress bar
                    overall_progress = st.progress(0)
                    overall_status = st.empty()
                    
                    # Her dosya için analiz
                    for file_idx, item in enumerate(all_texts):
                        overall_status.text(f"📄 Analiz ediliyor: {item['filename']} ({file_idx+1}/{len(all_texts)})")
                        
                        try:
                            # Cümlelere ayır
                            sentences = split_into_sentences(item['text'])
                            
                            # Anahtar kelime eşleşmelerini bul
                            matches = find_keyword_contexts(
                                sentences, 
                                keywords,
                                context_before,
                                context_after
                            )
                            
                            if matches:
                                # Her eşleşme için analiz yap
                                for idx, match in enumerate(matches):
                                    # Analiz
                                    analysis = analyze_text_with_all_models(match['context'])
                                    
                                    all_results.append({
                                        'filename': item['filename'],
                                        **match,
                                        **analysis
                                    })
                            
                        except Exception as e:
                            st.warning(f"⚠️ Analiz hatası: {item['filename']} - {str(e)[:100]}")
                        
                        # Overall progress güncelle
                        overall_progress.progress((file_idx + 1) / len(all_texts))
                    
                    overall_status.empty()
                    overall_progress.empty()
                    
                    if all_results:
                        # Sonuçları session state'e kaydet
                        st.session_state['results'] = all_results
                        st.session_state['analyzed'] = True
                        
                        st.success(f"✅ Analiz tamamlandı! Toplam {len(all_results)} eşleşme bulundu. 'Sonuçlar' sekmesine gidin.")
                    else:
                        st.error("❌ Hiçbir dosyada anahtar kelime bulunamadı!")

with tab2:
    st.header("📊 Detaylı Sonuçlar")
    
    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        results = st.session_state['results']
        
        st.info(f"📈 Toplam {len(results)} bağlam analiz edildi")
        
        # Filtreleme seçenekleri
        with st.expander("🔍 Filtreleme Seçenekleri"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Dosyaya göre filtrele
                all_files = sorted(list(set([r.get('filename', 'N/A') for r in results])))
                selected_files = st.multiselect(
                    "Dosya Seç",
                    options=all_files,
                    default=all_files[:5] if len(all_files) > 5 else all_files
                )
            
            with col2:
                # Anahtar kelimeye göre filtre
                all_keywords = sorted(list(set([r['keyword'] for r in results])))
                selected_keywords = st.multiselect(
                    "Anahtar Kelime Seç",
                    options=all_keywords,
                    default=all_keywords
                )
        
        # Filtrelenmiş sonuçlar
        filtered_results = [
            r for r in results 
            if r.get('filename', 'N/A') in selected_files and r['keyword'] in selected_keywords
        ]
        
        st.info(f"🔎 Gösterilen: {len(filtered_results)} / {len(results)}")
        
        # Her sonucu göster
        for idx, result in enumerate(filtered_results):
            with st.expander(f"🔍 {result.get('filename', 'N/A')} - Eşleşme {idx+1}: '{result['keyword']}' - Cümle {result['sentence_index']}"):
                
                # Context göster
                st.markdown("**📝 Bağlam:**")
                st.write(result['context'])
                
                st.markdown("---")
                
                # Model sonuçları
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Model 1: Hızlı Pilot**")
                    sentiment_1 = result['model_1']['sentiment']
                    st.metric("Sentiment", sentiment_1)
                
                with col2:
                    st.markdown("**Model 2: Haber**")
                    sentiment_2 = result['model_2']['sentiment']
                    st.metric("Sentiment", sentiment_2)
                
                with col3:
                    st.markdown("**Model 3: Detaylı**")
                    top_emotion = result['model_3']['top_emotions'][0]
                    st.metric(
                        "Top Duygu", 
                        top_emotion['label'],
                        f"{top_emotion['score']:.2%}"
                    )
                
                # Model 3 detayları
                st.markdown("**🎭 Top 5 Duygu (Model 3):**")
                emotion_df = pd.DataFrame(result['model_3']['top_emotions'])
                st.dataframe(emotion_df, use_container_width=True)
                
                # Radar chart
                fig = create_emotion_radar_chart(result['model_3']['top_emotions'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Özet tablo
        st.markdown("---")
        st.subheader("📋 Özet Tablo")
        df = create_results_dataframe(results)
        st.dataframe(df, use_container_width=True)
        
        # CSV indirme
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Sonuçları CSV olarak indir",
            data=csv,
            file_name="qatar_sentiment_results.csv",
            mime="text/csv"
        )
        
    else:
        st.info("👈 Önce 'Dosya Yükle' sekmesinden dosya yükleyin ve analiz başlatın.")

with tab3:
    st.header("📈 Genel İstatistikler ve Görselleştirmeler")
    
    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        results = st.session_state['results']
        
        # Özet metrikler
        st.subheader("📊 Özet Metrikler")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Eşleşme", len(results))
        
        with col2:
            unique_files = len(set([r.get('filename', 'N/A') for r in results]))
            st.metric("Analiz Edilen Dosya", unique_files)
        
        with col3:
            unique_keywords = len(set([r['keyword'] for r in results]))
            st.metric("Bulunan Anahtar Kelime", unique_keywords)
        
        with col4:
            avg_per_file = len(results) / unique_files if unique_files > 0 else 0
            st.metric("Dosya Başına Ort. Eşleşme", f"{avg_per_file:.1f}")
        
        st.markdown("---")
        
        # Grafikler
        st.subheader("📊 Görselleştirmeler")
        
        # 1. Sentiment Dağılımı
        st.markdown("### 1️⃣ Sentiment Dağılımı (Model 1 & 2)")
        fig_sentiment = create_sentiment_distribution_chart(results)
        if fig_sentiment:
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        st.markdown("---")
        
        # 2. Dosya Bazlı Analiz
        st.markdown("### 2️⃣ Dosya Bazlı Eşleşme Sayıları")
        fig_files = create_file_summary_chart(results)
        if fig_files:
            st.plotly_chart(fig_files, use_container_width=True)
        
        # Dosya detay tablosu
        with st.expander("📂 Tüm Dosyalar - Detaylı Tablo"):
            file_summary = {}
            for r in results:
                fname = r.get('filename', 'N/A')
                if fname not in file_summary:
                    file_summary[fname] = {
                        'Eşleşme Sayısı': 0,
                        'Positive (M1)': 0,
                        'Negative (M1)': 0,
                        'Neutral (M1)': 0
                    }
                file_summary[fname]['Eşleşme Sayısı'] += 1
                sentiment = r.get('model_1', {}).get('sentiment', '').lower()
                if 'positive' in sentiment:
                    file_summary[fname]['Positive (M1)'] += 1
                elif 'negative' in sentiment:
                    file_summary[fname]['Negative (M1)'] += 1
                elif 'neutral' in sentiment:
                    file_summary[fname]['Neutral (M1)'] += 1
            
            file_df = pd.DataFrame.from_dict(file_summary, orient='index')
            file_df = file_df.sort_values('Eşleşme Sayısı', ascending=False)
            st.dataframe(file_df, use_container_width=True)
        
        st.markdown("---")
        
        # 3. Anahtar Kelime Analizi
        st.markdown("### 3️⃣ Anahtar Kelime Bazlı Eşleşmeler")
        fig_keywords = create_keyword_summary_chart(results)
        if fig_keywords:
            st.plotly_chart(fig_keywords, use_container_width=True)
        
        # Anahtar kelime detay tablosu
        with st.expander("🔑 Anahtar Kelimeler - Detaylı Tablo"):
            keyword_summary = {}
            for r in results:
                kw = r.get('keyword', 'N/A')
                if kw not in keyword_summary:
                    keyword_summary[kw] = {
                        'Eşleşme Sayısı': 0,
                        'Positive (M1)': 0,
                        'Negative (M1)': 0,
                        'Neutral (M1)': 0
                    }
                keyword_summary[kw]['Eşleşme Sayısı'] += 1
                sentiment = r.get('model_1', {}).get('sentiment', '').lower()
                if 'positive' in sentiment:
                    keyword_summary[kw]['Positive (M1)'] += 1
                elif 'negative' in sentiment:
                    keyword_summary[kw]['Negative (M1)'] += 1
                elif 'neutral' in sentiment:
                    keyword_summary[kw]['Neutral (M1)'] += 1
            
            keyword_df = pd.DataFrame.from_dict(keyword_summary, orient='index')
            keyword_df = keyword_df.sort_values('Eşleşme Sayısı', ascending=False)
            st.dataframe(keyword_df, use_container_width=True)
        
        st.markdown("---")
        
        # 4. Model 3 - Top Duygular
        st.markdown("### 4️⃣ En Sık Görülen Duygular (Model 3)")
        all_emotions = {}
        for r in results:
            top_emotion = r.get('model_3', {}).get('top_emotions', [{}])[0]
            emotion_label = top_emotion.get('label', 'unknown')
            if emotion_label != 'unknown':
                all_emotions[emotion_label] = all_emotions.get(emotion_label, 0) + 1
        
        if all_emotions:
            emotion_series = pd.Series(all_emotions).sort_values(ascending=False)
            
            import plotly.graph_objects as go
            fig_emotions = go.Figure()
            fig_emotions.add_trace(go.Bar(
                x=emotion_series.index[:15],  # Top 15
                y=emotion_series.values[:15],
                marker_color='purple'
            ))
            fig_emotions.update_layout(
                title='En Sık Tespit Edilen 15 Duygu (Model 3)',
                xaxis_title='Duygu',
                yaxis_title='Frekans',
                height=400
            )
            st.plotly_chart(fig_emotions, use_container_width=True)
        
    else:
        st.info("👈 Önce 'Dosya Yükle' sekmesinden dosya yükleyin ve analiz başlatın.")

with tab4:
    st.header("ℹ️ Proje Hakkında")
    
    st.markdown("""
    ### 🎯 Amaç
    Qatar Dünya Kupası ile ilgili Almanca haber metinlerinde duygu analizi yapmak ve 
    farklı modellerin performanslarını karşılaştırmak.
    
    ### 📚 Kullanılan Modeller
    
    **1. Model 1: Hızlı Pilot**
    - Oliver Guhr et al. (LREC 2020)
    - 1.8M Almanca örnek
    - 3 kategori: positive, negative, neutral
    - F1 Score: ~0.96
    
    **2. Model 2: Haber Metinleri**
    - mdraw/german-news-sentiment-bert
    - Haber dili için özelleştirilmiş
    - 2007-2019 göç haberleri ile eğitilmiş
    
    **3. Model 3: Detaylı Akademik**
    - GoEmotions (Google, ACL 2020)
    - 27 farklı duygu kategorisi
    - Çok dilli BERT
    
    ### 🔬 Metodoloji
    - **Context Window:** Anahtar kelimenin 3 cümle öncesi ve sonrası
    - **Çoklu Model:** Üç farklı yaklaşımın karşılaştırması
    - **Akademik Standart:** Peer-reviewed modeller
    
    ### 📖 Kaynaklar
    - [Guhr et al. 2020 - LREC](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.202.pdf)
    - [GoEmotions - ACL 2020](https://aclanthology.org/2020.acl-main.372/)
    - [GitHub Repository](https://github.com/laikakos/laikaresearch)
    
    ### ⚡ Performans İpuçları (2500 PDF için)
    - Dosyalar batch olarak işlenir
    - Her 100 dosyada ilerleme gösterilir
    - Hatalı dosyalar atlanır ve listelenir
    - Toplam süre: ~30-60 dakika (dosya boyutuna bağlı)
    
    ### 📊 Yeni Özellikler
    - **İstatistikler Sekmesi:** Genel görselleştirmeler ve trendler
    - **Dosya Bazlı Analiz:** Her dosyanın detaylı sentiment dağılımı
    - **Anahtar Kelime Analizi:** Hangi kelime ne kadar etkili
    - **Model 3 Duygu Haritası:** En sık görülen 15 duygu
    """)

st.markdown("---")
st.markdown("💡 **İpucu:** Sidebar'dan anahtar kelimeleri ve context window ayarlarını özelleştirebilirsiniz.")
