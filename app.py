from utils.text_processor import (
    extract_text_from_docx,
    extract_text_from_pdf,  # ← EKLE
    split_into_sentences,
    find_keyword_contexts,
    clean_text
)

from utils.models import analyze_text_with_all_models
from utils.visualizer import (
    create_emotion_radar_chart,
    create_results_dataframe
)

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
    
    st.markdown("---")
    st.markdown("**Geliştirici:** laikaresearch")

# Ana içerik
tab1, tab2, tab3 = st.tabs(["📄 Dosya Yükle", "📊 Sonuçlar", "ℹ️ Hakkında"])

with tab1:
    st.header("Dosya Yükleme")
    
   uploaded_file = st.file_uploader(
    "Almanca haber dosyanızı yükleyin (.txt, .docx veya .pdf)",
    type=['txt', 'docx', 'pdf']  # ← pdf ekle
)
    
    if uploaded_file:
    # Dosyayı oku
    if uploaded_file.name.endswith('.docx'):
        text = extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith('.pdf'):  # ← EKLE
        text = extract_text_from_pdf(uploaded_file)  # ← EKLE
    else:
        text = uploaded_file.read().decode('utf-8')
        
        text = clean_text(text)
        
        st.success(f"✅ Dosya yüklendi: {uploaded_file.name}")
        st.info(f"📝 Toplam karakter: {len(text)}")
        
        # Önizleme
        with st.expander("📄 Metin Önizleme"):
            st.text(text[:1000] + "..." if len(text) > 1000 else text)
        
        # Analiz butonu
        if st.button("🚀 Analizi Başlat", type="primary"):
            with st.spinner("Analiz yapılıyor..."):
                
                # Cümlelere ayır
                sentences = split_into_sentences(text)
                st.info(f"📊 Toplam cümle: {len(sentences)}")
                
                # Anahtar kelime eşleşmelerini bul
                matches = find_keyword_contexts(
                    sentences, 
                    keywords,
                    context_before,
                    context_after
                )
                
                if not matches:
                    st.warning("❌ Anahtar kelime bulunamadı!")
                else:
                    st.success(f"✅ {len(matches)} eşleşme bulundu!")
                    
                    # Her eşleşme için analiz yap
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, match in enumerate(matches):
                        # Analiz
                        analysis = analyze_text_with_all_models(match['context'])
                        
                        results.append({
                            **match,
                            **analysis
                        })
                        
                        progress_bar.progress((idx + 1) / len(matches))
                    
                    # Sonuçları session state'e kaydet
                    st.session_state['results'] = results
                    st.session_state['analyzed'] = True
                    
                    st.success("✅ Analiz tamamlandı! 'Sonuçlar' sekmesine gidin.")

with tab2:
    st.header("📊 Analiz Sonuçları")
    
    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        results = st.session_state['results']
        
        st.info(f"📈 Toplam {len(results)} bağlam analiz edildi")
        
        # Her sonucu göster
        for idx, result in enumerate(results):
            with st.expander(f"🔍 Eşleşme {idx+1}: '{result['keyword']}' - Cümle {result['sentence_index']}"):
                
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
        st.info("👈 Önce 'Dosya Yükle' sekmesinden bir dosya yükleyin ve analiz başlatın.")

with tab3:
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
    """)

st.markdown("---")
st.markdown("💡 **İpucu:** Sidebar'dan anahtar kelimeleri ve context window ayarlarını özelleştirebilirsiniz.")
