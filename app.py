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
                        uploaded_file.seek(0)
                        try:
                            text = uploaded_file.read().decode('latin-1')
                        except:
                            uploaded_file.seek(0)
                            text = uploaded_file.read().decode('cp1252')
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
                    
                    # YENİ: İlerleme detay metrikleri
                    progress_cols = st.columns(4)
                    metric_files = progress_cols[0].empty()
                    metric_matches = progress_cols[1].empty()
                    metric_analyzed = progress_cols[2].empty()
                    metric_time = progress_cols[3].empty()
                    
                    import time
                    start_time = time.time()
                    
                    # Her dosya için analiz
                    for file_idx, item in enumerate(all_texts):
                        # YENİ: Daha detaylı ilerleme bilgisi
                        elapsed_time = time.time() - start_time
                        avg_time_per_file = elapsed_time / (file_idx + 1) if file_idx > 0 else 0
                        remaining_files = len(all_texts) - file_idx - 1
                        estimated_remaining = avg_time_per_file * remaining_files
                        
                        overall_status.text(
                            f"📄 Analiz ediliyor: {item['filename']} "
                            f"({file_idx+1}/{len(all_texts)}) - "
                            f"Toplam eşleşme: {len(all_results)}"
                        )
                        
                        # Metrikler güncelle
                        metric_files.metric("İşlenen Dosya", f"{file_idx+1}/{len(all_texts)}")
                        metric_matches.metric("Bulunan Eşleşme", len(all_results))
                        metric_analyzed.metric("Analiz Edilen", len(all_results))
                        if estimated_remaining > 0:
                            mins, secs = divmod(int(estimated_remaining), 60)
                            metric_time.metric("Tahmini Kalan", f"{mins}d {secs}s")
                        
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
                                    
                                    # Her 10 eşleşmede metrikleri güncelle
                                    if idx % 10 == 0:
                                        metric_matches.metric("Bulunan Eşleşme", len(all_results))
                                        metric_analyzed.metric("Analiz Edilen", len(all_results))
                            
                        except Exception as e:
                            st.warning(f"⚠️ Analiz hatası: {item['filename']} - {str(e)[:100]}")
                        
                        # Overall progress güncelle
                        overall_progress.progress((file_idx + 1) / len(all_texts))
                    
                    # Temizlik
                    overall_status.empty()
                    overall_progress.empty()
                    for col in progress_cols:
                        col.empty()
                    
                    # Toplam süre
                    total_time = time.time() - start_time
                    mins, secs = divmod(int(total_time), 60)
                    
                    if all_results:
                        # Sonuçları session state'e kaydet
                        st.session_state['results'] = all_results
                        st.session_state['analyzed'] = True
                        
                        st.success(
                            f"✅ Analiz tamamlandı! "
                            f"Toplam {len(all_results)} eşleşme bulundu. "
                            f"Süre: {mins} dakika {secs} saniye. "
                            f"'Sonuçlar' sekmesine gidin."
                        )
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
                    st.plotly_chart(fig, use_container_width=True, key=f"radar_chart_{idx}")
        
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
        
        # YENİ: OVERLAP ANALİZİ
        st.subheader("🔄 Overlap (Üst Üste Binme) Analizi")
        
        st.info("💡 Bu analiz, aynı cümlelerin farklı context'lerde kaç kez analiz edildiğini gösterir.")
        
        # Her dosya için cümle kullanımlarını hesapla
        file_sentence_usage = {}
        
        for r in results:
            fname = r.get('filename', 'N/A')
            context_range = r.get('context_range', (0, 0))
            
            if fname not in file_sentence_usage:
                file_sentence_usage[fname] = []
            
            # Bu eşleşmede kullanılan tüm cümle indekslerini ekle
            for sentence_idx in range(context_range[0], context_range[1]):
                file_sentence_usage[fname].append(sentence_idx)
        
        # Genel overlap istatistikleri
        all_used_sentences = []
        for sentences in file_sentence_usage.values():
            all_used_sentences.extend(sentences)
        
        total_sentence_usages = len(all_used_sentences)
        unique_sentences_used = len(set(all_used_sentences))
        
        if total_sentence_usages > 0:
            overlap_ratio = ((total_sentence_usages - unique_sentences_used) / total_sentence_usages) * 100
            avg_usage_per_sentence = total_sentence_usages / unique_sentences_used if unique_sentences_used > 0 else 0
        else:
            overlap_ratio = 0
            avg_usage_per_sentence = 0
        
        # Overlap metrikleri
        overlap_col1, overlap_col2, overlap_col3, overlap_col4 = st.columns(4)
        
        with overlap_col1:
            st.metric(
                "Toplam Cümle Kullanımı", 
                f"{total_sentence_usages:,}",
                help="Tüm analizlerde kullanılan toplam cümle sayısı (tekrarlarla birlikte)"
            )
        
        with overlap_col2:
            st.metric(
                "Benzersiz Cümle", 
                f"{unique_sentences_used:,}",
                help="Kaç farklı cümle analiz edildi"
            )
        
        with overlap_col3:
            st.metric(
                "Overlap Oranı", 
                f"{overlap_ratio:.1f}%",
                help="Tekrar eden cümlelerin yüzdesi"
            )
        
        with overlap_col4:
            st.metric(
                "Ort. Kullanım/Cümle", 
                f"{avg_usage_per_sentence:.2f}x",
                help="Her cümle ortalama kaç kez kullanıldı"
            )
        
        # Overlap yorumu
        if overlap_ratio < 20:
            st.success("✅ Düşük overlap: Cümleler çoğunlukla bir kez kullanılmış, temiz veri!")
        elif overlap_ratio < 50:
            st.info("ℹ️ Orta düzey overlap: Bazı cümleler birden fazla kez kullanılmış, kabul edilebilir.")
        else:
            st.warning("⚠️ Yüksek overlap: Aynı cümleler çok kez kullanılmış. İstatistikler şişik olabilir.")
        
        # Dosya bazlı overlap analizi
        with st.expander("📂 Dosya Bazlı Overlap Detayları"):
            file_overlap_data = []
            
            for fname, sentence_list in file_sentence_usage.items():
                total_usages = len(sentence_list)
                unique_sentences = len(set(sentence_list))
                
                if total_usages > 0:
                    file_overlap = ((total_usages - unique_sentences) / total_usages) * 100
                    avg_usage = total_usages / unique_sentences if unique_sentences > 0 else 0
                else:
                    file_overlap = 0
                    avg_usage = 0
                
                file_overlap_data.append({
                    'Dosya': fname,
                    'Toplam Kullanım': total_usages,
                    'Benzersiz Cümle': unique_sentences,
                    'Overlap %': f"{file_overlap:.1f}%",
                    'Ort. Kullanım': f"{avg_usage:.2f}x"
                })
            
            file_overlap_df = pd.DataFrame(file_overlap_data)
            file_overlap_df = file_overlap_df.sort_values('Toplam Kullanım', ascending=False)
            st.dataframe(file_overlap_df, use_container_width=True)
        
        st.markdown("---")
        
        # Grafikler
        st.subheader("📊 Görselleştirmeler")
        
        # 1. Sentiment Dağılımı
        st.markdown("### 1️⃣ Sentiment Dağılımı (Model 1 & 2)")
        fig_sentiment = create_sentiment_distribution_chart(results)
        if fig_sentiment:
            st.plotly_chart(fig_sentiment, use_container_width=True, key="sentiment_dist_chart")
        
        st.markdown("---")
        
        # 2. Dosya Bazlı Analiz
        st.markdown("### 2️⃣ Dosya Bazlı Eşleşme Sayıları")
        fig_files = create_file_summary_chart(results)
        if fig_files:
            st.plotly_chart(fig_files, use_container_width=True, key="file_summary_chart")
        
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
            st.plotly_chart(fig_keywords, use_container_width=True, key="keyword_summary_chart")
        
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
            st.plotly_chart(fig_emotions, use_container_width=True, key="emotion_freq_chart")
        
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
    - **Overlap Stratejisi:** Tüm eşleşmeler analiz edilir (maksimum kapsam)
    
    ### 📖 Kaynaklar
    - [Guhr et al. 2020 - LREC](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.202.pdf)
    - [GoEmotions - ACL 2020](https://aclanthology.org/2020.acl-main.372/)
    - [GitHub Repository](https://github.com/laikakos/laikaresearch)
    
    ### ⚡ Performans İpuçları (2500 PDF için)
    - Dosyalar batch olarak işlenir
    - Her dosya için canlı ilerleme takibi
    - Hatalı dosyalar atlanır ve listelenir
    - Toplam süre: ~30-90 dakika (dosya boyutuna bağlı)
    - Overlap analizi ile veri kalitesi takibi
    
    ### 📊 Özellikler
    - **Canlı İlerleme:** Dosya başı, eşleşme sayısı, tahmini kalan süre
    - **Overlap Analizi:** Aynı cümlelerin kaç kez kullanıldığını gösterir
    - **İstatistikler Sekmesi:** Genel görselleştirmeler ve trendler
    - **Dosya Bazlı Analiz:** Her dosyanın detaylı sentiment dağılımı
    - **Anahtar Kelime Analizi:** Hangi kelime ne kadar etkili
    - **Model 3 Duygu Haritası:** En sık görülen 15 duygu
    """)

st.markdown("---")
st.markdown("💡 **İpucu:** Sidebar'dan anahtar kelimeleri ve context window ayarlarını özelleştirebilirsiniz.")
