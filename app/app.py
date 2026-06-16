import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import base64
import os
import plotly.express as px

# CONFIGURATION
st.set_page_config(page_title="Resume Screening System", page_icon="🚀", layout="wide")

# HELPER FUNCTIONS 
def get_img_as_base64(file):
    """Конвертирует локальную картинку в формат Base64 для отображения в HTML"""
    try:
        if os.path.exists(file):
            with open(file, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
        return ""
    except Exception:
        return ""

@st.cache_resource
def load_assets():
    """Загрузка моделей и данных с кэшированием"""
    try:
        knn = joblib.load('app/knn_model.pkl')
        vectorizer = joblib.load('app/vectorizer.pkl')
        lsa = joblib.load('app/lsa_transformer.pkl')
        resumes = pd.read_csv('data/processed/resumes_cleaned.csv')
        # Используем обновленный файл со ссылками
        jobs = pd.read_csv('data/processed/adzuna_cleaned_final_.csv', delimiter=';')
        return knn, vectorizer, lsa, resumes, jobs
    except Exception as e:
        st.error(f"Error loading files: {e}. Check if all .pkl and .csv files exist.")
        st.stop()

def clean_input_text(text):
    return str(text).lower()

# SESSION STATE & NAVIGATION 
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

def navigate_to(page_name):
    st.session_state.page = page_name

# Загружаем ресурсы один раз
knn, tfidf, lsa, df_resumes, df_jobs = load_assets()

# PAGE: Home
def show_landing_page():
    # иконки (Base64)
    img1 = get_img_as_base64("app/step1.png")
    img2 = get_img_as_base64("app/step2.png")
    img3 = get_img_as_base64("app/step3.png")

    # ВЕРХНЯЯ ПАНЕЛЬ И ЗАГОЛОВКИ
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        .main-container {
            font-family: 'Inter', sans-serif;
            color: #FFFFFF;
        }

        /* Главный заголовок */
        .hero-title {
            text-align: center;
            font-size: 3.8rem;
            font-weight: 800;
            letter-spacing: -2px;
            margin-bottom: 0px;
            padding-top: 50px;
            background: #FFFFFF;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Краткое описание под заголовком */
        .hero-subtitle {
            text-align: center;
            color: #666666;
            font-size: 1.1rem;
            max-width: 600px;
            margin: 10px auto 50px auto;
            line-height: 1.6;
            font-weight: 400;
        }

        /* Кастомный разделитель */
        .divider-gradient {
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0) 100%);
            margin: 40px 0;
        }

        /* Заголовок методологии */
        .methodology-header {
            text-align: center;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 4px;
            color: #FF4B4B; /* Акцентный цвет */
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .methodology-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 50px;
            color: #FFFFFF;
            letter-spacing: -1px;
        }

        /* Стили для карточек (оставляем те же, что были в предыдущем шаге) */
        .step-card {
            background-color: #111111;
            border: 1px solid #222;
            border-radius: 24px;
            padding: 40px 30px;
            height: 420px;
            transition: all 0.4s ease;
            text-align: center;
        }
        .step-card:hover {
            border-color: #444;
            transform: translateY(-10px);
        }
        .step-icon { width: 80px; height: 80px; margin-bottom: 20px; }
        .step-title { font-size: 1.4rem; font-weight: 700; color: #FFF; margin-bottom: 15px; }
        .step-text { color: #888; font-size: 0.95rem; line-height: 1.6; }
        /* Премиальная кнопка старта */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #FF6B6B 0%, #E63946 100%);
            color: white;
            border: none;
            border-radius: 50px; /* Сильное скругление */
            padding: 10px 24px;
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 1px;
            box-shadow: 0 8px 20px rgba(230, 57, 70, 0.3);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        
        div.stButton > button:first-child:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 25px rgba(230, 57, 70, 0.5);
            background: linear-gradient(135deg, #FF5252 0%, #D90429 100%);
            color: white;
            border: none;
        }
        
        div.stButton > button:first-child:active {
            transform: translateY(1px);
            box-shadow: 0 4px 10px rgba(230, 57, 70, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

    # HERO SECTION
    st.markdown("<div class='hero-title'>Resume Screening System</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='hero-subtitle'>
            An advanced analytical tool leveraging NLP and Latent Semantic Analysis to match 
            your unique professional profile against thousands of live vacancies from 
            <span style='color: #FFFFFF; font-weight: 700;'>the Adzuna network</span>.
        </div>
    """, unsafe_allow_html=True)
    
    # Визуальный разделитель
    st.markdown("<div class='divider-gradient'></div>", unsafe_allow_html=True)
    
    # METHODOLOGY SECTION
    st.markdown("<div class='methodology-header'>The Pipeline</div>", unsafe_allow_html=True)
    st.markdown("<div class='methodology-title'>System Methodology</div>", unsafe_allow_html=True)
    

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown(f"""
            <div class='step-card'>
                <div class='step-icon-container'><img src='data:image/png;base64,{img1}' class='step-icon'/></div>
                <h3 style='text-align: center;'>1. Profile Input</h3>
                <p style='text-align: center; font-size: 0.9rem;'>Tell us about your tech stack and experience. Our NLP engine will parse every detail.</p>
                <div style='text-align: center; margin-top: 15px; color: #FF4B4B; font-weight: 700; font-size: 1.1rem;'>
                    Python • ML
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown(f"""
            <div class='step-card'>
                <div class='step-icon-container'><img src='data:image/png;base64,{img2}' class='step-icon'/></div>
                <h3 style='text-align: center;'>2. Classification</h3>
                <p style='text-align: center; font-size: 0.9rem;'>Our KNN model compares you with 9,500+ professionals to predict your role.</p>
                <div class='metric-val' style='margin-top: 20px; color: #00E676;'>83.7%</div>
                <div class='metric-label'>Model Accuracy</div>
            </div>
        """, unsafe_allow_html=True)

    with col_s3:
        st.markdown(f"""
            <div class='step-card'>
                <div class='step-icon-container'><img src='data:image/png;base64,{img3}' class='step-icon'/></div>
                <h3 style='text-align: center;'>3. Job Matching</h3>
                <p style='text-align: center; font-size: 0.9rem;'>Using LSA, we find the best semantic matches from 10,000+ live Adzuna vacancies.</p>
                <div class='metric-val' style='margin-top: 20px; color: #00c6ff;'>10,494</div>
                <div class='metric-label'>Jobs in Database</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.button("START SEARCHING FOR VACANCIES", use_container_width=True, type="primary", on_click=navigate_to, args=('app',))

    st.write("---")
    
    t1, t2, t3, t4 = st.columns(4)
    # TEAM SECTION
    st.markdown("<h3 style='text-align: center; font-weight: 700;'>Our Team</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>SDU University | School of Information Technologies and Applied Mathematics</p>", unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    
    t1, t2, t3, t4 = st.columns(4)
    
    team = [
        {"name": "Yerassyl Amangeldi", "role": "UI Developer", "image": "app/yerassyl.jpg", "col": t1},
        {"name": "Akerke Muratbekova", "role": "Data Scientist", "image": "app/akerke.jpg", "col": t2},
        {"name": "Magzhan Slyamgaziyev", "role": "ML Engineer", "image": "app/magzhan.jpg", "col": t3},
        {"name": "Diana Kozhabashayeva", "role": "Data Analyst", "image": "app/diana.jpg", "col": t4}
    ]
    
    for member in team:
        with member["col"]:
            try:
                st.image(member["image"], use_container_width=True) 
            except Exception:
                # Если файл не найден то выводим серый квадратик
                st.markdown("<div style='height: 140px; background-color: #222; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 2rem;'>👤</div>", unsafe_allow_html=True)
                
            st.markdown(f"<div style='font-weight: 700; margin-top: 10px;'>{member['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color: #666; font-size: 0.85rem;'>{member['role']}</div>", unsafe_allow_html=True)
# PAGE: MAIN APP 
def show_main_app():
    st.button("Back to Home", on_click=navigate_to, args=('landing',))
    st.title("Find Your Career Match")
    
    col_in, col_stats = st.columns([1.2, 1])

    with col_in:
        st.subheader("Professional Profile")
        pos = st.text_input("Desired Position:", placeholder="e.g., Backend Developer")
        skills = st.text_area("Key Skills:", placeholder="Java, Spring Boot, MySQL, AWS...")
        exp = st.text_area("Experience:", placeholder="Worked 2 years in FinTech...")
        projects = st.text_area("Projects:", placeholder="Built a microservices-based API...")
        salary = st.text_input("Expected Salary (Optional):", placeholder="e.g., $120,000")

        full_resume_text = f"{pos} {skills} {exp} {projects}"

        if st.button("Analyze & Match", type="primary"):
            # 1. Жесткая проверка: если поле Position пустое — выдаем ошибку и останавливаемся
            if not pos.strip():
                st.warning("The 'Desired Position' field is mandatory. Please fill it in.")
                
            # 2. Мягкая проверка (опционально): просим ввести еще и навыки для точности
            elif not skills.strip():
                st.warning("Please enter your Key Skills as well for an accurate match.")
                
            # Если оба поля заполнены, запускаем ML-модель
            else:
                clean_txt = clean_input_text(full_resume_text)
                res_tfidf = tfidf.transform([clean_txt])
                
                # ... дальше идет код с проверкой на nnz == 0 и KNN предсказанием ...
                
                # --- НОВАЯ ПРОВЕРКА НА БРЕД И НЕЗНАКОМЫЕ СЛОВА ---
                if res_tfidf.nnz == 0:
                    st.error("We couldn't recognize any professional keywords. Please enter meaningful English terms related to your skills or position.")
                else:
                    # KNN Prediction
                    pred_role = knn.predict(res_tfidf)[0]
                    
                    st.success("**Predicted Specialist Categories:**")
                    
                    clean_role = str(pred_role).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                    roles_list = [r.strip().upper() for r in clean_role.split(',')]
                    formatted_roles = ", ".join(roles_list)
                    
                    st.success(formatted_roles)
                
                # LSA Matching
                with st.spinner("Scanning Adzuna database..."):
                    res_lsa = lsa.transform(res_tfidf)
                    #
                    jobs_tfidf = tfidf.transform(df_jobs['job_text_clean'].astype(str))
                    jobs_lsa = lsa.transform(jobs_tfidf)
                    
                    sims = cosine_similarity(res_lsa, jobs_lsa).flatten()
                    top_idx = sims.argsort()[-5:][::-1]
                    
                    st.subheader("Your Top-5 Career Matches")
                    st.markdown("---")

                    for i, idx in enumerate(top_idx):
                        j = df_jobs.iloc[idx]
                        score = sims[idx]
                        
                        with st.container():
                            c1, c2 = st.columns([3, 1])
                            
                            with c1:
                                st.markdown(f"### {i+1}. {j['job_title']}")
                                st.markdown(f"**{j['company_name']}**")
                                st.caption(f"📍 {j.get('location', 'Remote / Not specified')}")
                            
                            with c2:
                                st.metric("Match Score", f"{score*100:.1f}%")
                            
                            st.write(f"**Role Insight:** {str(j['job_text_clean'])[:180]}...")
                            
                            with st.expander("Read More & Salary Details"):
                                st.write(j['job_text_clean'])
                                
                                s_min = j.get('salary_min')
                                s_max = j.get('salary_max')
                                if pd.notna(s_min) and s_min > 0:
                                    st.write(f"**Estimated Salary:** ${s_min:,.0f} - ${s_max:,.0f}")
                                
                                url = j.get('redirect_url')
                                if pd.notna(url) and str(url).startswith('http'):
                                    st.link_button("Apply on Website", str(url), use_container_width=True)
                            
                            st.divider()

    with col_stats:
        st.subheader("Global Analytics")
        st.write(f"Processed CVs: **{len(df_resumes):,}**")
        st.write(f"Live Jobs: **{len(df_jobs):,}**")
        st.markdown("---")

        # Оставляем только этот график (он будет висеть всегда)
        st.write("**Top 10 Professional Roles**")
        st.bar_chart(df_resumes['positions'].value_counts().head(10))
        
        st.info("The chart displays the most common job categories in our dataset.")

    # --- ДИСКЛЕЙМЕР ВНИЗУ СТРАНИЦЫ ---
    st.markdown("---") 
    
    st.warning(
        "**Disclaimer:** This powered tool provides job recommendations based on semantic text matching. "
        "It is designed to assist your job search but **does not guarantee employment or interview invitations**. "
        "The final hiring decision always depends on your actual skills, portfolio, and performance during the formal interview with the employer."
    )

# ROUTER 
if st.session_state.page == 'landing':
    show_landing_page()
else:
    show_main_app()
    