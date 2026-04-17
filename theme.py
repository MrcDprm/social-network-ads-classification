# theme.py
"""Streamlit uygulaması için tema yönetim modülü.
Bu modül, aydınlık (light) ve karanlık (dark) modlar için CSS değişkenlerini ('--bg-color', '--primary-color' vb.)
tanımlayarak uygulamanın renk paletini belirler ve bu değişkenleri bir stil etiketiyle arayüze enjekte eder.
"""
import streamlit as st

COMMON_CSS = """
    /* Main App Background Override */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        box-shadow: var(--shadow);
        border-right: 1px solid var(--border-color);
    }
    
    /* Make the inner sidebar content background transparent to show gradient */
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    [data-testid="stSidebarContent"] {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Target standard Streamlit input elements for rounded corners */
    .stButton > button {
        border-radius: 10px;
        transition: all 0.3s ease;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid var(--border-color);
    }
    .stButton > button:hover {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-color: var(--primary-color) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    .stSelectbox > div > div > div, 
    .stTextInput > div > div > div,
    .stNumberInput > div > div > div {
        border-radius: 10px;
        border-color: var(--border-color);
    }
    
    .stSlider > div > div > div[role="slider"] {
        border-radius: 50%;
        box-shadow: var(--shadow);
    }

    /* Dialog/Modal Styling */
    div[data-testid="stDialog"] > div {
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2) !important;
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color);
    }
    
    /* Metrics block styling */
    [data-testid="stMetricValue"] {
        color: var(--primary-color);
        font-weight: bold;
    }
"""

LIGHT_CSS = f"""
    :root {{
        --bg-color: #f9fafb;
        --text-color: #111827;
        --primary-color: #2563eb;
        --sidebar-bg: linear-gradient(180deg, #ffffff 0%, #f3f4f6 100%);
        --card-bg: #ffffff;
        --border-color: #e5e7eb;
        --shadow: 4px 0px 10px rgba(0, 0, 0, 0.05);
    }}
{{COMMON_CSS}}
"""

DARK_CSS = f"""
    :root {{
        --bg-color: #111827;
        --text-color: #f9fafb;
        --primary-color: #3b82f6;
        --sidebar-bg: linear-gradient(180deg, #1f2937 0%, #111827 100%);
        --card-bg: #1f2937;
        --border-color: #374151;
        --shadow: 4px 0px 10px rgba(0, 0, 0, 0.5);
    }}
{{COMMON_CSS}}
"""

def apply_theme(mode: str):
    """
    Seçilen temaya ('light' veya 'dark') ait CSS kurallarını Streamlit uygulamasına enjekte eder.
    Ayrıca JavaScript ile HTML sayfasına 'data-theme' özelliği (attribute) ekleyerek 
    tarayıcı tarafında ekstra CSS seçicilerinin kullanılabilmesini sağlar.
    """
    # Seçilen moda göre CSS metnini belirliyoruz.
    css = LIGHT_CSS if mode == "light" else DARK_CSS
    
    # st.markdown ile CSS'i doğrudan HTML içine <style> etiketiyle gömüyoruz.
    # unsafe_allow_html=True parametresi Streamlit'in HTML yorumlamasına izin verir.
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
    # JS kodu kullanarak belgenin (document) kök elemanına (documentElement) 'data-theme' niteliği atıyoruz.
    st.markdown(
        f"<script>document.documentElement.setAttribute('data-theme', '{mode}');</script>",
        unsafe_allow_html=True,
    )
