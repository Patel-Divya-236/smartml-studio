import streamlit as st

def apply_custom_theme() -> None:
    """Inject premium CSS styling into the current page to elevate aesthetics."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Apply Outfit font to all elements */
        html, body, [class*="css"], .stApp {
            font-family: 'Outfit', sans-serif !important;
        }

        /* Gradient header text */
        .gradient-text {
            background: linear-gradient(135deg, #FF4B4B, #9b51e0, #2d9cdb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        /* Glassmorphism card utility */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        /* Premium button styling */
        div.stButton > button {
            background: linear-gradient(135deg, #FF4B4B, #8e44ad) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3) !important;
            width: 100%;
        }
        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(255, 75, 75, 0.5) !important;
            color: #ffffff !important;
        }
        div.stButton > button:active {
            transform: translateY(0) !important;
        }

        /* Metric styling improvements */
        [data-testid="stMetricValue"] {
            font-weight: 700;
        }

        /* Custom scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
