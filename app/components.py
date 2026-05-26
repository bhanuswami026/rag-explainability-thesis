"""
Custom UI Components and Styling for the Streamlit Dashboard.
Implements modern CSS, Glassmorphism panels, and structured visualization components.
"""

import streamlit as st

def apply_custom_css():
    """
    Injects professional, publication-quality dark-mode theme CSS
    into the Streamlit app. Features smooth gradients, glassmorphism panels,
    and responsive card styling.
    """
    st.markdown(
        """
        <style>
        /* Base page overrides */
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: 'Inter', 'Outfit', 'Segoe UI', sans-serif;
        }
        
        /* Headers formatting */
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }
        
        h1 {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.8rem !important;
        }
        
        /* Custom containers representing thesis panels */
        .glass-card {
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        
        .glass-card:hover {
            border-color: rgba(0, 242, 254, 0.4);
            transform: translateY(-2px);
        }
        
        /* Custom metric card system */
        .metric-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            border-top: 3px solid #4facfe;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #00f2fe;
            margin-bottom: 0.2rem;
        }
        
        .metric-label {
            font-size: 0.8rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Highlight containers for text overlap */
        .highlight-container {
            background: rgba(15, 23, 42, 0.8);
            border-left: 4px solid #00f2fe;
            border-radius: 0 8px 8px 0;
            padding: 1rem;
            margin-bottom: 1rem;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        
        .source-tag {
            font-size: 0.75rem;
            background: rgba(0, 242, 254, 0.15);
            color: #00f2fe;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 0.5rem;
            border: 1px solid rgba(0, 242, 254, 0.3);
        }
        
        /* Sidebar customisations */
        .css-1cd4c57 {
            background-color: #0f172a;
        }
        
        /* Success/Progress elements */
        .stProgress > div > div > div > div {
            background-color: #00f2fe;
        }
        
        /* Custom buttons styling */
        .stButton > button {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
            color: #0b0f19 !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: scale(1.02) !important;
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.4) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_metric_dashboard(metrics: dict):
    """
    Renders RAG Triad scores in a premium grid layout.
    """
    context_rel = metrics.get("context_relevance", 0.0)
    groundedness = metrics.get("groundedness", 0.0)
    resp_rel = metrics.get("response_relevance", 0.0)
    latency = metrics.get("latency", 0.0)
    
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-card" style="border-top-color: #3b82f6;">
                <div class="metric-value">{context_rel:.2f}</div>
                <div class="metric-label">Context Relevance</div>
            </div>
            <div class="metric-card" style="border-top-color: #10b981;">
                <div class="metric-value">{groundedness:.2f}</div>
                <div class="metric-label">Groundedness</div>
            </div>
            <div class="metric-card" style="border-top-color: #8b5cf6;">
                <div class="metric-value">{resp_rel:.2f}</div>
                <div class="metric-label">Answer Relevance</div>
            </div>
            <div class="metric-card" style="border-top-color: #f59e0b;">
                <div class="metric-value">{latency:.2f}s</div>
                <div class="metric-label">Total Latency</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_glass_card(title: str, content: str, is_html: bool = False):
    """
    Utility to render a beautiful layout card.
    """
    if is_html:
        st.markdown(
            f"""
            <div class="glass-card">
                <h3 style="margin-top:0; color:#00f2fe; font-size:1.2rem; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:0.5rem; margin-bottom:1rem;">{title}</h3>
                {content}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="glass-card">
                <h3 style="margin-top:0; color:#00f2fe; font-size:1.2rem; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:0.5rem; margin-bottom:1rem;">{title}</h3>
                <p style="margin:0; font-size:0.95rem; line-height:1.6; color:#e2e8f0;">{content}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
def render_highlighted_source(source_name: str, page_num: int, similarity: float, html_content: str):
    """
    Renders retrieved context chunks with detailed source coordinates and a nice glass container.
    """
    st.markdown(
        f"""
        <div class="highlight-container">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem;">
                <span class="source-tag">{source_name} (Page {page_num})</span>
                <span style="font-size: 0.8rem; color: #a5f3fc; font-weight:600; background:rgba(6,182,212,0.1); border: 1px solid rgba(6,182,212,0.2); padding: 2px 6px; border-radius: 4px;">Cosine Sim: {similarity:.4f}</span>
            </div>
            <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.6;">
                {html_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
