"""
Streamlit Main Dashboard Application.
Serves as the M.Tech Thesis Interactive Workbench.
Implements the multi-tab evaluation interface for RAG explainability and post-hoc interpretability.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Add parent directory to path to ensure proper module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RAGPipeline
from explainability.similarity import SimilarityExplainer
from explainability.occlusion import ChunkOcclusionExplainer
from explainability.saliency import SaliencyExplainer
from evaluation.metrics import RAGEvaluator
from app.components import (
    apply_custom_css,
    render_metric_dashboard,
    render_glass_card,
    render_highlighted_source
)

# Page Configuration
st.set_page_config(
    page_title="RAG Explainability & Interpretability Workbench",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium styling
apply_custom_css()

# Session State Initialization
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False
if "indexed_chunks_count" not in st.session_state:
    st.session_state.indexed_chunks_count = 0
if "last_query_results" not in st.session_state:
    st.session_state.last_query_results = None
if "api_configured" not in st.session_state:
    st.session_state.api_configured = False

# Sidebar Configuration
with st.sidebar:
    st.markdown('<h2 style="color:#00f2fe; margin-top:0;">🎓 Thesis Panel</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        **Project**: Explainability of RAG Systems using post-hoc interpretability methods.
        
        **Researcher**: Divya Sharma
        
        **Degree**: Master of Technology (M.Tech)
        """
    )
    
    st.markdown("---")
    st.markdown("### 1. API Configuration")
    
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key (Optional)",
        value=env_key if env_key else st.session_state.get("gemini_key", ""),
        type="password",
        help="Obtain an API key from Google AI Studio. Required for Gemini models."
    )
    
    env_openai_key = os.getenv("OPENAI_API_KEY", "")
    openai_key_input = st.text_input(
        "OpenAI API Key (Optional)",
        value=env_openai_key if env_openai_key else st.session_state.get("openai_key", ""),
        type="password",
        help="Obtain an API key from OpenAI. Required if using GPT models."
    )
    
    if api_key_input:
        st.session_state.gemini_key = api_key_input
        st.session_state.api_configured = True
    else:
        st.session_state.api_configured = False
        
    if openai_key_input:
        st.session_state.openai_key = openai_key_input
        st.session_state.openai_configured = True
    else:
        st.session_state.openai_configured = False
        
    model_name_input = st.selectbox(
        "LLM Provider & Model",
        options=[
            "gemini-1.5-flash", 
            "gemini-2.0-flash", 
            "gemini-1.5-flash-latest", 
            "gemini-1.5-pro", 
            "gpt-4o-mini", 
            "gpt-4o"
        ],
        index=0,
        help="Select the model. All Gemini models are 100% free under the Google AI Studio free tier!"
    )
    
    is_openai_model = model_name_input.startswith("gpt-")
    if is_openai_model and not st.session_state.get("openai_configured", False):
        st.warning("Please configure your OpenAI API Key to enable GPT generation.")
    elif not is_openai_model and not st.session_state.get("api_configured", False):
        st.warning("Please configure your Gemini API Key to enable Gemini generation.")
        
    st.markdown("### 2. Pipeline Controls")
    chunk_size = st.slider("Chunk Size (Chars)", min_value=200, max_value=1200, value=600, step=100)
    chunk_overlap = st.slider("Chunk Overlap (Chars)", min_value=50, max_value=400, value=150, step=50)
    
    # Lazy initialization of RAG Pipeline to speed up load time
    if st.session_state.pipeline is None or \
       st.session_state.pipeline.chunker.chunk_size != chunk_size or \
       st.session_state.pipeline.chunker.chunk_overlap != chunk_overlap:
        with st.spinner("Initializing Local BGE Embedding Model (First time might take a minute)..."):
            st.session_state.pipeline = RAGPipeline(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                model_name=model_name_input
            )
    else:
        st.session_state.pipeline.model_name = model_name_input
            
    if st.session_state.api_configured and not is_openai_model:
        st.session_state.pipeline.configure_gemini(st.session_state.gemini_key)
        
    if st.session_state.get("openai_configured", False) and is_openai_model:
        st.session_state.pipeline.configure_openai(st.session_state.openai_key)
        
    st.markdown("### 3. Document Ingestion")
    uploaded_file = st.file_uploader("Upload reference PDF thesis document", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Index Document"):
            file_bytes = uploaded_file.read()
            with st.spinner("Parsing text, extracting pages, generating vector embeddings..."):
                st.session_state.pipeline.vector_store.clear()
                chunks_count = st.session_state.pipeline.ingest_document_bytes(
                    file_bytes, uploaded_file.name
                )
                st.session_state.indexed_chunks_count = chunks_count
                st.session_state.index_loaded = True
                st.success(f"Success! Indexed {chunks_count} overlapping segments using FAISS local store.")

# Main Application Title
st.markdown("<h1>Explainability & Interpretability Workbench for RAG Systems</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94a3b8; font-size:1.1rem; margin-top:0;'>Evaluating the causal necessity, semantic attribution, "
    "and word-level saliency mappings of black-box LLMs in Retrieval-Augmented Generation.</p>",
    unsafe_allow_html=True
)

if not st.session_state.index_loaded:
    # Landing page helper instructions
    st.info("👈 Please configure your **Gemini API Key** and **Upload a PDF document** in the sidebar to begin.")
    
    # Introduce the academic concepts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#00f2fe; margin:0 0 10px 0;'>🔍 Vector Similarity vs Retrieval Attention</h3>", unsafe_allow_html=True)
        st.markdown(
            "Vector databases perform search by computing cosine distance in a high-dimensional dense space "
            "(`BGE-small-en-v1.5` embeddings). We expose these raw scores and project them into an "
            "**attention probability distribution** (using Softmax normalized weights) to represent which sources "
            "the pipeline prioritizes before context assembly."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#10b981; margin:0 0 10px 0;'>🧪 Post-Hoc Causal Chunk Occlusion</h3>", unsafe_allow_html=True)
        st.markdown(
            "Since LLMs are black boxes, we cannot inspect their attention weights directly. "
            "Instead, we use a causal perturbation method: **Chunk Occlusion**. By systematically withholding "
            "each retrieved passage, re-triggering inference, and measuring response semantic drift, we isolate "
            "each source chunk's exact impact on final synthesis."
        )
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Document is processed, show Query workspace
    st.markdown("### 🔍 Query Document Database")
    
    query_col1, query_col2 = st.columns([4, 1])
    with query_col1:
        query_input = st.text_input(
            "Enter your research query:",
            placeholder="e.g. What is the primary methodology proposed in this paper?"
        )
    with query_col2:
        k_value = st.slider("Context segments (K)", min_value=1, max_value=5, value=3)
        
    if st.button("Execute RAG Pipeline"):
        if not query_input.strip():
            st.error("Please enter a valid query.")
        else:
            with st.spinner("Retrieving contexts and generating answer via Gemini..."):
                # Run main RAG operation
                rag_results = st.session_state.pipeline.query(query_input, k=k_value)
                
                # Check for errors in generated response
                if "Error" in rag_results["answer"] and not st.session_state.api_configured:
                    st.error("Generation failed. Please verify your Gemini API key in the sidebar.")
                else:
                    st.session_state.last_query_results = rag_results
                    st.toast("RAG Pipeline Executed successfully!")
                    
    # Render results if they exist
    if st.session_state.last_query_results is not None:
        results = st.session_state.last_query_results
        
        # Instantiate explainers
        sim_explainer = SimilarityExplainer()
        occlusion_explainer = ChunkOcclusionExplainer(st.session_state.pipeline.embedder)
        saliency_explainer = SaliencyExplainer()
        evaluator = RAGEvaluator(st.session_state.pipeline.embedder)
        
        # Precompute explainability and evaluation metrics
        chunks = results["retrieved_chunks"]
        scores = results["similarity_scores"]
        query = results["query"]
        answer = results["answer"]
        latencies = results["latency"]
        prompt = results["prompt"]
        
        # 1. Similarity mapping details
        retrieval_explanations = sim_explainer.explain_retrieval(chunks, scores)
        
        # 2. RAG Triad Evaluator
        context_rel_score = evaluator.evaluate_context_relevance(scores)
        groundedness_score, sentence_evals = evaluator.evaluate_groundedness(answer, chunks)
        response_rel_score = evaluator.evaluate_response_relevance(query, answer)
        
        # Display high level metrics
        metrics_dict = {
            "context_relevance": context_rel_score,
            "groundedness": groundedness_score,
            "response_relevance": response_rel_score,
            "latency": latencies["total"]
        }
        render_metric_dashboard(metrics_dict)
        
        # Set up Tabs
        tab_output, tab_explain, tab_eval = st.tabs([
            "📋 RAG Synthesis & Sources", 
            "🔬 Post-Hoc Explainability Workbench", 
            "🎓 Academic Evaluation & Verifiers"
        ])
        
        # ==========================================
        # TAB 1: RAG Output & Basic Sources
        # ==========================================
        with tab_output:
            col_ans, col_meta = st.columns([3, 2])
            with col_ans:
                st.markdown("### Generated Grounded Response")
                st.info(answer)
                
                # Show full prompt inside expander to provide input transparency
                with st.expander("📝 View Context-Augmented Prompt Sent to Gemini"):
                    st.code(prompt, language="markdown")
                    
            with col_meta:
                st.markdown("### Retrieved Source Passages (Context)")
                for idx, chunk_expl in enumerate(retrieval_explanations):
                    # Show standard highlights (simple token matching)
                    html_highlighted = saliency_explainer.generate_html_highlights(answer, chunk_expl["text"])
                    render_highlighted_source(
                        source_name=chunk_expl["source"],
                        page_num=chunk_expl["page"],
                        similarity=chunk_expl["similarity_score"],
                        html_content=html_highlighted
                    )

        # ==========================================
        # TAB 2: Explainability Workbench
        # ==========================================
        with tab_explain:
            st.markdown("### 🔬 Post-Hoc Interpretability & Explanation Engine")
            st.markdown(
                "These sections provide post-hoc visualizations explaining why chunks were retrieved "
                "and how they causally influenced the generation."
            )
            
            subtab_occl, subtab_ret, subtab_sal = st.tabs([
                "🧪 Causal Chunk Occlusion (Counterfactuals)",
                "📊 Semantic Similarity Share & Attention",
                "🎨 Sentence Word-level Saliency Highlight"
            ])
            
            # --- SUBTAB A: OCCLUSION ---
            with subtab_occl:
                st.markdown("#### 🧪 Counterfactual Perturbation (Occlusion Analysis)")
                st.markdown(
                    "This method removes one retrieved chunk at a time, re-runs answer generation, "
                    "and measures semantic distance. **High Semantic Shift** indicates that the omitted chunk "
                    "was critically necessary for the generated response."
                )
                
                with st.spinner("Executing counterfactual occlusion re-generations (calling Gemini K times)..."):
                    # Check and fetch occlusion data
                    occlusion_data = occlusion_explainer.explain_generation(
                        st.session_state.pipeline, query, chunks, answer
                    )
                
                # Render Bar Chart
                df_occl = pd.DataFrame(occlusion_data)
                
                # Create short label names for X axis
                df_occl["Label"] = df_occl.apply(
                    lambda r: f"{r['chunk_id'].split('_')[-1]} (Page {r['page']})", axis=1
                )
                
                fig_occl = px.bar(
                    df_occl,
                    x="causal_importance",
                    y="Label",
                    orientation="h",
                    title="Causal Necessity Score per Context Chunk",
                    labels={"causal_importance": "Causal Necessity Share", "Label": "Chunk Identifier"},
                    color="semantic_shift",
                    color_continuous_scale="Viridis",
                    template="plotly_dark"
                )
                fig_occl.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_occl, use_container_width=True)
                
                # Display side-by-side responses inside expander
                st.markdown("#### 🔄 Compare Counterfactual Generations")
                for item in occlusion_data:
                    with st.expander(f"🔍 Response WITHOUT Chunk {item['chunk_id'].split('_')[-1]} (Causal necessity: {item['causal_importance']:.1%})"):
                        col_orig, col_pert = st.columns(2)
                        with col_orig:
                            st.markdown("**Original Response (All Context):**")
                            st.caption(answer)
                        with col_pert:
                            st.markdown(f"**Perturbed Response (Omitting this chunk):**")
                            st.caption(item["perturbed_response"])
                        
                        st.markdown(
                            f"- **Lexical (Jaccard) Shift**: `{item['lexical_shift']:.4f}` | "
                            f"**Semantic (Cosine) Shift**: `{item['semantic_shift']:.4f}`"
                        )

            # --- SUBTAB B: SIMILARITY ---
            with subtab_ret:
                st.markdown("#### 📊 Vector Similarity vs Retrieval Attention Map")
                st.markdown(
                    "Here we compare the raw **Cosine Similarity** of the retrieved chunks "
                    "with their **Softmax Retrieval Attention Weights** (which model the prioritized allocation of context)."
                )
                
                df_sim = pd.DataFrame(retrieval_explanations)
                df_sim["Label"] = df_sim.apply(
                    lambda r: f"Chunk {r['chunk_id'].split('_')[-1]} (Page {r['page']})", axis=1
                )
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig_sim1 = px.bar(
                        df_sim,
                        x="Label",
                        y="similarity_score",
                        title="Raw Embedding Cosine Similarity",
                        labels={"similarity_score": "Cosine Similarity", "Label": "Context Chunk"},
                        template="plotly_dark"
                    )
                    fig_sim1.update_traces(marker_color='#3b82f6')
                    fig_sim1.update_layout(height=300)
                    st.plotly_chart(fig_sim1, use_container_width=True)
                    
                with col_chart2:
                    fig_sim2 = px.bar(
                        df_sim,
                        x="Label",
                        y="retrieval_attention_weight",
                        title="Softmax Retrieval Attention Allocation (τ = 0.05)",
                        labels={"retrieval_attention_weight": "Attention Weight", "Label": "Context Chunk"},
                        template="plotly_dark"
                    )
                    fig_sim2.update_traces(marker_color='#8b5cf6')
                    fig_sim2.update_layout(height=300)
                    st.plotly_chart(fig_sim2, use_container_width=True)
                    
                st.markdown(
                    "**Mathematical Note**: Softmax normalization transforms distance vectors into a "
                    "probability distribution $\\text{Softmax}(s_i / \\tau)$. A lower temperature $(\\tau = 0.05)$ "
                    "sharpens similarity margins, highlighting the system's focus on the absolute highest matching chunk."
                )

            # --- SUBTAB C: WORD-LEVEL SALIENCY HIGHLIGHTS ---
            with subtab_sal:
                st.markdown("#### 🎨 Response Lexical Saliency Highlights")
                st.markdown(
                    "This visualizer highlights which exact words inside the retrieved context chunks "
                    "were directly utilized (copied or close-matched) in the synthesized response."
                )
                
                for idx, chunk_expl in enumerate(retrieval_explanations):
                    html_highlighted = saliency_explainer.generate_html_highlights(answer, chunk_expl["text"])
                    st.markdown(f"**Chunk {chunk_expl['chunk_id'].split('_')[-1]} Saliency Alignment:**")
                    st.markdown(
                        f"<div style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); padding:1rem; border-radius:8px; line-height:1.6; margin-bottom:1rem;'>"
                        f"{html_highlighted}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        # ==========================================
        # TAB 3: Academic Evaluation
        # ==========================================
        with tab_eval:
            st.markdown("### 🎓 Academic Verification & Performance Analytics")
            st.markdown(
                "Verify the reliability of the RAG system using quantitative indicators "
                "suitable for peer-reviewed research."
            )
            
            eval_col1, eval_col2 = st.columns(2)
            
            with eval_col1:
                st.markdown("#### 🔍 Groundedness (Faithfulness) Analysis")
                st.markdown(
                    "Each sentence in the generated answer is embedded and mapped against "
                    "the source paragraphs. Low coverage scores alert you to potential **hallucinations**."
                )
                
                df_ground = pd.DataFrame(sentence_evals)
                if not df_ground.empty:
                    # Style based on overlap score
                    def color_score(val):
                        if val >= 0.8: return 'color: #10b981;' # solid grounded
                        elif val >= 0.5: return 'color: #f59e0b;' # loose deduction
                        else: return 'color: #ef4444;' # hallucination risk
                        
                    # Display clean table
                    for item in sentence_evals:
                        score = item["max_context_overlap"]
                        status = "✅ Grounded" if score >= 0.8 else "⚠️ Loose Deduction" if score >= 0.5 else "❌ Hallucination Risk"
                        
                        st.markdown(
                            f"<div style='border:1px solid rgba(255,255,255,0.05); padding:8px 12px; border-radius:6px; margin-bottom:6px; background:rgba(255,255,255,0.01);'>"
                            f"<strong>Sentence:</strong> <em>\"{item['sentence']}\"</em><br>"
                            f"<strong>Context Coverage Score:</strong> <span style='font-weight:bold; font-size:1.05rem;'>{score:.4f}</span> | "
                            f"<strong>Status:</strong> {status}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No sentences evaluated.")
                    
            with eval_col2:
                st.markdown("#### ⏱️ Latency Profiling")
                st.markdown(
                    "Visualizes execution time bottleneck. Retrieval maps BGE search on CPU "
                    "whereas Generation captures remote Gemini token synthesis latency."
                )
                
                lat_splits = evaluator.get_latency_profile(latencies)
                
                # Pie Chart for Latency
                fig_lat = go.Figure(data=[go.Pie(
                    labels=['Vector Retrieval (BGE + FAISS)', 'LLM Synthesis (Gemini API)'],
                    values=[lat_splits["retrieval_seconds"], lat_splits["generation_seconds"]],
                    hole=.4,
                    marker=dict(colors=['#3b82f6', '#10b981'])
                )])
                
                fig_lat.update_layout(
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=250,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_lat, use_container_width=True)
                
                st.markdown(
                    f"""
                    - **Vector Search Latency**: `{lat_splits['retrieval_seconds']:.4f}` seconds ({lat_splits['retrieval_percentage']:.1f}%)
                    - **LLM Generation Latency**: `{lat_splits['generation_seconds']:.4f}` seconds ({lat_splits['generation_percentage']:.1f}%)
                    - **Total Latency**: `{lat_splits['total_seconds']:.2f}` seconds
                    """
                )
