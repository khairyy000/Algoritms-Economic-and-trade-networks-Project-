import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="CSAI 330 Project | Category 6", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results.csv"
PLOT_PATH = BASE_DIR / "empirical_growth_plot.png"
VIDEOS_DIR = BASE_DIR / "videos"

st.title("CSAI 330: Construction & Visualization of Densest Subgraphs")
st.subheader("Category 6: Economic & Trade Networks")

# Sidebar
st.sidebar.markdown("### Team Project Details")
st.sidebar.info("**Team Size:** 10 Students\n\n**Category:** 6 (Economic & Trade)")
nav = st.sidebar.radio("Navigation", ["Project Overview", "Algorithm Analysis & Metrics", "Incremental Visualizations"])

data_exists = RESULTS_PATH.exists()

if nav == "Project Overview":
    st.write("### 1. Problem Definition & Graph Modeling")
    st.write("""
    In economic networks, nodes represent countries or economic entities, and edges represent bilateral trade flows or value chains.
    Identifying the **densest subgraph** highlights the most tightly integrated economic clusters and trade alliances.
    """)
    
    st.write("### 2. Employed Datasets (Category 6)")
    if data_exists:
        df = pd.read_csv(RESULTS_PATH)
        st.dataframe(df[['Dataset', 'Nodes', 'Edges']], width="stretch")
    else:
        st.warning("Please run `python main.py` first to process datasets.")

elif nav == "Algorithm Analysis & Metrics":
    st.write("### Empirical Cost Growth & Solution Quality")
    
    if PLOT_PATH.exists():
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.image(str(PLOT_PATH), caption="Execution Time vs Network Size (Log Scale)")
        with col2:
            st.write("#### Performance Summary")
            st.markdown("""
            * **Greedy Peeling (Charikar & Greedy++):** Linear time complexity $O(V + E)$, scales easily to large networks.
            * **Max-Flow Based (Goldberg & Triangle):** Computes exact solutions using binary search over min-cuts, but experiences high time complexity on dense graphs.
            """)
            
        st.write("---")
        st.write("### Comparative Density Results & Execution Time")
        df = pd.read_csv(RESULTS_PATH)
        
        # Display Comparative Density
        st.write("**Achieved Density Scores ($|E| / |V|$):**")
        st.dataframe(df[['Dataset', 'Charikar_Density', 'GreedyPP_Density', 'Goldberg_Density', 'Triangle_Density']], width="stretch")
        
        # Display Execution Times
        st.write("**Execution Times (Seconds):**")
        st.dataframe(df[['Dataset', 'Charikar_Time', 'GreedyPP_Time', 'Goldberg_Time', 'Triangle_Time']], width="stretch")
    else:
        st.warning("Plot not found. Run `python main.py` first.")

elif nav == "Incremental Visualizations":
    st.write("### Subgraph Construction Demos")
    if data_exists:
        df = pd.read_csv(RESULTS_PATH)
        datasets = df['Dataset'].tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            dataset_choice = st.selectbox("Select Dataset", datasets)
        with col2:
            algo_map = {
                "Charikar's Greedy": "charikar",
                "Greedy++": "greedypp",
                "Goldberg's Exact": "goldberg",
                "Exact Triangle": "triangle"
            }
            algo_choice = st.selectbox("Select Algorithm", list(algo_map.keys()))
            
        video_path = VIDEOS_DIR / f"{dataset_choice}_{algo_map[algo_choice]}.mp4"
        gif_path = video_path.with_suffix('.gif')
        
        st.write("---")
        if video_path.exists():
            st.video(str(video_path))
        elif gif_path.exists():
            st.image(str(gif_path))
        else:
            st.error(f"Visualization missing for {dataset_choice} - {algo_choice}.")
    else:
        st.warning("Run `python main.py` to generate visualization videos.")