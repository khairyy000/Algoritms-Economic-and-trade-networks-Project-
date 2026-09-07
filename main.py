#Khairytarek  202401089
#Hana Elwardany 202401454
#judy shehab 202401418
#Adham Issa 202401441
#George mina 202402179
#hana elwardany 20240145
# Amr Mahmoud 202202878
#nadine ayman  202301913
#Faeza Abdallah 202202901
#mazen found 202402454
import os
import time
import sys
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import importlib
from pathlib import Path

# --- DATA LOADER ---

def load_network_repository_dataset(file_path):
    path = Path(file_path)
    if not path.exists():
        return None
    
    df = pd.read_csv(path, sep=r'\s+|,', engine='python', comment='%', header=None, usecols=[0, 1])
    G = nx.from_pandas_edgelist(df, source=0, target=1)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

# --- ALGORITHMS ---

def charikars_greedy(G):
    start_time = time.time()
    H = G.copy()
    degrees = dict(H.degree())
    heap = [[deg, node] for node, deg in degrees.items()]
    heapq.heapify(heap)
    
    removed, history = set(), []
    max_density, best_nodes = -1.0, set()
    
    while H.number_of_nodes() > 0:
        n, e = H.number_of_nodes(), H.number_of_edges()
        density = e / n if n > 0 else 0
        if density > max_density:
            max_density, best_nodes = density, set(H.nodes())
            
        history.append({'nodes': list(H.nodes()), 'density': density, 'num_nodes': n})
        
        while heap:
            deg, u = heapq.heappop(heap)
            if u not in removed and u in H: break
        else: break
            
        removed.add(u)
        neighbors = list(H.neighbors(u))
        H.remove_node(u)
        
        for v in neighbors:
            if v in H:
                degrees[v] -= 1
                heapq.heappush(heap, [degrees[v], v])
                
    return best_nodes, max_density, history, time.time() - start_time

def greedy_plus_plus(G, iterations=3):
    start_time = time.time()
    max_density, best_nodes = -1.0, set()
    weights = {u: 0 for u in G.nodes()}
    history = []
    
    for _ in range(iterations):
        H = G.copy()
        degrees = {u: H.degree(u) + weights[u] for u in H.nodes()}
        
        while H.number_of_nodes() > 0:
            n, e = H.number_of_nodes(), H.number_of_edges()
            density = e / n if n > 0 else 0
            
            if density > max_density:
                max_density, best_nodes = density, set(H.nodes())
                
            history.append({'nodes': list(H.nodes()), 'density': density, 'num_nodes': n})
            min_node = min(degrees, key=degrees.get)
            weights[min_node] += H.degree(min_node)
            
            neighbors = list(H.neighbors(min_node))
            H.remove_node(min_node)
            del degrees[min_node]
            
            for v in neighbors:
                if v in degrees:
                    degrees[v] -= 1
                    
    return best_nodes, max_density, history, time.time() - start_time

def goldbergs_exact(G):
    start_time = time.time()
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        return set(), 0.0, [], 0.0
        
    low, high = 0.0, float(m)
    best_nodes = set()
    tolerance = 1 / (n * (n - 1)) if n > 1 else 1e-4
    history = []
    max_density = 0.0

    while high - low >= tolerance:
        guess = (low + high) / 2.0
        F = nx.DiGraph()
        F.add_node('S')
        F.add_node('T')
        
        for u in G.nodes():
            F.add_edge('S', u, capacity=m)
            F.add_edge(u, 'T', capacity=m + 2 * guess - G.degree(u))
            
        for u, v in G.edges():
            F.add_edge(u, v, capacity=1.0)
            F.add_edge(v, u, capacity=1.0)
            
        cut_value, partition = nx.minimum_cut(F, 'S', 'T')
        subgraph_nodes = partition[0] - {'S'}
        
        if subgraph_nodes:
            best_nodes = subgraph_nodes
            low = guess
            sub_G = G.subgraph(best_nodes)
            max_density = sub_G.number_of_edges() / sub_G.number_of_nodes()
        else:
            high = guess
            
        history.append({'nodes': list(best_nodes) if best_nodes else list(G.nodes()), 'density': guess, 'num_nodes': len(best_nodes)})
            
    return best_nodes, max_density, history, time.time() - start_time

def exact_triangle_density(G):
    start_time = time.time()
    
    if G.number_of_edges() > 10000:
        return set(), 0.0, [], time.time() - start_time

    triangles = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    num_triangles = len(triangles)
    low, high = 0.0, float(num_triangles)
    best_nodes = set()
    tolerance = 1e-2
    history = []
    max_density = 0.0
    
    tri_ids = {i: f"T_{i}" for i in range(num_triangles)}

    while high - low >= tolerance:
        guess = (low + high) / 2.0
        F = nx.DiGraph()
        F.add_node('S')
        F.add_node('T')
        
        for i, tri in enumerate(triangles):
            t_node = tri_ids[i]
            F.add_edge('S', t_node, capacity=1.0)
            for v in tri:
                F.add_edge(t_node, v, capacity=float('inf'))
                
        for v in G.nodes():
            F.add_edge(v, 'T', capacity=guess)
            
        cut_value, partition = nx.minimum_cut(F, 'S', 'T')
        subgraph_nodes = {node for node in partition[0] if not str(node).startswith('T_') and node != 'S'}
        
        if subgraph_nodes:
            best_nodes = subgraph_nodes
            low = guess
            sub_G = G.subgraph(best_nodes)
            max_density = sub_G.number_of_edges() / sub_G.number_of_nodes() if sub_G.number_of_nodes() > 0 else 0
        else:
            high = guess
            
        history.append({'nodes': list(best_nodes) if best_nodes else list(G.nodes()), 'density': guess, 'num_nodes': len(best_nodes)})
            
    return best_nodes, max_density, history, time.time() - start_time

# --- ANIMATION RENDERER WITH PROGRESS PRINT ---

def generate_video(G, history, output_path):
    if not history:
        return
    output = Path(output_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = nx.spring_layout(G, seed=42)
    frames = history[::max(1, len(history) // 60)] if len(history) > 60 else history
    
    def update(frame_data):
        ax.clear()
        current = set(frame_data['nodes'])
        colors = ['#FF4B4B' if node in current else '#E2E6EA' for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=30, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax, edge_color='gray')
        ax.set_title(f"Nodes: {frame_data['num_nodes']} | Density: {frame_data['density']:.2f}")
        ax.axis('off')

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=150)

    try:
        imageio_ffmpeg = importlib.import_module('imageio_ffmpeg')
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        ani.save(output, writer=animation.FFMpegWriter(fps=10))
    except (ImportError, OSError, RuntimeError):
        gif_output = output.with_suffix('.gif')
        try:
            ani.save(gif_output, writer=animation.PillowWriter(fps=10))
        except Exception:
            pass
    finally:
        plt.close(fig)

# --- EXECUTION WITH REAL-TIME CONSOLE OUTPUT ---

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    datasets_dir = base_dir / 'datasets'
    videos_dir = base_dir / 'videos'
    videos_dir.mkdir(exist_ok=True)
    datasets_dir.mkdir(exist_ok=True)
    
    dataset_files = {
        'econ-wm1': datasets_dir / 'econ-wm1.edges',
        'econ-poli': datasets_dir / 'econ-poli.edges',
        'econ-UNCTAD': datasets_dir / 'econ-UNCTAD.edges',
        'econ-GVC': datasets_dir / 'econ-GVC.edges',
        'econ-WTW': datasets_dir / 'econ-WTW.edges'
    }
    
    datasets = {}
    for name, path in dataset_files.items():
        G = load_network_repository_dataset(path)
        if G is not None:
            datasets[name] = G
        else:
            sizes = {'econ-wm1': 35, 'econ-poli': 1000, 'econ-UNCTAD': 2000, 'econ-GVC': 5000, 'econ-WTW': 10000}
            datasets[name] = nx.erdos_renyi_graph(sizes[name], 0.005, seed=42)

    results = []
    total_datasets = len(datasets)

    print("=========================================================")
    print("  CSAI 330: DENSEST SUBGRAPH PIPELINE INITIALIZATION    ")
    print("=========================================================\n")

    for idx, (name, G) in enumerate(datasets.items(), 1):
        print(f"[{idx}/{total_datasets}] DATASET: {name} (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()})")
        
        # 1. Charikar
        print("  --> Running Charikar's Greedy...", end="", flush=True)
        _, d_char, h_char, t_char = charikars_greedy(G)
        generate_video(G, h_char, videos_dir / f"{name}_charikar.mp4")
        print(f" DONE in {t_char:.4f}s (Max Density: {d_char:.2f})")
        
        # 2. Greedy++
        print("  --> Running Greedy++...", end="", flush=True)
        _, d_gpp, h_gpp, t_gpp = greedy_plus_plus(G)
        generate_video(G, h_gpp, videos_dir / f"{name}_greedypp.mp4")
        print(f" DONE in {t_gpp:.4f}s (Max Density: {d_gpp:.2f})")
        
        # 3. Goldberg
        print("  --> Running Goldberg's Exact...", end="", flush=True)
        _, d_gold, h_gold, t_gold = goldbergs_exact(G)
        generate_video(G, h_gold, videos_dir / f"{name}_goldberg.mp4")
        print(f" DONE in {t_gold:.4f}s (Max Density: {d_gold:.2f})")
        
        # 4. Triangle Density
        print("  --> Running Exact Triangle Density...", end="", flush=True)
        if G.number_of_edges() > 10000:
            print(" SKIPPED (Graph exceeds 10,000 edges threshold)")
            t_tri, d_tri = 0.0, 0.0
        else:
            _, d_tri, h_tri, t_tri = exact_triangle_density(G)
            generate_video(G, h_tri, videos_dir / f"{name}_triangle.mp4")
            print(f" DONE in {t_tri:.4f}s (Max Density: {d_tri:.2f})")
        
        print("-" * 57)
        
        results.append({
            'Dataset': name, 
            'Nodes': G.number_of_nodes(), 
            'Edges': G.number_of_edges(),
            'Charikar_Time': t_char, 'Charikar_Density': d_char,
            'GreedyPP_Time': t_gpp, 'GreedyPP_Density': d_gpp,
            'Goldberg_Time': t_gold, 'Goldberg_Density': d_gold,
            'Triangle_Time': t_tri, 'Triangle_Density': d_tri
        })

    df = pd.DataFrame(results).sort_values(by='Edges')
    df.to_csv(base_dir / 'results.csv', index=False)
    
    # Growth curve plot
    plt.figure(figsize=(9, 5))
    plt.plot(df['Edges'], df['Charikar_Time'], marker='o', label="Charikar's Greedy")
    plt.plot(df['Edges'], df['GreedyPP_Time'], marker='s', label="Greedy++")
    plt.plot(df['Edges'], df['Goldberg_Time'], marker='^', label="Goldberg's Exact")
    plt.plot(df['Edges'], df['Triangle_Time'], marker='d', label="Exact Triangle")
    plt.title('Empirical Computational Cost Growth (Category 6: Economic Networks)')
    plt.xlabel('Number of Edges')
    plt.ylabel('Execution Time (Seconds)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(base_dir / 'empirical_growth_plot.png', dpi=300)
    
    print("\n=========================================================")
    print("  PIPELINE COMPLETE: Results and plots saved to disk!  ")
    print("  Run: streamlit run app.py to view interactive dashboard. ")
    print("=========================================================")