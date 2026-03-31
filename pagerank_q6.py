"""
SC4052 Cloud Computing – Assignment 2, Question 6
PageRank: Closed-Form Derivation, Implementation, Validation, and AI Crawler Extension
Student implementation for the web-Google_10k.txt dataset
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path
import time, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0.  STYLING
# ─────────────────────────────────────────────
BLUE   = "#1f4e79"
ORANGE = "#c55a11"
GREEN  = "#375623"
RED    = "#c00000"
GREY   = "#7f7f7f"
LIGHT  = "#dce6f1"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})


# ─────────────────────────────────────────────
# 1.  LOAD DATASET
# ─────────────────────────────────────────────
def build_sparse_stochastic_from_file(path):
    """
    Build the column-stochastic hyperlink matrix S (scipy CSC) directly from file.
    Uses a two-pass construction to remain memory efficient on large edge lists.

    Dangling columns are handled during iteration, not embedded in S.
    Returns S (n×n CSC), node_to_idx dict, idx_to_node list, non-self edge count.
    """
    path = Path(path)
    nodes = set()
    out_deg = defaultdict(int)
    m_nonself = 0

    # First pass: discover node set, out-degree, and valid edge count.
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            u, v = map(int, line.split())
            nodes.update([u, v])
            if u != v:
                out_deg[u] += 1
                m_nonself += 1

    nodes = sorted(nodes)
    n2i = {v: i for i, v in enumerate(nodes)}

    # Second pass: fill sparse triplets with pre-allocated arrays.
    rows = np.empty(m_nonself, dtype=np.int32)
    cols = np.empty(m_nonself, dtype=np.int32)
    data = np.empty(m_nonself, dtype=np.float64)

    k = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            u, v = map(int, line.split())
            if u == v:
                continue
            rows[k] = n2i[v]               # column j points TO row i
            cols[k] = n2i[u]
            data[k] = 1.0 / out_deg[u]
            k += 1

    n = len(nodes)
    S = sp.csc_matrix((data, (rows, cols)), shape=(n, n))
    return S, n2i, nodes, m_nonself


# ─────────────────────────────────────────────
# 2.  CLOSED-FORM SOLVER  r = (I-(1-p)S)^{-1} pv
# ─────────────────────────────────────────────
def pagerank_closed_form(S, p=0.15, n_small=None):
    """
    Exact closed-form solution.
    Only feasible for small n (dense inversion).
    """
    n = S.shape[0]
    if n_small is None:
        n_small = n
    # Use only the first n_small nodes for tractability
    S_d = S[:n_small, :n_small].toarray()
    v   = np.ones(n_small) / n_small
    A   = np.eye(n_small) - (1 - p) * S_d
    rhs = p * v
    r   = np.linalg.solve(A, rhs)
    r  /= r.sum()
    return r


# ─────────────────────────────────────────────
# 3.  POWER-ITERATION SOLVER
# ─────────────────────────────────────────────
def pagerank_power(S, p=0.15, tol=1e-10, max_iter=200):
    """
    Standard power-iteration (works on full sparse matrix).
    r_{t+1} = (1-p)*S*r_t + p*v  (dangling correction included via uniform tele)
    """
    n   = S.shape[0]
    v   = np.ones(n) / n
    r   = v.copy()
    history = []
    t0  = time.time()
    for it in range(max_iter):
        Sr     = S.dot(r)
        # Handle dangling nodes: add missing mass uniformly
        dangle = 1.0 - Sr.sum()
        r_new  = (1 - p) * Sr + (p + (1 - p) * dangle) * v
        delta  = np.abs(r_new - r).sum()
        history.append(delta)
        r = r_new
        if delta < tol:
            break
    elapsed = time.time() - t0
    return r, history, it + 1, elapsed


# ─────────────────────────────────────────────
# 4.  JACOBI-STYLE ITERATIVE SOLVER
# ─────────────────────────────────────────────
def pagerank_jacobi(S, p=0.15, tol=1e-10, max_iter=200):
    """
    Solve (I-(1-p)S)r = p*v iteratively using the splitting
    r^{(k+1)} = (1-p)*S*r^{(k)} + p*v  (same as power but kept separate for clarity).
    """
    n   = S.shape[0]
    v   = np.ones(n) / n
    r   = v.copy()
    history = []
    for it in range(max_iter):
        r_new = (1 - p) * S.dot(r) + p * v
        r_new /= r_new.sum()
        delta  = np.abs(r_new - r).sum()
        history.append(delta)
        r = r_new
        if delta < tol:
            break
    return r, history, it + 1


# ─────────────────────────────────────────────
# 5.  SMALL TOY GRAPH (for closed-form comparison)
# ─────────────────────────────────────────────
def toy_graph():
    """5-node toy graph for illustration."""
    n = 5
    # Adjacency: 0->1, 0->2, 1->2, 2->0, 3->2, 4->3
    edges = [(0,1),(0,2),(1,2),(2,0),(3,2),(4,3)]
    S = np.zeros((n, n))
    out = defaultdict(list)
    for u, v in edges:
        out[u].append(v)
    for u in range(n):
        if out[u]:
            for v in out[u]:
                S[v, u] = 1.0 / len(out[u])
        else:
            S[:, u] = 1.0 / n   # dangling
    return S, n


# ─────────────────────────────────────────────
# FIGURE 1: p-sensitivity on the toy graph
# ─────────────────────────────────────────────
def fig1_p_sensitivity(out_dir):
    S, n = toy_graph()
    p_vals = np.linspace(0.01, 0.99, 60)
    scores = np.zeros((n, len(p_vals)))

    for j, p in enumerate(p_vals):
        A   = np.eye(n) - (1 - p) * S
        v   = np.ones(n) / n
        r   = np.linalg.solve(A, p * v)
        r  /= r.sum()
        scores[:, j] = r

    colours = [BLUE, ORANGE, GREEN, RED, GREY]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    for i in range(n):
        ax.plot(p_vals, scores[i], color=colours[i], lw=2, label=f"Node {i}")
    ax.axvline(0.15, color='black', ls='--', lw=1, alpha=0.6, label="p = 0.15 (typical)")
    ax.set_xlabel("Teleportation Probability p")
    ax.set_ylabel("PageRank Score")
    ax.set_title("Q1 & Q2: How PageRank Changes with p\n(5-node toy graph)", fontweight='bold')
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.fill_between(p_vals, scores.max(0) - scores.min(0), alpha=0.3, color=BLUE, label="Score range (max−min)")
    ax2.plot(p_vals, scores.max(0) - scores.min(0), color=BLUE, lw=2)
    ax2.axvline(0.15, color='black', ls='--', lw=1, alpha=0.6)
    ax2.set_xlabel("Teleportation Probability p")
    ax2.set_ylabel("Score Spread (max − min)")
    ax2.set_title("Ranking Inequality vs p\n(large spread → link-dominated)", fontweight='bold')
    ax2.legend(fontsize=8)

    # annotation boxes
    for ax, txt, x, y in [
        (axes[0], "Link-dominated\n(unequal scores)", 0.03, 0.38),
        (axes[0], "Uniform\n(equal scores)", 0.75, 0.215),
    ]:
        ax.annotate(txt, xy=(x, y), fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=BLUE, lw=0.8))

    fig.tight_layout()
    path = f"{out_dir}/fig1_p_sensitivity.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ─────────────────────────────────────────────
# FIGURE 2: Closed-form vs Jacobi on toy graph
# ─────────────────────────────────────────────
def fig2_method_comparison(out_dir):
    S_dense, n = toy_graph()
    p = 0.15
    v = np.ones(n) / n

    # Closed form
    A  = np.eye(n) - (1 - p) * S_dense
    r_cf = np.linalg.solve(A, p * v)
    r_cf /= r_cf.sum()

    # Jacobi
    S_sp = sp.csc_matrix(S_dense)
    r_j, hist, iters = pagerank_jacobi(S_sp, p=p)

    l1 = np.abs(r_cf - r_j).sum()
    l2 = np.linalg.norm(r_cf - r_j)
    linf = np.abs(r_cf - r_j).max()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # A: side-by-side bars
    ax = axes[0]
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w/2, r_cf, w, color=BLUE,   label="Closed-form", alpha=0.85)
    ax.bar(x + w/2, r_j,  w, color=ORANGE, label="Jacobi",      alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"Node {i}" for i in x])
    ax.set_ylabel("PageRank Score"); ax.set_title("Q3 & Q4: Method Comparison\n(bars overlap perfectly)", fontweight='bold')
    ax.legend(fontsize=8)

    # B: convergence
    ax = axes[1]
    ax.semilogy(hist, color=BLUE, lw=2)
    ax.axhline(1e-10, color=RED, ls='--', lw=1, label="Tolerance 1e-10")
    ax.set_xlabel("Iteration"); ax.set_ylabel("L1 Residual |r_new − r|")
    ax.set_title(f"Convergence of Jacobi Method\n(converged in {iters} iters)", fontweight='bold')
    ax.legend(fontsize=8)

    # C: error table text
    ax = axes[2]
    ax.axis('off')
    table_data = [
        ["Metric", "Value"],
        ["L1 error",  f"{l1:.3e}"],
        ["L2 error",  f"{l2:.3e}"],
        ["L∞ error",  f"{linf:.3e}"],
        ["Iterations", str(iters)],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    for (r2, c), cell in tbl.get_celld().items():
        if r2 == 0:
            cell.set_facecolor(BLUE); cell.set_text_props(color='white', fontweight='bold')
        elif r2 % 2 == 0:
            cell.set_facecolor(LIGHT)
    ax.set_title("Error Metrics\n(machine-precision agreement)", fontweight='bold')

    fig.tight_layout()
    path = f"{out_dir}/fig2_method_comparison.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return path, l1, l2, linf, iters


# ─────────────────────────────────────────────
# FIGURE 3: Large-graph (10k) power iteration
# ─────────────────────────────────────────────
def fig3_large_graph(S, nodes, r_power, history, n_iters, elapsed, out_dir, dataset_name):
    n = len(nodes)
    ranks = np.argsort(r_power)[::-1]
    top_k = 15

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A: convergence
    ax = axes[0]
    ax.semilogy(history, color=BLUE, lw=2)
    ax.axhline(1e-10, color=RED, ls='--', lw=1, label="Tolerance 1e-10")
    ax.set_xlabel("Iteration"); ax.set_ylabel("L1 Residual")
    ax.set_title(f"Power-Iteration Convergence\n{dataset_name} (n={n:,}, {n_iters} iters, {elapsed:.1f}s)", fontweight='bold')
    ax.legend(fontsize=8)

    # B: top-k bar
    ax = axes[1]
    top_ids  = [nodes[ranks[i]] for i in range(top_k)]
    top_vals = [r_power[ranks[i]] for i in range(top_k)]
    colors   = [BLUE if i < 5 else ORANGE if i < 10 else GREEN for i in range(top_k)]
    bars = ax.barh(range(top_k-1, -1, -1), top_vals, color=colors, alpha=0.85)
    ax.set_yticks(range(top_k-1, -1, -1))
    ax.set_yticklabels([f"Node {v}" for v in top_ids], fontsize=7)
    ax.set_xlabel("PageRank Score")
    ax.set_title(f"Top-{top_k} Pages by PageRank\n({dataset_name})", fontweight='bold')

    # C: score distribution (log-log)
    ax = axes[2]
    sorted_r = np.sort(r_power)[::-1]
    ax.loglog(np.arange(1, n+1), sorted_r, color=BLUE, lw=1.5)
    ax.set_xlabel("Rank (log scale)"); ax.set_ylabel("PageRank Score (log scale)")
    ax.set_title("PageRank Score Distribution\n(Power-Law Tail)", fontweight='bold')

    fig.tight_layout()
    path = f"{out_dir}/fig3_large_graph.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ─────────────────────────────────────────────
# FIGURE 4: Closed-form vs Jacobi on 200-node sub-graph
# ─────────────────────────────────────────────
def fig4_subgraph_comparison(S, r_power, nodes, out_dir):
    M = 200      # sub-graph size for exact solve
    p = 0.15
    S_sub = S[:M, :M]

    # Closed form on sub-graph
    S_d = S_sub.toarray()
    n   = M
    v   = np.ones(n) / n
    A   = np.eye(n) - (1 - p) * S_d
    r_cf = np.linalg.solve(A, p * v)
    r_cf /= r_cf.sum()

    # Jacobi on sub-graph
    r_jac, hist, iters = pagerank_jacobi(S_sub, p=p)

    # Power on sub-graph (for triple comparison)
    r_pow_sub, _, _ = pagerank_jacobi(S_sub, p=p, tol=1e-12)

    l1  = np.abs(r_cf - r_jac).sum()
    linf = np.abs(r_cf - r_jac).max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A: scatter closed vs Jacobi
    ax = axes[0]
    ax.scatter(r_cf, r_jac, s=15, alpha=0.6, color=BLUE, edgecolors='none')
    mn, mx = r_cf.min(), r_cf.max()
    ax.plot([mn, mx], [mn, mx], color=RED, lw=1.5, ls='--', label="y=x (perfect)")
    ax.set_xlabel("Closed-form PageRank"); ax.set_ylabel("Jacobi PageRank")
    ax.set_title(f"Closed-Form vs Jacobi\n200-node sub-graph (L1={l1:.2e})", fontweight='bold')
    ax.legend(fontsize=8)

    # B: convergence Jacobi
    ax = axes[1]
    ax.semilogy(hist, color=ORANGE, lw=2)
    ax.axhline(1e-10, color=RED, ls='--', lw=1, label="Tolerance")
    ax.set_xlabel("Iteration"); ax.set_ylabel("L1 Residual")
    ax.set_title(f"Jacobi Convergence\n(sub-graph, {iters} iters)", fontweight='bold')
    ax.legend(fontsize=8)

    # C: rank correlation bar
    ax = axes[2]
    rank_cf  = np.argsort(np.argsort(-r_cf))
    rank_jac = np.argsort(np.argsort(-r_jac))
    ax.scatter(rank_cf, rank_jac, s=10, alpha=0.5, color=GREEN, edgecolors='none')
    ax.plot([0, M], [0, M], color=RED, lw=1.5, ls='--', label="y=x")
    ax.set_xlabel("Rank (closed-form)"); ax.set_ylabel("Rank (Jacobi)")
    ax.set_title("Rank Concordance\n(closed-form vs Jacobi)", fontweight='bold')
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = f"{out_dir}/fig4_subgraph_comparison.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return path, l1, linf, iters


# ─────────────────────────────────────────────
# FIGURE 5: AI Crawler Prioritization
# ─────────────────────────────────────────────
def fig5_crawler(out_dir):
    # Small directed web graph
    web_graph = {
        "/docs/pagerank-overview":  ["/docs/markov-chains", "/reference/eigenvectors"],
        "/docs/markov-chains":      ["/docs/pagerank-overview", "/blog/crawler-design"],
        "/reference/eigenvectors":  ["/docs/pagerank-overview", "/docs/markov-chains"],
        "/blog/crawler-design":     ["/blog/search-quality", "/search?q=pagerank"],
        "/blog/search-quality":     ["/docs/pagerank-overview"],
        "/search?q=pagerank":       ["/blog/search-quality"],
        "/login":                   ["/docs/pagerank-overview"],
        "/research?q=pagerank":     ["/docs/pagerank-overview", "/reference/eigenvectors"],
    }

    # Pre-computed PageRank (via small power iteration on this graph)
    all_pages = list(web_graph.keys())
    n  = len(all_pages)
    p2i = {p: i for i, p in enumerate(all_pages)}
    S  = np.zeros((n, n))
    for u, targets in web_graph.items():
        if targets:
            for v in targets:
                if v in p2i:
                    S[p2i[v], p2i[u]] += 1.0 / len(targets)
        else:
            S[:, p2i[u]] = 1.0 / n

    p = 0.15
    v = np.ones(n) / n
    r = v.copy()
    for _ in range(500):
        r_new = (1-p)*S.dot(r) + p*v
        r_new /= r_new.sum()
        if np.abs(r_new-r).sum() < 1e-12:
            break
        r = r_new
    pr = {page: r[i] for page, i in p2i.items()}

    # Quality heuristic
    GOOD_PATTERNS  = ['docs', 'reference', 'research', 'blog', 'guide', 'paper']
    BAD_PATTERNS   = ['login', 'logout', 'search?', 'cart', 'checkout', 'admin']
    ROBOTS_BLOCKED = {'/login', '/search?q=pagerank'}

    def quality_score(url):
        score = 50   # base
        for pat in GOOD_PATTERNS:
            if pat in url.lower():
                score += 20; break
        for pat in BAD_PATTERNS:
            if pat in url.lower():
                score -= 40; break
        if len(url) > 20: score += 5
        return max(0, min(100, score))

    rows = []
    for url in all_pages:
        qs      = quality_score(url)
        blocked = url in ROBOTS_BLOCKED
        final   = 50 * pr[url] + 0.5 * qs
        if blocked:
            final *= 0.05
        rows.append({
            "url": url, "pagerank": pr[url], "quality": qs,
            "blocked": blocked, "final": final
        })

    rows.sort(key=lambda x: -x['final'])
    top_k = 5

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # A: final score bar with colour by blocked status
    ax = axes[0]
    urls_short = [r['url'][-25:] for r in rows]
    finals     = [r['final']    for r in rows]
    colours_bar = [RED if r['blocked'] else GREEN for r in rows]
    bars = ax.barh(range(len(rows)-1, -1, -1), finals, color=colours_bar, alpha=0.85)
    ax.set_yticks(range(len(rows)-1, -1, -1))
    ax.set_yticklabels(urls_short, fontsize=7.5)
    ax.set_xlabel("Final Crawl Score")
    ax.set_title("Crawler Priority Ranking\n(Green=Allowed, Red=Blocked)", fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN,label='Allowed'),Patch(color=RED,label='Blocked')],
              fontsize=8, loc='lower right')

    # B: component breakdown for top-k
    ax = axes[1]
    top_rows = rows[:top_k]
    x   = np.arange(top_k)
    w   = 0.25
    pr_contrib = [50*r['pagerank'] for r in top_rows]
    qs_contrib = [0.5*r['quality'] for r in top_rows]
    fin_vals   = [r['final']       for r in top_rows]
    ax.bar(x-w,   pr_contrib, w, color=BLUE,   alpha=0.85, label="PageRank (×50)")
    ax.bar(x,     qs_contrib, w, color=ORANGE, alpha=0.85, label="Quality (×0.5)")
    ax.bar(x+w,   fin_vals,   w, color=GREEN,  alpha=0.85, label="Final Score")
    ax.set_xticks(x)
    ax.set_xticklabels([r['url'].split('/')[-1][:15] for r in top_rows], rotation=20, fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title(f"Score Components\n(Top-{top_k} URLs)", fontweight='bold')
    ax.legend(fontsize=7)

    # C: PageRank vs Quality scatter
    ax = axes[2]
    pr_vals = [r['pagerank'] for r in rows]
    qs_vals = [r['quality']  for r in rows]
    colours_sc = [RED if r['blocked'] else BLUE for r in rows]
    ax.scatter(pr_vals, qs_vals, c=colours_sc, s=80, alpha=0.85, edgecolors='white', lw=0.5)
    for r2 in rows:
        ax.annotate(r2['url'].split('/')[-1][:12],
                    (r2['pagerank'], r2['quality']),
                    fontsize=6, ha='left', va='bottom',
                    xytext=(2, 2), textcoords='offset points')
    ax.set_xlabel("PageRank Score"); ax.set_ylabel("Quality Score")
    ax.set_title("PageRank vs Quality Heuristic\n(Blocked pages in red)", fontweight='bold')

    fig.suptitle("AI Crawler Prioritization: GPTBot-like Strategy\n"
                 "FinalScore = 50·PageRank + 0.5·Quality  (blocked pages heavily penalised)",
                 fontweight='bold', fontsize=10, y=1.01)
    fig.tight_layout()
    path = f"{out_dir}/fig5_crawler.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return path, rows[:top_k]


# ─────────────────────────────────────────────
# FIGURE 6: p-sensitivity on large graph
# ─────────────────────────────────────────────
def fig6_large_p_sensitivity(S, nodes, out_dir, dataset_name):
    p_vals  = [0.05, 0.15, 0.30, 0.50, 0.85]
    results = {}
    for p in p_vals:
        r, _, iters = pagerank_jacobi(S, p=p, tol=1e-8, max_iter=150)
        results[p] = r

    n = len(nodes)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for p in p_vals:
        sorted_r = np.sort(results[p])[::-1][:500]
        ax.semilogy(np.arange(1, 501), sorted_r, lw=2, label=f"p={p}")
    ax.set_xlabel("Rank (top 500)"); ax.set_ylabel("PageRank Score (log)")
    ax.set_title(f"Top-500 Score Profiles for Different p\n({dataset_name})", fontweight='bold')
    ax.legend(fontsize=8)

    ax2 = axes[1]
    spreads = [results[p].max() - results[p].min() for p in p_vals]
    entropies = [-np.sum(results[p] * np.log(results[p] + 1e-15)) for p in p_vals]
    ax2twin = ax2.twinx()
    ax2.bar(range(len(p_vals)), spreads, color=BLUE, alpha=0.7, label="Score Spread")
    ax2twin.plot(range(len(p_vals)), entropies, 'o-', color=ORANGE, lw=2, label="Entropy H(r)")
    ax2.set_xticks(range(len(p_vals)))
    ax2.set_xticklabels([f"p={p}" for p in p_vals])
    ax2.set_ylabel("Score Spread (max−min)", color=BLUE)
    ax2twin.set_ylabel("Entropy H(r)", color=ORANGE)
    ax2.set_title("Ranking Inequality vs Entropy as p Varies\n(Large graph)", fontweight='bold')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2twin.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper right')

    fig.tight_layout()
    path = f"{out_dir}/fig6_p_sensitivity_large.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ─────────────────────────────────────────────
# TRUSTCHAIN PAGERANK EXTENSION
# ─────────────────────────────────────────────
DOMAIN_TRUST = {
    'academic':      1.0,
    'documentation': 0.95,
    'news':          0.70,
    'blog':          0.60,
    'social_media':  0.40,
    'unknown':       0.50,
}


def assign_domain_type(url):
    """Classify a URL's domain type based on patterns."""
    url_lower = url.lower()
    if any(p in url_lower for p in ['arxiv', 'scholar.google', 'research', 'paper', 'ieee']):
        return 'academic'
    elif any(p in url_lower for p in ['docs.', 'docs-', 'documentation', 'api', 'reference']):
        return 'documentation'
    elif any(p in url_lower for p in ['reuters', 'bbc', 'cnn', 'guardian', 'nytimes']):
        return 'news'
    elif any(p in url_lower for p in ['medium', 'blog', 'wordpress', 'substack']):
        return 'blog'
    elif any(p in url_lower for p in ['reddit', 'twitter', 'facebook', 'instagram', 'linkedin']):
        return 'social_media'
    else:
        return 'unknown'


def trust_weighted_pagerank(web_graph, domain_types, p=0.15, tol=1e-10, max_iter=500):
    """
    Compute TrustChain PageRank.

    Parameters
    ----------
    web_graph   : dict  {url: [outlink_url, ...]}
    domain_types: dict  {url: domain_type_string}
    p           : float teleportation probability
    Returns
    -------
    pr          : dict  {url: score}
    history     : list  L1 residuals per iteration
    """
    pages = list(web_graph.keys())
    n     = len(pages)
    p2i   = {pg: i for i, pg in enumerate(pages)}

    # Build weighted column-stochastic matrix
    S = np.zeros((n, n))
    for u, targets in web_graph.items():
        valid = [v for v in targets if v in p2i]
        if valid:
            trust_u = DOMAIN_TRUST.get(domain_types.get(u, 'unknown'), 0.50)
            for v in valid:
                S[p2i[v], p2i[u]] += trust_u / len(valid)
        else:
            # dangling: uniform with trust weight
            trust_u = DOMAIN_TRUST.get(domain_types.get(u, 'unknown'), 0.50)
            S[:, p2i[u]] = trust_u / n

    v   = np.ones(n) / n
    r   = v.copy()
    history = []
    for _ in range(max_iter):
        r_new = (1 - p) * S.dot(r) + p * v
        r_new /= r_new.sum()          # re-normalise
        delta  = np.abs(r_new - r).sum()
        history.append(delta)
        r = r_new
        if delta < tol:
            break

    return {pg: r[p2i[pg]] for pg in pages}, history


def fig7_trustchain(out_dir):
    """Generate the TrustChain PageRank comparison figure."""
    import matplotlib.patches as mpatches

    # Web graph with richer domain diversity
    web_graph = {
        'arxiv.org/pagerank-paper':        ['docs.python.org/scipy', 'scholar.google.com/citations'],
        'scholar.google.com/citations':    ['arxiv.org/pagerank-paper', 'docs.python.org/scipy'],
        'docs.python.org/scipy':           ['arxiv.org/pagerank-paper'],
        'medium.com/pagerank-explained':   ['arxiv.org/pagerank-paper', 'docs.python.org/scipy'],
        'reddit.com/r/MachineLearning':    ['arxiv.org/pagerank-paper', 'medium.com/pagerank-explained'],
        'twitter.com/ai_news':             ['reddit.com/r/MachineLearning', 'medium.com/pagerank-explained'],
        'reuters.com/ai-article':          ['arxiv.org/pagerank-paper', 'docs.python.org/scipy'],
        'bbc.com/tech/ai-story':           ['reuters.com/ai-article', 'twitter.com/ai_news'],
    }

    domain_types = {url: assign_domain_type(url) for url in web_graph}

    # Standard PageRank (all trust = 1.0)
    uniform_trust = {url: 'unknown' for url in web_graph}  # all weight 0.5? No — use 1.0 baseline
    # For fair comparison, build standard PR with equal weights
    pages = list(web_graph.keys())
    n = len(pages)
    p2i = {pg: i for i, pg in enumerate(pages)}
    S_std = np.zeros((n, n))
    for u, targets in web_graph.items():
        valid = [v for v in targets if v in p2i]
        if valid:
            for v in valid:
                S_std[p2i[v], p2i[u]] += 1.0 / len(valid)
        else:
            S_std[:, p2i[u]] = 1.0 / n

    p_val = 0.15
    v_vec = np.ones(n) / n
    r_std = v_vec.copy()
    for _ in range(500):
        r_new = (1 - p_val) * S_std.dot(r_std) + p_val * v_vec
        r_new /= r_new.sum()
        if np.abs(r_new - r_std).sum() < 1e-12: break
        r_std = r_new
    pr_std = {pg: r_std[p2i[pg]] for pg in pages}

    pr_trust, hist_trust = trust_weighted_pagerank(web_graph, domain_types, p=p_val)

    # ── sensitivity: vary trust weights ──
    weight_scales  = np.linspace(0.1, 1.0, 20)
    top_node_trust = []
    top_node_std   = [max(pr_std.values())] * 20
    for scale in weight_scales:
        scaled_trust = {dt: min(1.0, w * scale / 0.5)
                        for dt, w in DOMAIN_TRUST.items()}
        # quick re-run
        S_sc = np.zeros((n, n))
        for u, targets in web_graph.items():
            valid = [v for v in targets if v in p2i]
            tw = scaled_trust.get(domain_types.get(u, 'unknown'), 0.5)
            if valid:
                for v in valid:
                    S_sc[p2i[v], p2i[u]] += tw / len(valid)
            else:
                S_sc[:, p2i[u]] = tw / n
        r_sc = v_vec.copy()
        for _ in range(300):
            r_n = (1-p_val)*S_sc.dot(r_sc)+p_val*v_vec; r_n/=r_n.sum()
            if np.abs(r_n-r_sc).sum()<1e-10: break
            r_sc = r_n
        top_node_trust.append(r_sc.max())

    # ── Figure layout ──
    fig = plt.figure(figsize=(17, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    DOMAIN_COLOURS = {
        'academic':      '#1f4e79',
        'documentation': '#375623',
        'news':          '#843c0c',
        'blog':          '#7b3f00',
        'social_media':  '#c00000',
        'unknown':       '#595959',
    }

    # A: bar comparison std vs trust
    ax = fig.add_subplot(gs[0, 0])
    x   = np.arange(n)
    w   = 0.35
    std_vals   = [pr_std[pg]   for pg in pages]
    trust_vals = [pr_trust[pg] for pg in pages]
    ax.bar(x - w/2, std_vals,   w, color=BLUE,   alpha=0.85, label='Standard PR')
    ax.bar(x + w/2, trust_vals, w, color=ORANGE, alpha=0.85, label='TrustChain PR')
    ax.set_xticks(x)
    ax.set_xticklabels([pg.split('/')[0][:14] for pg in pages], rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('PageRank Score')
    ax.set_title('Standard vs TrustChain PageRank\n(per page)', fontweight='bold')
    ax.legend(fontsize=8)

    # B: domain-type trust weight bar
    ax2 = fig.add_subplot(gs[0, 1])
    dtypes = list(DOMAIN_TRUST.keys())
    dvals  = list(DOMAIN_TRUST.values())
    dcols  = [DOMAIN_COLOURS[d] for d in dtypes]
    ax2.bar(range(len(dtypes)), dvals, color=dcols, alpha=0.85, edgecolor='white')
    ax2.set_xticks(range(len(dtypes)))
    ax2.set_xticklabels(dtypes, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Trust Weight T[domain]')
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Domain-Type Trust Weights\n(TrustChain schema)', fontweight='bold')
    ax2.axhline(1.0, color='grey', lw=0.8, ls='--', alpha=0.5)

    # C: convergence
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(hist_trust, color=GREEN, lw=2, label='TrustChain')
    ax3.axhline(1e-10, color=RED, ls='--', lw=1, label='Tolerance')
    ax3.set_xlabel('Iteration'); ax3.set_ylabel('L1 Residual')
    ax3.set_title(f'TrustChain Convergence\n({len(hist_trust)} iterations)', fontweight='bold')
    ax3.legend(fontsize=8)

    # D: domain-coloured scatter: std PR vs trust PR
    ax4 = fig.add_subplot(gs[1, 0])
    for pg in pages:
        dt   = domain_types[pg]
        col  = DOMAIN_COLOURS.get(dt, '#595959')
        ax4.scatter(pr_std[pg], pr_trust[pg], color=col, s=90, zorder=3,
                    edgecolors='white', lw=0.5)
        ax4.annotate(pg.split('/')[0][:10], (pr_std[pg], pr_trust[pg]),
                     fontsize=6, xytext=(3, 3), textcoords='offset points')
    mn = min(list(pr_std.values()) + list(pr_trust.values()))
    mx = max(list(pr_std.values()) + list(pr_trust.values()))
    ax4.plot([mn, mx], [mn, mx], 'k--', lw=1, label='y=x (no change)')
    ax4.set_xlabel('Standard PageRank'); ax4.set_ylabel('TrustChain PageRank')
    ax4.set_title('Score Shift: Standard → TrustChain\n(above y=x = promoted)', fontweight='bold')
    patches = [mpatches.Patch(color=DOMAIN_COLOURS[d], label=d) for d in DOMAIN_COLOURS]
    ax4.legend(handles=patches, fontsize=6, loc='upper left')

    # E: rank reordering heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    rank_std   = {pg: r for r, pg in enumerate(sorted(pages, key=lambda x: -pr_std[x]))}
    rank_trust = {pg: r for r, pg in enumerate(sorted(pages, key=lambda x: -pr_trust[x]))}
    rank_deltas = [rank_std[pg] - rank_trust[pg] for pg in pages]
    bar_cols = [GREEN if d > 0 else RED if d < 0 else GREY for d in rank_deltas]
    ax5.barh(range(n), rank_deltas, color=bar_cols, alpha=0.85)
    ax5.set_yticks(range(n))
    ax5.set_yticklabels([pg.split('/')[0][:14] for pg in pages], fontsize=7)
    ax5.axvline(0, color='black', lw=0.8)
    ax5.set_xlabel('Rank Change (+ = promoted by TrustChain)')
    ax5.set_title('Rank Reordering\n(TrustChain vs Standard)', fontweight='bold')

    # F: sensitivity of top-node score to weight scale
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(weight_scales, top_node_std,  color=BLUE,   lw=2, ls='--', label='Standard PR (top node)')
    ax6.plot(weight_scales, top_node_trust, color=ORANGE, lw=2, label='TrustChain PR (top node)')
    ax6.fill_between(weight_scales, top_node_std, top_node_trust,
                     alpha=0.15, color=GREEN, label='Trust advantage')
    ax6.set_xlabel('Trust Weight Scale Factor')
    ax6.set_ylabel('Top-Node PageRank Score')
    ax6.set_title('Sensitivity to Trust Weight Scale\n(how much trust amplifies authority)', fontweight='bold')
    ax6.legend(fontsize=8)

    fig.suptitle(
        'TrustChain PageRank Extension\n'
        r'$r_{t+1}[v] = (1-p)\sum_u T[\mathrm{domain}(u)]\cdot S[v,u]\cdot r_t[u] + p\cdot v[v]$',
        fontweight='bold', fontsize=11)

    path = f'{out_dir}/fig7_trustchain.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')
    return path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import os

    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="SC4052 Q6 PageRank analysis")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(base_dir / "data" / "web-Google.txt"),
        help="Path to edge-list dataset (default: data/web-Google.txt)",
    )
    args = parser.parse_args()

    out_dir = base_dir / "outputs"
    data_path = Path(args.dataset)
    dataset_name = data_path.name

    os.makedirs(out_dir, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print("=" * 60)
    print("SC4052 Cloud Computing – Question 6 PageRank Solution")
    print("=" * 60)

    # ── Fig 1: p sensitivity toy graph
    print("\n[1/6] Generating p-sensitivity figure (toy graph)...")
    p1 = fig1_p_sensitivity(out_dir)

    # ── Fig 2: method comparison toy graph
    print("[2/6] Generating method comparison figure (toy graph)...")
    p2, l1_toy, l2_toy, linf_toy, iters_toy = fig2_method_comparison(out_dir)

    # ── Load large graph
    print(f"[3/6] Loading {dataset_name}...")
    S, n2i, node_list, edge_count = build_sparse_stochastic_from_file(data_path)
    n = len(node_list)
    print(f"      Nodes: {n:,}  Edges: {edge_count:,}  Matrix nnz: {S.nnz:,}")

    # ── Power iteration on large graph
    print("      Running power iteration...")
    r_power, history, n_iters, elapsed = pagerank_power(S, p=0.15)
    print(f"      Converged in {n_iters} iterations ({elapsed:.2f}s)")

    # ── Fig 3: large graph results
    print("[4/6] Generating large-graph figure...")
    p3 = fig3_large_graph(S, node_list, r_power, history, n_iters, elapsed, out_dir, dataset_name)

    # ── Fig 4: sub-graph closed-form vs Jacobi
    print("[5/6] Generating sub-graph comparison figure...")
    p4, l1_sub, linf_sub, iters_sub = fig4_subgraph_comparison(S, r_power, node_list, out_dir)

    # ── Fig 5: crawler
    print("[6a/6] Generating AI crawler figure...")
    p5, top_urls = fig5_crawler(out_dir)

    # ── Fig 6: p sensitivity large graph
    print("[6b/6] Generating large-graph p-sensitivity figure...")
    p6 = fig6_large_p_sensitivity(S, node_list, out_dir, dataset_name)

    # ── Fig 7: TrustChain PageRank extension
    print("[7/7] Generating TrustChain PageRank figure...")
    p7 = fig7_trustchain(str(out_dir))

    # ── Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(f"\n[Toy graph – method comparison]")
    print(f"  L1 error:   {l1_toy:.3e}")
    print(f"  L2 error:   {l2_toy:.3e}")
    print(f"  L∞ error:   {linf_toy:.3e}")
    print(f"  Iterations: {iters_toy}")

    print(f"\n[200-node sub-graph]")
    print(f"  L1 error:   {l1_sub:.3e}")
    print(f"  L∞ error:   {linf_sub:.3e}")
    print(f"  Iterations: {iters_sub}")

    print(f"\n[Large graph – {dataset_name}]")
    print(f"  Nodes: {n:,}, Edges: {edge_count:,}")
    print(f"  Power iteration: {n_iters} iters, {elapsed:.2f}s")
    top5 = np.argsort(r_power)[::-1][:5]
    print("  Top-5 nodes:")
    for rank, idx in enumerate(top5, 1):
        print(f"    {rank}. Node {node_list[idx]:>7}  score={r_power[idx]:.6f}")

    print(f"\n[AI Crawler – Top URLs to crawl]")
    for i, row in enumerate(top_urls, 1):
        status = "BLOCKED" if row['blocked'] else "allowed"
        print(f"  {i}. {row['url']:35s}  PR={row['pagerank']:.4f}  Q={row['quality']}  Final={row['final']:.3f}  [{status}]")

    print(f"\n[TrustChain PageRank Extension]")
    print(f"  Domain trust weights applied: {list(DOMAIN_TRUST.keys())}")
    print(f"  Figure generated with domain-aware ranking analysis")

    print("\nAll figures saved to:", str(out_dir))
