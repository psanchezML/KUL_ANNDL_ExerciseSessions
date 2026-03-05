"""
Modern Hopfield Networks & Self-Attention — Educational Demo
=============================================================
Bridges classical Hopfield networks to Modern Hopfield Networks and shows the
equivalence with Transformer self-attention.  Uses MNIST digits for intuitive
visualisation wherever possible.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Modern Hopfield Networks & Self-Attention",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# MNIST helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading MNIST …")
def load_mnist():
    """Return one clean bipolar representative per digit class (0-9)."""
    try:
        from keras.datasets import mnist
        (x, y), _ = mnist.load_data()
        out = {}
        for d in range(10):
            out[d] = np.where(
                x[np.where(y == d)[0][0]].astype(float) >= 128, 1.0, -1.0
            ).flatten()
        return out
    except (ImportError, ModuleNotFoundError):
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        x, y = bunch.data.astype(float), bunch.target.astype(int)
        out = {}
        for d in range(10):
            out[d] = np.where(x[np.where(y == d)[0][0]] >= 128, 1.0, -1.0)
        return out


def add_noise(pattern, frac, rng):
    """Flip *frac* fraction of bits in a bipolar pattern."""
    p = pattern.copy()
    n_flip = max(1, int(frac * len(p)))
    p[rng.choice(len(p), n_flip, replace=False)] *= -1
    return p


def to_pil(pattern, display_size=140):
    """Bipolar vector -> PIL Image upscaled with nearest-neighbour."""
    arr = ((pattern.reshape(28, 28) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").resize(
        (display_size, display_size), Image.NEAREST
    )


# ──────────────────────────────────────────────────────────────────────────────
# Core Hopfield helpers
# ──────────────────────────────────────────────────────────────────────────────

def classical_hopfield_weights(patterns: np.ndarray) -> np.ndarray:
    """Hebbian weight matrix.  patterns: (P, d) with +/-1 entries."""
    P, d = patterns.shape
    W = (patterns.T @ patterns) / P
    np.fill_diagonal(W, 0)
    return W


def classical_hopfield_retrieve(W, query, steps=20):
    """Synchronous sign-activation retrieval.  Returns list of states."""
    trajectory = [query.copy()]
    q = query.copy().astype(float)
    for _ in range(steps):
        q_new = np.sign(W @ q)
        q_new[q_new == 0] = 1
        trajectory.append(q_new.copy())
        if np.array_equal(q_new, q):
            break
        q = q_new
    return trajectory


def classical_energy(W, state):
    return -0.5 * state @ W @ state


def modern_hopfield_energy(X, q, beta):
    """X: (d, P) stored patterns as columns, q: (d,), beta: float."""
    logits = beta * X.T @ q
    mx = np.max(logits)
    return -np.log(np.sum(np.exp(logits - mx))) - mx + 0.5 * q @ q


def modern_hopfield_update(X, q, beta):
    """One Modern Hopfield update step."""
    logits = beta * X.T @ q
    logits -= np.max(logits)
    weights = np.exp(logits) / np.sum(np.exp(logits))
    return X @ weights, weights


def modern_hopfield_retrieve(X, query, beta, steps=20):
    """Iterative retrieval.  Returns trajectories, energies, and final weights."""
    q = query.copy().astype(float)
    trajectory = [q.copy()]
    energies = [modern_hopfield_energy(X, q, beta)]
    last_w = None
    for _ in range(steps):
        q_new, last_w = modern_hopfield_update(X, q, beta)
        trajectory.append(q_new.copy())
        energies.append(modern_hopfield_energy(X, q_new, beta))
        if np.linalg.norm(q_new - q) < 1e-8:
            break
        q = q_new
    return trajectory, energies, last_w


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def hamming_distance(a, b):
    return int(np.sum(np.sign(a) != np.sign(b)))


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
ALL_REPS = load_mnist()

PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("Modern Hopfield Networks & Self-Attention")

with st.expander("What is this demo?", expanded=True):
    st.markdown(r"""
This demo bridges **three ideas**:

1. **Classical Hopfield Networks** — store patterns via Hebbian learning,
   recall by iterating sign(W q).  Capacity is limited to ~0.14 d patterns.
2. **Modern Hopfield Networks** (Ramsauer et al., 2021) — replace the sign
   activation with a **softmax**, giving exponential capacity and smooth
   energy landscapes.
3. **Transformer Self-Attention** — the Modern Hopfield update turns out to
   be **structurally identical** to one step of self-attention.

Use the **four tabs** below to explore each concept interactively.
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "Classical vs Modern Retrieval",
    "Self-Attention Equivalence",
    "Temperature Explorer",
    "Capacity Comparison",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Classical vs Modern Retrieval (MNIST)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Classical vs Modern Hopfield Retrieval")

    with st.expander("What does this tab show?", expanded=False):
        st.markdown("""
We store several MNIST digit images as patterns in both a **classical** and
a **modern** Hopfield network, then corrupt one digit with noise and ask
each network to recall the original.

- **Classical** uses the Hebbian weight matrix and sign activation —
  it works well within capacity (~0.14 x dimension) but degrades beyond it.
- **Modern** uses a softmax-weighted combination of stored patterns —
  it can store exponentially more patterns and produces smooth retrieval.

**What to try:**
- Store 3-4 digits with low noise — both should succeed.
- Store 8+ digits — classical starts failing (capacity ~0.14 x 784 = ~110,
  but interference between similar digit shapes causes earlier failure).
- Increase noise — modern degrades more gracefully than classical.
""")

    col_cfg, col_vis = st.columns([1, 2.5])

    with col_cfg:
        st.subheader("Controls")

        stored_digits_t1 = st.multiselect(
            "Digits to store", list(range(10)), [0, 1, 3, 7],
            key="t1_digits",
            help="Which MNIST digit prototypes to memorise in both networks.",
        )
        if len(stored_digits_t1) < 2:
            st.warning("Select at least 2 digits.")
            st.stop()

        query_digit_t1 = st.selectbox(
            "Query digit", list(range(10)), index=0, key="t1_query",
            help="The digit to corrupt with noise and try to recall. "
                 "Try a digit that is NOT stored to see what happens!",
        )
        noise_t1 = st.slider(
            "Noise level", 0.0, 0.60, 0.25, 0.05, key="t1_noise",
            help="Fraction of pixels randomly flipped in the query.",
        )
        beta_t1 = st.slider(
            "Beta (inverse temperature)", 0.5, 50.0, 10.0, 0.5,
            key="t1_beta",
            help="Controls how sharply the Modern Hopfield network focuses "
                 "on the nearest stored pattern.  Higher = sharper.",
        )
        seed_t1 = st.number_input("Seed", value=42, step=1, key="t1_seed")

        st.markdown("---")
        st.markdown(
            "| Control | Effect |\n"
            "|:--------|:-------|\n"
            "| **Digits to store** | More digits = more interference |\n"
            "| **Query digit** | Try stored vs non-stored digits |\n"
            "| **Noise** | Higher = harder recall |\n"
            "| **Beta** | Higher = sharper modern recall |"
        )

    # --- Compute ---
    dtup_t1 = tuple(sorted(stored_digits_t1))
    patterns_t1 = np.array([ALL_REPS[d] for d in dtup_t1])
    rng1 = np.random.default_rng(int(seed_t1))
    noisy_query = add_noise(ALL_REPS[query_digit_t1], noise_t1, rng1)

    W_t1 = classical_hopfield_weights(patterns_t1)
    cl_traj = classical_hopfield_retrieve(W_t1, noisy_query, steps=30)
    cl_energies = [classical_energy(W_t1, s) for s in cl_traj]
    cl_result = cl_traj[-1]

    X_t1 = patterns_t1.T.astype(float)
    mo_traj, mo_energies, mo_weights = modern_hopfield_retrieve(
        X_t1, noisy_query.astype(float), beta_t1, steps=30
    )
    mo_result = mo_traj[-1]

    cl_nearest = dtup_t1[np.argmax(patterns_t1 @ cl_result)]
    mo_nearest_idx = np.argmax(patterns_t1 @ mo_result)
    mo_nearest = dtup_t1[mo_nearest_idx]

    with col_vis:
        # Stored patterns gallery
        st.subheader("Stored Patterns")
        gal_cols = st.columns(min(len(dtup_t1), 10))
        for i, d in enumerate(dtup_t1):
            with gal_cols[i]:
                st.image(to_pil(ALL_REPS[d], 80), caption=f"Digit {d}")

        st.markdown("---")

        # Recall comparison
        st.subheader("Recall Comparison")
        is_novel = query_digit_t1 not in dtup_t1
        if is_novel:
            st.warning(
                f"Digit **{query_digit_t1}** is **not stored**. Both networks "
                "will map it to whichever stored pattern is closest."
            )

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.image(to_pil(ALL_REPS[query_digit_t1], 130), caption="Original")
        with rc2:
            st.image(to_pil(noisy_query, 130), caption=f"Noisy ({noise_t1:.0%})")
        with rc3:
            cl_ham = hamming_distance(cl_result, ALL_REPS[query_digit_t1])
            st.image(to_pil(cl_result, 130),
                     caption=f"Classical -> {cl_nearest}")
            st.caption(f"Hamming = {cl_ham}")
        with rc4:
            mo_ham = hamming_distance(np.sign(mo_result), ALL_REPS[query_digit_t1])
            st.image(to_pil(mo_result, 130),
                     caption=f"Modern -> {mo_nearest}")
            st.caption(f"Hamming = {mo_ham}")

        st.markdown("---")

        # Attention weights (modern)
        st.subheader("Modern Hopfield Attention Weights")
        st.caption(
            "How much the modern network 'attends' to each stored pattern. "
            "The output is a weighted combination of all stored patterns "
            "according to these weights."
        )
        if mo_weights is not None:
            fig_w1 = go.Figure(go.Bar(
                x=[f"Digit {d}" for d in dtup_t1],
                y=mo_weights,
                marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(dtup_t1))],
                text=[f"{w:.3f}" for w in mo_weights],
                textposition="auto",
            ))
            fig_w1.update_layout(
                yaxis_title="Attention weight", height=280,
                margin=dict(t=20, b=30, l=30, r=10),
            )
            st.plotly_chart(fig_w1, use_container_width=True)

        # Energy curves
        st.subheader("Energy over Iterations")
        st.caption(
            "Both networks minimise an energy function during recall. "
            "The modern energy decreases smoothly; classical energy may "
            "jump or oscillate near capacity limits."
        )
        fig_e1 = go.Figure()
        fig_e1.add_trace(go.Scatter(
            y=cl_energies, mode="lines+markers", name="Classical",
            line=dict(color="coral", width=2),
        ))
        fig_e1.add_trace(go.Scatter(
            y=mo_energies, mode="lines+markers", name="Modern",
            line=dict(color="royalblue", width=2),
        ))
        fig_e1.update_layout(
            xaxis_title="Iteration", yaxis_title="Energy",
            height=300, margin=dict(t=20, b=30, l=30, r=10),
        )
        st.plotly_chart(fig_e1, use_container_width=True)

    cap_t1 = int(0.14 * 784)
    st.info(
        f"**Capacity note:** Classical capacity ~ 0.14 x 784 = **~{cap_t1}** "
        f"patterns. You are storing **{len(dtup_t1)}** patterns. "
        "Even below the theoretical limit, digits with similar shapes "
        "(e.g. 3 and 8) cause interference in the classical network. "
        "The modern network handles this gracefully via soft attention."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Self-Attention Equivalence
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Modern Hopfield Update = Self-Attention")

    with st.expander("What does this tab show?", expanded=True):
        st.markdown(r"""
The **Modern Hopfield update** and **Transformer self-attention** are
**the same computation** with different names for the variables.

| Modern Hopfield | Self-Attention | Role |
|:----------------|:---------------|:-----|
| Stored patterns $\mathbf{X}$ (columns) | Keys $\mathbf{K}$ (rows) | The "memory" being queried |
| Query $\boldsymbol{\xi}$ | Query $\mathbf{q}$ | The input looking for a match |
| Inverse temperature $\beta$ | Scaling $1/\sqrt{d_k}$ | Controls sharpness of attention |
| Retrieved pattern $\boldsymbol{\xi}_\text{new}$ | Attention output | Weighted combination of stored items |

**Hopfield update:**
$$\boldsymbol{\xi}_\text{new} = \mathbf{X}\;\text{softmax}(\beta\,\mathbf{X}^\top\boldsymbol{\xi})$$

**Self-attention (single query):**
$$\text{Attn}(\mathbf{q},\mathbf{K},\mathbf{V}) = \mathbf{V}^\top\;\text{softmax}\!\left(\frac{\mathbf{K}\,\mathbf{q}}{\sqrt{d_k}}\right)$$

Setting $\mathbf{K}^\top\!=\!\mathbf{X}^\top$, $\mathbf{V}^\top\!=\!\mathbf{X}$,
$\beta\!=\!1/\sqrt{d_k}$ makes them **identical**.

Below, you can edit small matrices and verify the equivalence numerically.
""")

    col_inp, col_out = st.columns([1, 2])

    with col_inp:
        st.subheader("Controls")
        d_k = st.slider(
            "Dimension d_k", 2, 8, 3, key="t2_dk",
            help="Dimensionality of keys, queries, and values.",
        )
        n_keys = st.slider(
            "Number of keys / patterns T", 2, 8, 4, key="t2_T",
            help="How many stored patterns (keys) the network holds.",
        )
        seed_t2 = st.number_input("Seed", value=7, step=1, key="t2_seed")

        rng2 = np.random.default_rng(int(seed_t2))
        K_default = np.round(rng2.standard_normal((n_keys, d_k)), 2)
        V_default = np.round(rng2.standard_normal((n_keys, d_k)), 2)
        q_default = np.round(rng2.standard_normal(d_k), 2)

        st.markdown("**Query** q  (1 x d_k)")
        q_str = st.text_input(
            "q (comma-separated)", ",".join(map(str, q_default)), key="t2_q",
        )
        q_vec = np.array([float(x.strip()) for x in q_str.split(",")])

        st.markdown("**Keys** K  (T x d_k) -- one row per key")
        K_str = st.text_area(
            "K (rows on separate lines, values comma-separated)",
            "\n".join(",".join(map(str, row)) for row in K_default),
            height=120, key="t2_K",
        )
        K_mat = np.array([
            [float(x) for x in row.split(",")]
            for row in K_str.strip().splitlines()
        ])

        st.markdown("**Values** V  (T x d_k)")
        V_str = st.text_area(
            "V (same format as K)",
            "\n".join(",".join(map(str, row)) for row in V_default),
            height=120, key="t2_V",
        )
        V_mat = np.array([
            [float(x) for x in row.split(",")]
            for row in V_str.strip().splitlines()
        ])

        st.info(
            "**Tip:** Set V = K (paste the same values) to see exact "
            "equivalence between Hopfield and Attention outputs."
        )

    with col_out:
        beta_equiv = 1.0 / np.sqrt(d_k)

        # Modern Hopfield path
        X_hop = K_mat.T
        logits_hop = beta_equiv * X_hop.T @ q_vec
        weights_hop = softmax(logits_hop)
        result_hop = X_hop @ weights_hop

        # Self-Attention path
        scores_att = (K_mat @ q_vec) / np.sqrt(d_k)
        weights_att = softmax(scores_att)
        result_att = V_mat.T @ weights_att

        st.subheader("Step-by-step Computation")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Modern Hopfield")
            st.markdown(
                r"**Step 1:** Compute similarity logits: "
                r"$\beta\,\mathbf{X}^\top\mathbf{q}$"
            )
            st.write(np.round(logits_hop, 4))
            st.markdown(
                "**Step 2:** Softmax to get attention weights "
                "(how much each pattern contributes):"
            )
            st.write(np.round(weights_hop, 4))
            st.markdown(
                r"**Step 3:** Weighted sum of patterns: "
                r"$\mathbf{X} \cdot \text{weights}$"
            )
            st.write(np.round(result_hop, 4))

        with c2:
            st.markdown("#### Self-Attention")
            st.markdown(
                r"**Step 1:** Compute scaled dot-product scores: "
                r"$\mathbf{K}\mathbf{q}/\sqrt{d_k}$"
            )
            st.write(np.round(scores_att, 4))
            st.markdown(
                "**Step 2:** Softmax to get attention weights:"
            )
            st.write(np.round(weights_att, 4))
            st.markdown(
                r"**Step 3:** Weighted sum of values: "
                r"$\mathbf{V}^\top \cdot \text{weights}$"
            )
            st.write(np.round(result_att, 4))

        match = np.allclose(result_hop, result_att, atol=1e-6)
        if match:
            st.success(
                "The two outputs are **identical**. This confirms the "
                "structural equivalence when K = V (stored patterns serve "
                "as both keys and values)."
            )
        else:
            st.warning(
                "Outputs differ because V != K. In the Hopfield framework, "
                "keys and values are the same matrix (the stored patterns). "
                "Set V = K to see exact equivalence."
            )

        st.subheader("Attention Weights Comparison")
        st.caption(
            "These bar charts show how much weight each stored pattern "
            "(key) receives.  When K = V, the Hopfield weights and "
            "Attention weights are identical."
        )
        fig_w = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Hopfield weights", "Attention weights"],
        )
        fig_w.add_trace(go.Bar(
            x=[f"Pattern {i+1}" for i in range(len(weights_hop))],
            y=weights_hop,
            marker_color="#636EFA",
            text=[f"{w:.3f}" for w in weights_hop],
            textposition="auto",
        ), row=1, col=1)
        fig_w.add_trace(go.Bar(
            x=[f"Key {i+1}" for i in range(len(weights_att))],
            y=weights_att,
            marker_color="#EF553B",
            text=[f"{w:.3f}" for w in weights_att],
            textposition="auto",
        ), row=1, col=2)
        fig_w.update_layout(
            height=280, margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig_w, use_container_width=True)

        st.metric(
            "Modern Hopfield Energy at query",
            f"{modern_hopfield_energy(X_hop, q_vec, beta_equiv):.4f}",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Temperature (beta) Explorer (MNIST)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Temperature (Beta) Explorer")

    with st.expander("What does this tab show?", expanded=True):
        st.markdown(r"""
The **inverse temperature** $\beta$ controls how sharply the Modern Hopfield
network focuses on the nearest stored pattern:

| Beta value | Behaviour | Analogy |
|:-----------|:----------|:--------|
| **Low** (e.g. 0.1) | Weights are nearly **uniform** across all patterns | Blurry average of everything |
| **Medium** (e.g. 5) | Weights **favour** the closest pattern but others contribute | Soft recall with ghosting |
| **High** (e.g. 50) | Weight concentrates on the **single closest** pattern | Crisp nearest-neighbour lookup |

In Transformers, $\beta = 1/\sqrt{d_k}$, so the effective temperature
scales with the key dimension.

**What to look for below:** as you drag the beta slider, watch the
retrieved image transition from a ghostly blend of all stored digits
(low beta) to a sharp, clean single digit (high beta).
""")

    col_t3l, col_t3r = st.columns([1, 2.5])

    with col_t3l:
        st.subheader("Controls")
        stored_digits_t3 = st.multiselect(
            "Digits to store", list(range(10)), [1, 4, 7],
            key="t3_digits",
            help="Which digit prototypes are in memory.",
        )
        if len(stored_digits_t3) < 2:
            st.warning("Select at least 2 digits.")
            st.stop()

        query_digit_t3 = st.selectbox(
            "Query digit", stored_digits_t3, index=0, key="t3_query",
            help="The digit to corrupt and present as query.",
        )
        noise_t3 = st.slider(
            "Noise", 0.0, 0.60, 0.20, 0.05, key="t3_noise",
            help="Fraction of pixels flipped.",
        )
        beta_t3 = st.slider(
            "Beta (inverse temperature)", 0.01, 80.0, 5.0, 0.1,
            key="t3_beta",
            help="**THE key slider.** Drag it to watch the transition "
                 "from soft average to nearest-neighbour.",
        )
        seed_t3 = st.number_input("Seed", value=7, step=1, key="t3_seed")

        st.markdown("---")
        st.markdown(
            "| Control | Effect |\n"
            "|:--------|:-------|\n"
            "| **Beta low** | Blurry superposition of all digits |\n"
            "| **Beta high** | Crisp recall of nearest digit |\n"
            "| **More digits** | More patterns visible at low beta |"
        )

    dtup_t3 = tuple(sorted(stored_digits_t3))
    patterns_t3 = np.array([ALL_REPS[d] for d in dtup_t3])
    X_t3 = patterns_t3.T.astype(float)

    rng3 = np.random.default_rng(int(seed_t3))
    noisy_t3 = add_noise(ALL_REPS[query_digit_t3], noise_t3, rng3)

    # Retrieve at selected beta
    logits_t3 = beta_t3 * X_t3.T @ noisy_t3
    weights_t3 = softmax(logits_t3)
    retrieved_t3 = X_t3 @ weights_t3

    with col_t3r:
        # Stored patterns
        st.subheader("Stored Patterns")
        gal_t3 = st.columns(min(len(dtup_t3), 10))
        for i, d in enumerate(dtup_t3):
            with gal_t3[i]:
                st.image(to_pil(ALL_REPS[d], 80), caption=f"Digit {d}")

        st.markdown("---")

        # Current retrieval
        st.subheader(f"Retrieval at Beta = {beta_t3:.1f}")

        r1, r2, r3 = st.columns([1, 1, 1.5])
        with r1:
            st.image(to_pil(noisy_t3, 160), caption="Noisy query")
        with r2:
            st.image(to_pil(retrieved_t3, 160), caption="Retrieved pattern")
        with r3:
            st.markdown("**Attention weights** (per stored pattern):")
            fig_bw = go.Figure(go.Bar(
                x=[f"Digit {d}" for d in dtup_t3],
                y=weights_t3,
                marker_color=[PALETTE[i % len(PALETTE)]
                              for i in range(len(dtup_t3))],
                text=[f"{w:.3f}" for w in weights_t3],
                textposition="auto",
            ))
            fig_bw.update_layout(
                yaxis_title="Weight", height=220,
                margin=dict(t=10, b=30, l=30, r=10),
            )
            st.plotly_chart(fig_bw, use_container_width=True)

        dominant_idx = np.argmax(weights_t3)
        dominant_digit = dtup_t3[dominant_idx]
        dominant_w = weights_t3[dominant_idx]
        entropy = -np.sum(weights_t3 * np.log(weights_t3 + 1e-12))
        max_entropy = np.log(len(dtup_t3))

        m1, m2, m3 = st.columns(3)
        m1.metric("Dominant pattern", f"Digit {dominant_digit} ({dominant_w:.1%})")
        m2.metric("Weight entropy", f"{entropy:.3f}")
        m3.metric(
            "Sharpness",
            f"{(1 - entropy / max_entropy) * 100:.0f}%",
            help="0% = perfectly uniform (all patterns equal), "
                 "100% = all weight on one pattern.",
        )

        st.markdown("---")

        # Filmstrip across beta values
        st.subheader("Filmstrip: Retrieval Across Beta Values")
        st.caption(
            "Each image shows what the network retrieves at a different "
            "beta.  Low beta = blurry blend of all stored digits.  "
            "High beta = sharp single-digit recall."
        )

        beta_range = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        film_cols = st.columns(len(beta_range))
        for k, b in enumerate(beta_range):
            lg = b * X_t3.T @ noisy_t3
            w = softmax(lg)
            ret = X_t3 @ w
            with film_cols[k]:
                st.image(to_pil(ret, 80), caption=f"b={b}")

        # Weight distribution chart
        st.subheader("Attention Weight Distribution Across Beta")
        st.caption(
            "At low beta, weights are nearly uniform (every pattern "
            "contributes equally).  As beta increases, weight concentrates "
            "on the closest pattern."
        )
        fig_wd = go.Figure()
        for b in beta_range:
            lg = b * X_t3.T @ noisy_t3
            w = softmax(lg)
            fig_wd.add_trace(go.Bar(
                x=[f"Digit {d}" for d in dtup_t3],
                y=w,
                name=f"b={b}",
            ))
        fig_wd.update_layout(
            barmode="group",
            xaxis_title="Stored pattern",
            yaxis_title="Weight",
            height=350,
            margin=dict(t=20, b=30, l=30, r=10),
            legend_title="Beta",
        )
        st.plotly_chart(fig_wd, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Capacity Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Storage Capacity: Classical vs Modern")

    with st.expander("What does this tab show?", expanded=True):
        st.markdown(r"""
**Storage capacity** is the maximum number of patterns a Hopfield network
can reliably store and retrieve.

| Network | Capacity | Why? |
|:--------|:---------|:-----|
| **Classical** | ~0.14 x d (linear in dimension) | Hebbian weight matrix creates cross-talk between patterns |
| **Modern** | Exponential in d | Softmax separates patterns much more effectively |

This tab runs a systematic test: for increasing numbers of stored random
binary patterns, it checks whether each network can perfectly recall a
randomly chosen stored pattern.  The resulting curve shows where each
network breaks down.

**Controls explained:**
- **Dimension d**: length of each pattern vector (higher = more capacity).
- **Max patterns**: how far to push the test.
- **Trials per count**: averaged over this many random pattern sets (more = smoother curve).
- **Beta**: inverse temperature for the modern network.
""")

    col_cap, col_plot = st.columns([1, 2])

    with col_cap:
        st.subheader("Controls")
        d_cap = st.slider(
            "Dimension d", 8, 128, 32, step=4, key="t4_d",
            help="Length of each binary pattern.  Classical capacity = ~0.14 x d.",
        )
        max_patterns = st.slider(
            "Max patterns to test", 5, 200, min(80, 5 * d_cap), step=5,
            key="t4_max",
            help="The test sweeps from 1 to this many stored patterns.",
        )
        n_trials = st.slider(
            "Trials per count", 1, 20, 5, key="t4_trials",
            help="Each data point is averaged over this many random trials.",
        )
        beta_cap = st.slider(
            "Modern beta", 1.0, 30.0, 10.0, step=1.0, key="t4_beta",
            help="Inverse temperature for the modern network.",
        )
        seed_cap = st.number_input("Seed", value=123, step=1, key="t4_seed")

    pattern_counts = list(range(1, max_patterns + 1, max(1, max_patterns // 25)))
    classical_acc = []
    modern_acc = []

    rng4 = np.random.default_rng(int(seed_cap))

    progress_bar = col_plot.progress(0.0, text="Running capacity test ...")
    total_steps = len(pattern_counts)

    for step_idx, P in enumerate(pattern_counts):
        c_correct, m_correct = 0, 0
        for _ in range(n_trials):
            pats = rng4.choice([-1, 1], size=(P, d_cap))
            W_c = classical_hopfield_weights(pats)
            idx_c = rng4.integers(0, P)
            traj_c = classical_hopfield_retrieve(W_c, pats[idx_c].copy(), steps=30)
            if np.array_equal(traj_c[-1], pats[idx_c]):
                c_correct += 1
            X_c = pats.T.astype(float)
            m_traj, _, _ = modern_hopfield_retrieve(
                X_c, pats[idx_c].astype(float), beta_cap, steps=30
            )
            closest = pats[np.argmax(pats @ m_traj[-1])]
            if np.array_equal(closest, pats[idx_c]):
                m_correct += 1

        classical_acc.append(c_correct / n_trials)
        modern_acc.append(m_correct / n_trials)
        progress_bar.progress(
            (step_idx + 1) / total_steps,
            text=f"Tested P = {P} / {max_patterns}",
        )

    progress_bar.empty()

    with col_plot:
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Scatter(
            x=pattern_counts, y=classical_acc,
            mode="lines+markers", name="Classical Hopfield",
            line=dict(color="coral", width=2),
        ))
        fig_cap.add_trace(go.Scatter(
            x=pattern_counts, y=modern_acc,
            mode="lines+markers", name="Modern Hopfield",
            line=dict(color="royalblue", width=2),
        ))
        cap_line = 0.14 * d_cap
        fig_cap.add_vline(
            x=cap_line, line_dash="dash", line_color="gray",
            annotation_text=f"0.14*d = {cap_line:.1f}",
            annotation_position="top right",
        )
        fig_cap.update_layout(
            title=f"Retrieval accuracy vs stored patterns  (d = {d_cap})",
            xaxis_title="Number of stored patterns P",
            yaxis_title="Retrieval accuracy",
            yaxis_range=[-0.05, 1.05],
            height=450,
            margin=dict(t=50, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_cap, use_container_width=True)

    st.markdown(f"""
**How to read this plot (d = {d_cap}):**

- The **coral line** (classical) drops sharply around P = {cap_line:.0f}
  (the dashed line), which is the theoretical capacity ~0.14 x d.
- The **blue line** (modern) stays near 100% far beyond that threshold
  thanks to the exponential capacity of the softmax energy.

| Property | Classical | Modern |
|:---------|:----------|:-------|
| Capacity scaling | ~0.14 x d = **{cap_line:.1f}** | **Exponential** in d |
| Update rule | sign(W x q) | X * softmax(beta X^T q) |
| Energy | -1/2 q^T W q | -log sum exp(beta xi^T q) + 1/2 ||q||^2 |
| Convergence | May oscillate near capacity | Smooth, monotone decrease |

**Try increasing d** to see how both capacity curves shift — classical
grows linearly while modern capacity grows much faster.
""")
