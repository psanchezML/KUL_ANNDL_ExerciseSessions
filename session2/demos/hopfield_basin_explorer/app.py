"""Hopfield Basin Explorer — interactive Streamlit demo for MNIST digit recall."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from hopfield import HopfieldNetwork

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hopfield Basin Explorer", layout="wide")

PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

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
    p[rng.choice(len(p), int(frac * len(p)), replace=False)] *= -1
    return p


def to_pil(pattern, display_size=140):
    """Bipolar vector -> PIL Image upscaled with nearest-neighbour."""
    arr = ((pattern.reshape(28, 28) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").resize(
        (display_size, display_size), Image.NEAREST
    )


@st.cache_resource(show_spinner="Training Hopfield network …")
def build_net(digits, alg):
    return HopfieldNetwork(np.array([ALL_REPS[d] for d in digits]), alg=alg)


@st.cache_data(show_spinner="Fitting PCA …")
def fit_pca(digits, _h):
    return PCA(n_components=2).fit(np.array([ALL_REPS[d] for d in digits]))


# ── Data ─────────────────────────────────────────────────────────────────────
ALL_REPS = load_mnist()

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
sb = st.sidebar
sb.title("Hopfield Basin Explorer")
sb.markdown("---")

sb.subheader("Network Configuration")
stored_digits = sb.multiselect(
    "Digits to store", list(range(10)), [0, 1, 2, 3, 4],
    help="Prototype digit classes memorised by the network.",
)
if not stored_digits:
    st.warning("Select at least one digit to store.")
    st.stop()

algorithm = sb.radio(
    "Learning rule", ["Hebb", "LSSM"], index=1,
    help="**Hebb**: outer-product rule.  **LSSM**: pseudo-inverse (higher capacity).",
)

sb.markdown("---")
sb.subheader("Recall Settings")

query_digit = sb.selectbox(
    "Digit to corrupt & recall", list(range(10)),
    help="Pick any digit (0-9). If the digit is **not** stored, the network "
         "will pull it toward the nearest stored pattern — try it!",
)
noise = sb.slider(
    "Noise level", 0.0, 0.80, 0.30, 0.05,
    help="Fraction of pixels randomly flipped.",
)
max_iter = sb.slider("Max iterations", 1, 60, 20)
sync = sb.checkbox(
    "Synchronous update", True,
    help="Synchronous flips all neurons at once; asynchronous updates one at a time.",
)
seed = sb.number_input("Random seed", value=42, step=1)

# ── Build network ────────────────────────────────────────────────────────────
dtup = tuple(sorted(stored_digits))
net = build_net(dtup, algorithm)
tgts = net.targets
labs = list(dtup)
pca = fit_pca(dtup, hash(tgts.tobytes()))

# Sidebar metrics
sb.markdown("---")
sb.subheader("Network Metrics")
sb.markdown(f"**Dimension:** {net.D}  (28x28)")
sb.markdown(f"**Stored:** {net.num_patterns}  |  **Capacity (Hebb):** ~{net.theoretical_capacity}")
sb.table([
    {"Digit": str(d), "Stable": "Yes" if net.is_stable(tgts[i]) else "No"}
    for i, d in enumerate(dtup)
])

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════════

st.title("Hopfield Basin Explorer")

# ── Introduction ─────────────────────────────────────────────────────────────
with st.expander("What is this demo?", expanded=True):
    st.markdown(r"""
A **Hopfield network** is a recurrent neural network that acts as a
**content-addressable (associative) memory**.  Patterns are stored as
**local minima** of an energy function — think of a ball resting at the
bottom of a valley on a hilly surface.

When a **noisy or partial** version of a stored pattern is presented, the
network's dynamics "roll downhill" in the energy landscape until the state
**converges** to the nearest stored pattern.  This is *recall by association*.

$$E(\mathbf{s}) \;=\; -\tfrac{1}{2}\,\mathbf{s}^{\!\top} W \mathbf{s} \;-\; \mathbf{b}^{\!\top}\mathbf{s}$$

At every update step, the energy either **decreases or stays the same**
(Lyapunov property), guaranteeing convergence.

### How to use this demo

| Control | Purpose |
|:--------|:--------|
| **Digits to store** | Choose which digit prototypes to memorise |
| **Learning rule** | *Hebb* (simple, lower capacity) vs *LSSM* (pseudo-inverse, higher capacity) |
| **Digit to recall** | Which stored digit to corrupt with noise |
| **Noise level** | Fraction of pixels flipped — higher = harder task |
| **Step slider** | Scrub through iterations: watch the image sharpen and energy drop |
| **Play (3-D chart)** | Animate the trajectory sliding downhill in the energy landscape |

### What to look for

- As you advance steps, the **image sharpens** and the **energy decreases**.
- In the 3-D landscape, the trajectory slides **downhill** toward a stored pattern.
- The **Hamming distance** to the nearest target drops toward zero when recall succeeds.
- Try storing **> 7 digits with Hebb** to see **capacity breakdown** (spurious attractors).
- Compare **Hebb vs LSSM** to see how the pseudo-inverse rule increases capacity.
""")

st.markdown("---")

# ── Stored Patterns Gallery ──────────────────────────────────────────────────

st.header("Stored Patterns")
st.markdown(
    "The network has memorised these **prototype images**.  "
    "Each one sits at a **local energy minimum** (attractor).  "
    "Any nearby state in the 784-dimensional space will be pulled toward "
    "one of these patterns during recall.\n\n"
    "Below each image: **E** is the Hopfield energy at that pattern "
    "(lower = deeper valley = stronger attractor), and **stable** means "
    "the pattern is a true fixed point — presenting it to the network "
    "returns itself unchanged."
)

cols_gal = st.columns(min(len(dtup), 10))
for i, d in enumerate(dtup):
    with cols_gal[i]:
        st.image(to_pil(tgts[i], 120), caption=f"Digit {d}")
        e = net.energy(tgts[i])
        stable = net.is_stable(tgts[i])
        st.caption(f"E = {e:.1f} | {'stable' if stable else 'unstable'}")

st.markdown("---")

# ── Run simulation ───────────────────────────────────────────────────────────

rng = np.random.default_rng(int(seed))
noisy_input = add_noise(ALL_REPS[query_digit], noise, rng)

states, energies = net.simulate(noisy_input, num_iter=max_iter, sync=sync)
traj = states if states.ndim == 2 else states[0]   # (D, T)
T = traj.shape[1]

hamm = np.empty(T, dtype=int)
near = np.empty(T, dtype=int)
for t in range(T):
    idx, dist = net.nearest_target(traj[:, t])
    hamm[t] = dist
    near[t] = labs[idx]

# ── Recall Summary ───────────────────────────────────────────────────────────

st.header("Recall Summary")

is_novel = query_digit not in dtup
if is_novel:
    st.warning(
        f"Digit **{query_digit}** is **not stored** in the network.  "
        "The network has never seen this pattern, so it will converge to "
        "whichever stored digit is closest in state space — this is "
        "associative memory in action!"
    )

st.markdown(
    "Overview of the full recall process — from the **clean original**, through "
    "**noise corruption**, to the **network's output** after convergence."
)

c_orig, c_noisy, c_arrow, c_final = st.columns([1, 1, 0.4, 1])
with c_orig:
    st.image(to_pil(ALL_REPS[query_digit], 160), caption="Original (clean)")
with c_noisy:
    st.image(to_pil(noisy_input, 160), caption=f"Noisy input ({noise:.0%})")
with c_arrow:
    st.markdown("")
    st.markdown("")
    st.markdown("### >>>")
with c_final:
    final_state = traj[:, -1]
    fidx, fdist = net.nearest_target(final_state)
    recalled_label = labs[fidx]
    success = fdist == 0
    tag = "perfect recall" if success else f"Hamming = {fdist}"
    st.image(to_pil(final_state, 160), caption=f"Recalled: digit {recalled_label} ({tag})")

if not success:
    st.warning(
        f"Recall is **imperfect** — the final state differs from digit "
        f"{recalled_label} by {fdist} pixels.  Try reducing noise, storing "
        f"fewer digits, or switching to LSSM."
    )
elif recalled_label != query_digit:
    st.warning(
        f"The network recalled digit **{recalled_label}** instead of "
        f"**{query_digit}** — the noisy input was closer to a different "
        f"basin of attraction."
    )
else:
    st.success(
        f"The network successfully recalled digit **{recalled_label}** "
        f"from a {noise:.0%}-corrupted input."
    )

st.markdown("---")

# ── Step-by-Step Recall ──────────────────────────────────────────────────────

st.header("Step-by-Step Recall")
st.markdown(
    f"Digit **{query_digit}** corrupted with **{noise:.0%}** noise.  "
    "Drag the slider to scrub through iterations and watch the image "
    "**sharpen** as the energy **drops**.\n\n"
    "- **Energy** measures how \"comfortable\" the network is in its current "
    "state — each update step is guaranteed to lower it (or keep it flat) "
    "until the state settles into a minimum.\n"
    "- **Hamming distance** counts how many pixels differ between the current "
    "state and the nearest stored pattern.  When it reaches **0**, recall is "
    "perfect."
)

step = st.slider("Iteration step", 0, T - 1, 0, key="recall_step")

col_img, col_charts = st.columns([1, 2.5])

with col_img:
    st.image(to_pil(traj[:, step], 220), caption=f"Step {step}")
    st.markdown(
        f"| Metric | Value |\n"
        f"|:---|---:|\n"
        f"| **Energy** | `{energies[step]:.2f}` |\n"
        f"| **Hamming** | `{hamm[step]}` |\n"
        f"| **Nearest** | digit `{near[step]}` |"
    )

with col_charts:
    fig_metrics, (ax_e, ax_h) = plt.subplots(
        1, 2, figsize=(9, 3), facecolor="#0E1117"
    )
    for ax in (ax_e, ax_h):
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("#444")

    ax_e.plot(energies, color="#636EFA", lw=2)
    ax_e.axvline(step, color="red", lw=1.5, ls="--", alpha=0.8)
    ax_e.scatter([step], [energies[step]], color="red", s=80, zorder=5)
    ax_e.set_xlabel("Iteration", color="white")
    ax_e.set_ylabel("Energy", color="white")
    ax_e.set_title("Energy over iterations", color="white", fontsize=11)

    ax_h.plot(hamm, color="#00CC96", lw=2)
    ax_h.axvline(step, color="red", lw=1.5, ls="--", alpha=0.8)
    ax_h.scatter([step], [hamm[step]], color="red", s=80, zorder=5)
    ax_h.set_xlabel("Iteration", color="white")
    ax_h.set_ylabel("Hamming distance", color="white")
    ax_h.set_title("Distance to nearest stored pattern", color="white", fontsize=11)

    fig_metrics.tight_layout()
    st.pyplot(fig_metrics, use_container_width=True)
    plt.close(fig_metrics)

# Filmstrip
st.subheader("Recall Filmstrip")
st.caption(
    "Key snapshots during recall.  Compare the first frame (noisy) to the "
    "last (recalled) to see how the network cleans up the input."
)
snap_idx = sorted(set([0, 1, 2, 3, 5, 10, 15, T // 2, T - 1]) & set(range(T)))
film_cols = st.columns(len(snap_idx))
for k, si in enumerate(snap_idx):
    with film_cols[k]:
        st.image(to_pil(traj[:, si], 90), caption=f"t = {si}")

st.markdown("---")

# ── 3-D Animated Energy Landscape ───────────────────────────────────────────

st.header("3-D Energy Landscape (Animated)")
st.markdown(r"""
Each digit image is a **784-dimensional** vector (28x28 pixels), so we
cannot plot trajectories directly.  We use **PCA (Principal Component
Analysis)** to compress those 784 dimensions into just **2** — PCA finds
the two directions along which the stored patterns vary the most, giving
the best possible 2-D "birds-eye view" of the state space.

| Axis | Meaning |
|:-----|:--------|
| **PC 1** (horizontal) | First principal component — the direction of greatest variance among stored patterns |
| **PC 2** (depth) | Second principal component — the next most informative direction |
| **Energy** (vertical) | Hopfield energy $E(\mathbf{s})$ — stored patterns sit at minima (valleys) |

- **Diamond markers** = stored digit prototypes sitting at energy minima
- **Gold trajectory** = path of the corrupted input through state space
- **Red sphere** = current network state

Press **Play** to watch the corrupted digit **slide downhill** toward the
nearest stored attractor.
""")

# Projections
tgt_2d = pca.transform(tgts)
tgt_E = np.array([net.energy(t) for t in tgts])
pts = traj.T                      # (T, D)
pts_2d = pca.transform(pts)       # (T, 2)
dcol = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(dtup)}

fig3d = go.Figure()

# Trace 0: stored patterns (constant across frames)
fig3d.add_trace(go.Scatter3d(
    x=tgt_2d[:, 0], y=tgt_2d[:, 1], z=tgt_E,
    mode="markers+text",
    marker=dict(
        size=10,
        color=[dcol[d] for d in dtup],
        symbol="diamond",
        line=dict(width=1, color="white"),
    ),
    text=[str(d) for d in dtup],
    textposition="top center",
    textfont=dict(size=14, color="white"),
    name="Stored patterns",
    hovertemplate=(
        "Digit %{text}<br>PC1 = %{x:.2f}<br>"
        "PC2 = %{y:.2f}<br>Energy = %{z:.1f}<extra></extra>"
    ),
))

# Trace 1: trajectory so far (animated)
fig3d.add_trace(go.Scatter3d(
    x=[pts_2d[0, 0]], y=[pts_2d[0, 1]], z=[energies[0]],
    mode="lines+markers",
    line=dict(width=4, color="gold"),
    marker=dict(size=3, color="gold"),
    name="Trajectory",
))

# Trace 2: current-position marker (animated)
fig3d.add_trace(go.Scatter3d(
    x=[pts_2d[0, 0]], y=[pts_2d[0, 1]], z=[energies[0]],
    mode="markers",
    marker=dict(size=14, color="red", symbol="circle",
                line=dict(width=2, color="white")),
    name="Current state",
    hovertemplate="Step 0<br>Energy = %{z:.1f}<extra></extra>",
))

# Animation frames — only update traces 1 & 2
frames = []
for t in range(T):
    frames.append(go.Frame(
        data=[
            go.Scatter3d(
                x=pts_2d[:t + 1, 0],
                y=pts_2d[:t + 1, 1],
                z=energies[:t + 1],
                mode="lines+markers",
                line=dict(width=4, color="gold"),
                marker=dict(size=3, color="gold"),
                name="Trajectory",
                showlegend=False,
            ),
            go.Scatter3d(
                x=[pts_2d[t, 0]],
                y=[pts_2d[t, 1]],
                z=[energies[t]],
                mode="markers",
                marker=dict(size=14, color="red", symbol="circle",
                            line=dict(width=2, color="white")),
                name="Current state",
                showlegend=False,
                hovertemplate=(
                    f"Step {t}<br>Energy = {energies[t]:.1f}<extra></extra>"
                ),
            ),
        ],
        traces=[1, 2],
        name=str(t),
    ))

fig3d.frames = frames

fig3d.update_layout(
    updatemenus=[dict(
        type="buttons", showactive=False,
        x=0.05, y=0, xanchor="right", yanchor="top",
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=200, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=50),
                )],
            ),
            dict(
                label="Pause",
                method="animate",
                args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode="immediate",
                )],
            ),
        ],
    )],
    sliders=[dict(
        active=0,
        steps=[
            dict(
                args=[[str(t)], dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                )],
                label=str(t),
                method="animate",
            )
            for t in range(T)
        ],
        x=0.1, len=0.9, xanchor="left",
        y=0, yanchor="top",
        currentvalue=dict(prefix="Step: ", visible=True, font=dict(size=14)),
    )],
    template="plotly_dark",
    scene=dict(
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        zaxis_title="Energy",
    ),
    height=650,
    margin=dict(l=0, r=0, t=30, b=80),
    legend=dict(font=dict(size=12)),
)

st.plotly_chart(fig3d, use_container_width=True)

st.info(
    "**How to read this plot:**\n\n"
    "- **Diamonds** are stored patterns sitting at energy minima (valleys).\n"
    "- The **gold trajectory** shows the network state evolving from the "
    "noisy input toward an attractor.\n"
    "- The **red sphere** marks the current state — watch it slide "
    "downhill as you press Play.\n"
    "- Energy **always decreases** (Lyapunov property), guaranteeing convergence.\n"
    "- If the trajectory reaches a stored pattern, recall succeeded; if it "
    "settles elsewhere, it found a **spurious attractor**."
)

st.markdown("---")

# ── Multi-probe trajectories (advanced) ──────────────────────────────────────

with st.expander("Multi-Probe Trajectories (advanced)", expanded=False):
    st.markdown(
        "Multiple noisy copies of **each** stored digit, all converging "
        "toward their respective basins of attraction.  This visualises "
        "how the energy landscape partitions the state space into basins.\n\n"
        "The axes are the same as above: **PC 1** and **PC 2** (the two "
        "PCA directions that best separate the stored patterns) on the "
        "horizontal plane, and **Energy** on the vertical axis.  "
        "Each colour represents a different stored digit class. Notice how "
        "trajectories of the same colour cluster together — they share "
        "a **basin of attraction**."
    )
    N_PER_DIGIT = 3
    fig_mp = go.Figure()
    fig_mp.add_trace(go.Scatter3d(
        x=tgt_2d[:, 0], y=tgt_2d[:, 1], z=tgt_E,
        mode="markers+text",
        marker=dict(
            size=10,
            color=[dcol[d] for d in dtup],
            symbol="diamond",
            line=dict(width=1, color="white"),
        ),
        text=[str(d) for d in dtup],
        textposition="top center",
        textfont=dict(size=14, color="white"),
        name="Stored patterns",
    ))
    rng2 = np.random.default_rng(int(seed) + 7)
    for i, d in enumerate(dtup):
        for j in range(N_PER_DIGIT):
            p = add_noise(ALL_REPS[d], noise, rng2)
            s, e = net.simulate(p, num_iter=max_iter, sync=sync)
            tr = (s if s.ndim == 2 else s[0]).T
            pr = pca.transform(tr)
            fig_mp.add_trace(go.Scatter3d(
                x=pr[:, 0], y=pr[:, 1], z=e,
                mode="lines+markers",
                line=dict(width=2, color=dcol[d]),
                marker=dict(size=2, color=dcol[d]),
                name=f"Digit {d}" if j == 0 else None,
                showlegend=(j == 0),
                legendgroup=str(d),
                opacity=0.7,
            ))
    fig_mp.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="PC 1", yaxis_title="PC 2", zaxis_title="Energy",
        ),
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_mp, use_container_width=True)
