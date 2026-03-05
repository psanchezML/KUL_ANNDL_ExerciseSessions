import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import requests
import os
import io
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras  # noqa: E402

st.set_page_config(layout="wide", page_title="Time Series Model Comparison Explorer")

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def prepare_timeseries(timeseries, lag):
    data = np.array([timeseries[i : i + lag] for i in range(len(timeseries) - lag)])
    targets = timeseries[lag:]
    return data, targets


def normalize(ts, params=None):
    if params is None:
        params = (np.mean(ts), np.std(ts))
    return (ts - params[0]) / params[1], params


def rescale(ts, params):
    return ts * params[1] + params[0]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_santa_fe():
    local_path = os.path.join(os.path.dirname(__file__), "SantaFe.npz")
    if not os.path.exists(local_path):
        url = "https://raw.githubusercontent.com/KULasagna/ANN_DL_public/master/session2/SantaFe.npz"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    data = np.load(local_path)
    key = list(data.keys())[0]
    series = data[key].flatten().astype(np.float64)
    train_val = series[:1000]
    test = series[1000:1100]
    return train_val, test, series[:1100]


def generate_mackey_glass(tau=30, n_points=2000, beta=0.2, gamma=0.1, n=10):
    x = np.zeros(n_points + tau + 500)
    x[:tau] = 0.9 + 0.2 * np.random.randn(tau)
    for t in range(tau, len(x) - 1):
        x[t + 1] = x[t] + beta * x[t - tau] / (1 + x[t - tau] ** n) - gamma * x[t]
    return x[500 + tau : 500 + tau + n_points]


@st.cache_data
def load_mackey_glass(tau):
    series = generate_mackey_glass(tau=tau)
    split = int(len(series) * 0.8)
    train_val = series[:split]
    test = series[split : split + 100]
    return train_val, test, series[: split + 100]


@st.cache_data
def load_sunspot():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
    df = pd.read_csv(url, header=0, names=["Month", "Sunspots"])
    series = df["Sunspots"].values.astype(np.float64)
    train_val = series[:2800]
    test = series[2800:2900]
    return train_val, test, series[:2900]


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def compute_acf(x, max_lag=100):
    x = x - np.mean(x)
    n = len(x)
    result = np.correlate(x, x, mode="full")
    result = result[n - 1 :]
    result = result / result[0]
    return result[: max_lag + 1]


def compute_pacf(acf_vals, max_lag=100):
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0
    if max_lag == 0:
        return pacf
    pacf[1] = acf_vals[1]
    phi = np.zeros((max_lag + 1, max_lag + 1))
    phi[1, 1] = acf_vals[1]
    for k in range(2, max_lag + 1):
        num = acf_vals[k] - sum(phi[k - 1, j] * acf_vals[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * acf_vals[j] for j in range(1, k))
        if abs(den) < 1e-12:
            break
        phi[k, k] = num / den
        pacf[k] = phi[k, k]
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
    return pacf


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_mlp(lag, hidden_units=32, lr=0.001):
    model = keras.Sequential([
        keras.layers.Dense(hidden_units, activation="tanh", input_shape=(lag,)),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def build_rnn(lag, hidden_units=32, lr=0.001):
    model = keras.Sequential([
        keras.layers.SimpleRNN(hidden_units, activation="tanh", input_shape=(lag, 1)),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def build_lstm(lag, hidden_units=32, lr=0.001):
    model = keras.Sequential([
        keras.layers.LSTM(hidden_units, input_shape=(lag, 1)),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


# ---------------------------------------------------------------------------
# Iterative (recursive) prediction
# ---------------------------------------------------------------------------

def iterative_predict(model, seed_window, n_steps, model_type, norm_params):
    """Feed predictions back as input to forecast n_steps ahead."""
    preds = []
    current = seed_window.copy()
    lag = len(current)
    for _ in range(n_steps):
        if model_type == "MLP":
            inp = current.reshape(1, lag)
        else:
            inp = current.reshape(1, lag, 1)
        p = model.predict(inp, verbose=0)[0, 0]
        preds.append(p)
        current = np.append(current[1:], p)
    return np.array(preds)


# ---------------------------------------------------------------------------
# Training helper with progress
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_train, X_val, y_val, epochs, progress_bar, label):
    history = {"loss": [], "val_loss": []}
    for epoch in range(epochs):
        h = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=1, batch_size=32, verbose=0)
        history["loss"].append(h.history["loss"][0])
        history["val_loss"].append(h.history["val_loss"][0])
        progress_bar.progress((epoch + 1) / epochs, text=f"{label}: epoch {epoch+1}/{epochs}")
    return history


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Configuration")

dataset_name = st.sidebar.radio("Dataset", ["Santa Fe Laser", "Mackey-Glass", "Sunspot Numbers"])

mg_tau = 30
if dataset_name == "Mackey-Glass":
    mg_tau = st.sidebar.slider("Mackey-Glass τ (tau)", 10, 50, 30)

lag = st.sidebar.slider("Lag (window size)", 1, 50, 10)
val_ratio = st.sidebar.slider("Validation split ratio", 0.05, 0.4, 0.2, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("MLP")
mlp_hidden = st.sidebar.slider("MLP hidden units", 8, 128, 32, key="mlp_h")
mlp_epochs = st.sidebar.slider("MLP epochs", 5, 200, 50, key="mlp_e")
mlp_lr = st.sidebar.select_slider("MLP learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001, key="mlp_lr")

st.sidebar.subheader("Simple RNN")
rnn_hidden = st.sidebar.slider("RNN hidden units", 8, 128, 32, key="rnn_h")
rnn_epochs = st.sidebar.slider("RNN epochs", 5, 200, 50, key="rnn_e")
rnn_lr = st.sidebar.select_slider("RNN learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001, key="rnn_lr")

st.sidebar.subheader("LSTM")
lstm_hidden = st.sidebar.slider("LSTM hidden units", 8, 128, 32, key="lstm_h")
lstm_epochs = st.sidebar.slider("LSTM epochs", 5, 200, 50, key="lstm_e")
lstm_lr = st.sidebar.select_slider("LSTM learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001, key="lstm_lr")

st.sidebar.markdown("---")
train_button = st.sidebar.button("🚀 Train All Models", use_container_width=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def get_dataset(name, tau):
    if name == "Santa Fe Laser":
        return load_santa_fe()
    elif name == "Mackey-Glass":
        return load_mackey_glass(tau)
    else:
        return load_sunspot()


train_val, test_data, full_series = get_dataset(dataset_name, mg_tau)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Time Series Model Comparison Explorer")
st.caption("Compare MLP, Simple RNN, and LSTM on time series prediction tasks")

tab_eda, tab_train, tab_deep, tab_bias = st.tabs([
    "📊 Exploratory Data Analysis",
    "🏋️ Model Training & Comparison",
    "🔬 Model Deep Dive",
    "🧠 Inductive Bias Explainer",
])

# ===== TAB 1: EDA =====
with tab_eda:
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("Basic Statistics")
        stats = {
            "Length": len(full_series),
            "Mean": f"{np.mean(full_series):.4f}",
            "Std": f"{np.std(full_series):.4f}",
            "Min": f"{np.min(full_series):.4f}",
            "Max": f"{np.max(full_series):.4f}",
            "Median": f"{np.median(full_series):.4f}",
        }
        st.dataframe(pd.DataFrame(stats.items(), columns=["Metric", "Value"]), hide_index=True, use_container_width=True)

    with col1:
        st.subheader("Raw Time Series")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(
            y=train_val, mode="lines", name="Train/Val",
            line=dict(color="#636EFA"),
        ))
        fig_raw.add_trace(go.Scatter(
            x=np.arange(len(train_val), len(train_val) + len(test_data)),
            y=test_data, mode="lines", name="Test",
            line=dict(color="#EF553B"),
        ))
        fig_raw.update_layout(
            xaxis_title="Time step", yaxis_title="Value",
            template="plotly_white", height=350, margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    # Rolling statistics
    st.subheader("Rolling Mean & Standard Deviation (window=50)")
    window = 50
    rolling_mean = pd.Series(full_series).rolling(window).mean().values
    rolling_std = pd.Series(full_series).rolling(window).std().values

    fig_roll = make_subplots(rows=1, cols=2, subplot_titles=["Rolling Mean", "Rolling Std"])
    fig_roll.add_trace(go.Scatter(y=full_series, mode="lines", name="Series", opacity=0.3, line=dict(color="#636EFA")), row=1, col=1)
    fig_roll.add_trace(go.Scatter(y=rolling_mean, mode="lines", name="Rolling Mean", line=dict(color="#FF6692", width=2)), row=1, col=1)
    fig_roll.add_trace(go.Scatter(y=full_series, mode="lines", name="Series", opacity=0.3, showlegend=False, line=dict(color="#636EFA")), row=1, col=2)
    fig_roll.add_trace(go.Scatter(y=rolling_std, mode="lines", name="Rolling Std", line=dict(color="#00CC96", width=2)), row=1, col=2)
    fig_roll.update_layout(template="plotly_white", height=300, margin=dict(t=40, b=30))
    st.plotly_chart(fig_roll, use_container_width=True)

    # ACF / PACF
    st.subheader("Autocorrelation & Partial Autocorrelation (100 lags)")
    acf_vals = compute_acf(full_series, max_lag=100)
    pacf_vals = compute_pacf(acf_vals, max_lag=100)
    confidence = 1.96 / np.sqrt(len(full_series))

    fig_acf, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    axes[0].bar(range(len(acf_vals)), acf_vals, width=0.4, color="#636EFA")
    axes[0].axhline(confidence, ls="--", color="red", lw=0.8)
    axes[0].axhline(-confidence, ls="--", color="red", lw=0.8)
    axes[0].set_title("ACF")
    axes[0].set_xlabel("Lag")
    axes[1].bar(range(len(pacf_vals)), pacf_vals, width=0.4, color="#00CC96")
    axes[1].axhline(confidence, ls="--", color="red", lw=0.8)
    axes[1].axhline(-confidence, ls="--", color="red", lw=0.8)
    axes[1].set_title("PACF")
    axes[1].set_xlabel("Lag")
    plt.tight_layout()
    st.pyplot(fig_acf)
    plt.close(fig_acf)

    # Periodogram
    st.subheader("Spectral Density (Periodogram)")
    from scipy.signal import periodogram as scipy_periodogram
    freqs, psd = scipy_periodogram(full_series, fs=1.0)
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(x=freqs[1:], y=psd[1:], mode="lines", line=dict(color="#AB63FA")))
    fig_psd.update_layout(
        xaxis_title="Frequency", yaxis_title="Power Spectral Density",
        yaxis_type="log", template="plotly_white", height=300, margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_psd, use_container_width=True)

    # Automated insights
    st.subheader("Automated Insights")
    dominant_freq = freqs[1:][np.argmax(psd[1:])]
    dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else np.inf
    acf_first_zero = next((i for i in range(1, len(acf_vals)) if acf_vals[i] < 0), len(acf_vals))

    insights = []
    insights.append(f"**Series length**: {len(full_series)} samples.")
    insights.append(f"**Dominant spectral period**: ~{dominant_period:.1f} time steps (frequency {dominant_freq:.4f}).")
    insights.append(f"**ACF first zero crossing** at lag ~{acf_first_zero}, suggesting short-range correlations decay over ~{acf_first_zero} steps.")

    if dataset_name == "Santa Fe Laser":
        insights.append("The Santa Fe Laser series shows quasi-periodic chaotic dynamics with strong local structure. "
                        "An MLP with a fixed window can exploit this local structure effectively.")
    elif dataset_name == "Mackey-Glass":
        insights.append(f"The Mackey-Glass series (τ={mg_tau}) is a chaotic system with long-range temporal dependencies. "
                        "LSTM networks with gating mechanisms are well-suited to capture these dependencies.")
    else:
        insights.append("The Sunspot series exhibits a clear ~11-year (132-month) cycle with amplitude variations. "
                        "Sequential models like RNNs can naturally process the periodic structure.")

    for ins in insights:
        st.markdown(f"- {ins}")


# ===== TAB 2: Model Training & Comparison =====
with tab_train:
    st.header("Model Training & Comparison")

    if not train_button:
        st.info("Configure parameters in the sidebar and click **Train All Models** to begin.")
    else:
        ts_norm, norm_params = normalize(train_val)
        test_norm = (test_data - norm_params[0]) / norm_params[1]

        X_all, y_all = prepare_timeseries(ts_norm, lag)
        n_val = max(1, int(len(X_all) * val_ratio))
        X_train, X_val = X_all[:-n_val], X_all[-n_val:]
        y_train, y_val = y_all[:-n_val], y_all[-n_val:]

        X_train_rnn = X_train.reshape(X_train.shape[0], lag, 1)
        X_val_rnn = X_val.reshape(X_val.shape[0], lag, 1)

        histories = {}
        models = {}
        preds_dict = {}
        metrics = {}

        st.subheader("Training Progress")
        prog_mlp = st.progress(0, text="MLP: waiting...")
        prog_rnn = st.progress(0, text="RNN: waiting...")
        prog_lstm = st.progress(0, text="LSTM: waiting...")

        # Train MLP
        mlp_model = build_mlp(lag, mlp_hidden, mlp_lr)
        histories["MLP"] = train_model(mlp_model, X_train, y_train, X_val, y_val, mlp_epochs, prog_mlp, "MLP")
        models["MLP"] = mlp_model

        # Train RNN
        rnn_model = build_rnn(lag, rnn_hidden, rnn_lr)
        histories["RNN"] = train_model(rnn_model, X_train_rnn, y_train, X_val_rnn, y_val, rnn_epochs, prog_rnn, "RNN")
        models["RNN"] = rnn_model

        # Train LSTM
        lstm_model = build_lstm(lag, lstm_hidden, lstm_lr)
        histories["LSTM"] = train_model(lstm_model, X_train_rnn, y_train, X_val_rnn, y_val, lstm_epochs, prog_lstm, "LSTM")
        models["LSTM"] = lstm_model

        # Iterative prediction on test set
        seed_window = ts_norm[-lag:]
        n_test = len(test_norm)

        for name, mdl in models.items():
            preds_norm = iterative_predict(mdl, seed_window, n_test, name, norm_params)
            preds_orig = rescale(preds_norm, norm_params)
            preds_dict[name] = preds_orig
            mse = np.mean((preds_orig - test_data) ** 2)
            mae = np.mean(np.abs(preds_orig - test_data))
            metrics[name] = {"MSE": f"{mse:.6f}", "MAE": f"{mae:.6f}"}

        st.success("Training complete!")

        # Loss curves
        st.subheader("Training Loss Curves")
        fig_loss = go.Figure()
        colors = {"MLP": "#636EFA", "RNN": "#EF553B", "LSTM": "#00CC96"}
        for name, hist in histories.items():
            fig_loss.add_trace(go.Scatter(y=hist["loss"], mode="lines", name=f"{name} train", line=dict(color=colors[name])))
            fig_loss.add_trace(go.Scatter(y=hist["val_loss"], mode="lines", name=f"{name} val", line=dict(color=colors[name], dash="dash")))
        fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="MSE Loss", yaxis_type="log", template="plotly_white", height=400)
        st.plotly_chart(fig_loss, use_container_width=True)

        # Predictions vs actual
        st.subheader("Test Set: Predictions vs Actual")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_data, mode="lines", name="Actual", line=dict(color="black", width=2)))
        for name, pred in preds_dict.items():
            fig_pred.add_trace(go.Scatter(y=pred, mode="lines", name=name, line=dict(color=colors[name])))
        fig_pred.update_layout(xaxis_title="Time step", yaxis_title="Value", template="plotly_white", height=400)
        st.plotly_chart(fig_pred, use_container_width=True)

        # Metrics table
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.index.name = "Model"
        st.dataframe(metrics_df, use_container_width=True)

        # Residuals
        st.subheader("Residual Plots")
        fig_res = make_subplots(rows=1, cols=3, subplot_titles=list(preds_dict.keys()))
        for idx, (name, pred) in enumerate(preds_dict.items()):
            residuals = test_data - pred
            fig_res.add_trace(go.Scatter(y=residuals, mode="lines+markers", marker=dict(size=3),
                                         name=name, line=dict(color=colors[name])), row=1, col=idx + 1)
            fig_res.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=idx + 1)
        fig_res.update_layout(template="plotly_white", height=300, showlegend=False, margin=dict(t=40, b=30))
        st.plotly_chart(fig_res, use_container_width=True)

        # Error distribution
        st.subheader("Error Distribution")
        fig_hist, axes_h = plt.subplots(1, 3, figsize=(14, 3.5))
        for idx, (name, pred) in enumerate(preds_dict.items()):
            residuals = test_data - pred
            axes_h[idx].hist(residuals, bins=20, color=colors[name], alpha=0.7, edgecolor="white")
            axes_h[idx].set_title(f"{name} Residuals")
            axes_h[idx].set_xlabel("Error")
            axes_h[idx].axvline(0, color="black", ls="--", lw=0.8)
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        # Store results in session state
        st.session_state["histories"] = histories
        st.session_state["models"] = models
        st.session_state["preds_dict"] = preds_dict
        st.session_state["metrics"] = metrics
        st.session_state["norm_params"] = norm_params
        st.session_state["ts_norm"] = ts_norm
        st.session_state["test_norm"] = test_norm
        st.session_state["trained"] = True


# ===== TAB 3: Model Deep Dive =====
with tab_deep:
    st.header("Model Deep Dive")

    if "trained" not in st.session_state or not st.session_state["trained"]:
        st.info("Train models first (Tab 2) to unlock the deep dive analysis.")
    else:
        stored_models = st.session_state["models"]
        stored_preds = st.session_state["preds_dict"]
        stored_nparams = st.session_state["norm_params"]
        stored_ts_norm = st.session_state["ts_norm"]

        selected_model = st.selectbox("Select model", list(stored_models.keys()))

        # Architecture info
        st.subheader(f"{selected_model} Architecture")
        model_obj = stored_models[selected_model]
        arch_lines = []
        model_obj.summary(print_fn=lambda x: arch_lines.append(x))
        st.code("\n".join(arch_lines), language=None)

        total_params = model_obj.count_params()
        st.metric("Total Parameters", f"{total_params:,}")

        # Lag sensitivity
        st.subheader("Lag Sensitivity Analysis")
        st.caption("MSE on test set for different lag values (re-trains a lightweight version)")

        lag_values = [5, 10, 15, 20, 30, 40, 50]
        lag_values = [l for l in lag_values if l < len(train_val) - 10]

        if st.button("Run Lag Sensitivity Analysis", key="lag_sens"):
            lag_mses = []
            lag_prog = st.progress(0, text="Running lag sensitivity...")
            for i, test_lag in enumerate(lag_values):
                ts_n, npar = normalize(train_val)
                t_n = (test_data - npar[0]) / npar[1]
                Xa, ya = prepare_timeseries(ts_n, test_lag)
                n_v = max(1, int(len(Xa) * val_ratio))
                Xt, Xv = Xa[:-n_v], Xa[-n_v:]
                yt, yv = ya[:-n_v], ya[-n_v:]

                if selected_model == "MLP":
                    m = build_mlp(test_lag, mlp_hidden, mlp_lr)
                    m.fit(Xt, yt, validation_data=(Xv, yv), epochs=30, batch_size=32, verbose=0)
                elif selected_model == "RNN":
                    m = build_rnn(test_lag, rnn_hidden, rnn_lr)
                    m.fit(Xt.reshape(-1, test_lag, 1), yt,
                          validation_data=(Xv.reshape(-1, test_lag, 1), yv),
                          epochs=30, batch_size=32, verbose=0)
                else:
                    m = build_lstm(test_lag, lstm_hidden, lstm_lr)
                    m.fit(Xt.reshape(-1, test_lag, 1), yt,
                          validation_data=(Xv.reshape(-1, test_lag, 1), yv),
                          epochs=30, batch_size=32, verbose=0)

                seed = ts_n[-test_lag:]
                pn = iterative_predict(m, seed, len(test_data), selected_model, npar)
                po = rescale(pn, npar)
                lag_mses.append(np.mean((po - test_data) ** 2))
                lag_prog.progress((i + 1) / len(lag_values), text=f"Lag {test_lag} done")

            fig_lag = go.Figure()
            fig_lag.add_trace(go.Scatter(x=lag_values, y=lag_mses, mode="lines+markers",
                                         marker=dict(size=8), line=dict(color="#636EFA", width=2)))
            fig_lag.update_layout(xaxis_title="Lag", yaxis_title="MSE", template="plotly_white", height=350)
            st.plotly_chart(fig_lag, use_container_width=True)

        # Forecast horizon analysis
        st.subheader("Forecast Horizon Analysis")
        st.caption("How prediction error grows with forecast horizon")
        if selected_model in stored_preds:
            pred = stored_preds[selected_model]
            horizons = list(range(1, len(pred) + 1))
            cumul_mse = [np.mean((pred[:h] - test_data[:h]) ** 2) for h in horizons]

            fig_hor = go.Figure()
            fig_hor.add_trace(go.Scatter(x=horizons, y=cumul_mse, mode="lines",
                                          line=dict(color="#EF553B", width=2)))
            fig_hor.update_layout(xaxis_title="Forecast Horizon (steps)", yaxis_title="Cumulative MSE",
                                  template="plotly_white", height=350)
            st.plotly_chart(fig_hor, use_container_width=True)


# ===== TAB 4: Inductive Bias Explainer =====
with tab_bias:
    st.header("Inductive Bias Explainer")

    bias_info = {
        "Santa Fe Laser": {
            "winner": "MLP",
            "explanation": (
                "### Why MLP wins on the Santa Fe Laser dataset\n\n"
                "The Santa Fe Laser series originates from a far-infrared NH₃ laser operating in a "
                "chaotic regime. Despite being chaotic, the series has **strong local structure**: "
                "the next value depends heavily on a finite, fixed-size window of recent values.\n\n"
                "**MLP's advantage**: A feedforward network with a fixed-size input window is a natural "
                "fit. It directly maps a lag vector to a prediction without the overhead of sequential "
                "processing. The function mapping recent inputs to the next value is smooth and well-captured "
                "by dense layers with tanh activations.\n\n"
                "**Why not RNN/LSTM?** These models introduce recurrent connections that are useful for "
                "variable-length or very long dependencies, but the Santa Fe series doesn't require them. "
                "The extra parameters and more complex optimization landscape can actually hurt performance."
            ),
        },
        "Sunspot Numbers": {
            "winner": "RNN",
            "explanation": (
                "### Why Simple RNN works well on Sunspot data\n\n"
                "Sunspot numbers exhibit a clear **~11-year (132-month) cycle** with gradual amplitude "
                "modulation. The temporal structure is periodic and sequential in nature.\n\n"
                "**RNN's advantage**: The recurrent hidden state naturally accumulates information about "
                "the current phase within the solar cycle. The moderate-range dependencies (~132 steps) "
                "are within the effective memory range of simple RNNs, especially when combined with "
                "a reasonable lag window.\n\n"
                "**Why not MLP?** A fixed window may not always capture the full cycle context. "
                "An MLP requires the lag to be at least as large as the relevant dependency range.\n\n"
                "**Why not LSTM?** The gating mechanism of LSTMs adds complexity that isn't necessary "
                "for the relatively simple periodic structure. The vanishing gradient problem that LSTMs "
                "solve isn't severe here because the dependencies are moderate, not extremely long-range."
            ),
        },
        "Mackey-Glass": {
            "winner": "LSTM",
            "explanation": (
                f"### Why LSTM excels on Mackey-Glass (τ={mg_tau})\n\n"
                "The Mackey-Glass equation is a delay differential equation that produces chaotic dynamics "
                f"when τ is large enough (τ={mg_tau}). The key feature is that the current state depends on "
                f"the state **{mg_tau} time steps in the past**, creating long-range temporal dependencies.\n\n"
                "**LSTM's advantage**: The gating mechanism (forget gate, input gate, output gate) allows "
                "the network to selectively remember or forget information over long time spans. The cell "
                f"state can carry information from {mg_tau} steps ago to the present, which is precisely what "
                "the Mackey-Glass dynamics require.\n\n"
                "**Why not MLP?** While an MLP with lag ≥ τ could theoretically capture the dependency, "
                "it must learn the complex nonlinear mapping from a large input vector, which is hard.\n\n"
                "**Why not Simple RNN?** Simple RNNs suffer from the **vanishing gradient problem**, making "
                f"it difficult to propagate gradient information across {mg_tau} time steps during training. "
                "The LSTM's gating mechanism directly addresses this limitation."
            ),
        },
    }

    info = bias_info[dataset_name]
    st.markdown(info["explanation"])

    # Performance comparison chart
    if "metrics" in st.session_state:
        st.subheader("Model Performance Comparison")
        met = st.session_state["metrics"]
        model_names = list(met.keys())
        mse_vals = [float(met[m]["MSE"]) for m in model_names]
        mae_vals = [float(met[m]["MAE"]) for m in model_names]

        colors_bar = ["#636EFA", "#EF553B", "#00CC96"]
        highlight = [1.0 if m == info["winner"] else 0.5 for m in model_names]

        fig_comp = make_subplots(rows=1, cols=2, subplot_titles=["MSE Comparison", "MAE Comparison"])
        fig_comp.add_trace(go.Bar(
            x=model_names, y=mse_vals,
            marker=dict(color=colors_bar, opacity=highlight),
            text=[f"{v:.6f}" for v in mse_vals], textposition="auto",
        ), row=1, col=1)
        fig_comp.add_trace(go.Bar(
            x=model_names, y=mae_vals,
            marker=dict(color=colors_bar, opacity=highlight),
            text=[f"{v:.6f}" for v in mae_vals], textposition="auto",
        ), row=1, col=2)
        fig_comp.update_layout(template="plotly_white", height=400, showlegend=False, margin=dict(t=50, b=30))
        st.plotly_chart(fig_comp, use_container_width=True)

        best_model = model_names[np.argmin(mse_vals)]
        expected = info["winner"]
        if best_model == expected:
            st.success(f"**Result matches theory!** {best_model} achieves the lowest MSE, as expected for the {dataset_name} dataset.")
        else:
            st.warning(
                f"**Interesting result:** {best_model} achieved the lowest MSE, while theory suggests "
                f"{expected} should perform best on {dataset_name}. This can happen due to hyperparameter "
                "choices, limited training, or the specific lag value. Try adjusting parameters!"
            )
    else:
        st.info("Train models (Tab 2) to see the performance comparison chart here.")

    # Theoretical summary
    st.subheader("Quick Reference: Model Inductive Biases")
    bias_df = pd.DataFrame({
        "Model": ["MLP", "Simple RNN", "LSTM"],
        "Architecture": [
            "Feedforward, fixed-size input",
            "Recurrent hidden state, sequential",
            "Recurrent with gates (forget, input, output)",
        ],
        "Memory Type": [
            "Fixed window (lag size)",
            "Decaying short-to-medium range",
            "Selective long-range via cell state",
        ],
        "Best For": [
            "Strong local structure, fixed dependencies",
            "Moderate sequential/periodic patterns",
            "Long-range dependencies, chaotic systems",
        ],
        "Weakness": [
            "Cannot see beyond the window",
            "Vanishing gradients for long sequences",
            "More parameters, slower to train",
        ],
    })
    st.dataframe(bias_df, hide_index=True, use_container_width=True)
