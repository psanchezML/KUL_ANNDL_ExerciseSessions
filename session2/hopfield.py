"""
Enhanced Hopfield Network implementation with visualization and analysis tools.

Provides a comprehensive toolkit for studying Hopfield networks as associative
memories, including energy analysis, basin-of-attraction mapping, capacity
testing, and rich visualizations suitable for educational exploration.

References
----------
[Hebb]       D. O. Hebb, "The Organization of Behavior", 1949.
[LSSM]       J. Li, A. N. Michel, W. Porod, "Analysis and Synthesis of a Class
             of Neural Networks: Linear Systems Operating on a Closed
             Hypercube", IEEE Trans. Circuits Syst., 1989.
[Hopfield82] J. J. Hopfield, "Neural networks and physical systems with
             emergent collective computational abilities", PNAS, 1982.
"""

import numpy as np
from dataclasses import dataclass
from itertools import product


@dataclass
class NormalizationParameters:
    shape: tuple[int]
    m: float
    M: float


class HopfieldNetwork:
    """
    Hopfield network for associative memory with analysis and visualization.

    A Hopfield network stores patterns as attractors of a dynamical system.
    Given a (possibly corrupted) input the network evolves toward the nearest
    stored pattern.

    Parameters
    ----------
    targets : array_like
        Matrix (T, D) -- T target patterns each of dimension D.
    alg : {'LSSM', 'Hebb'}, default 'LSSM'
        Learning algorithm.  'Hebb' uses Hebbian learning; 'LSSM' uses the
        Linear Saturating System Model (MATLAB ``newhop`` equivalent).

    Examples
    --------
    >>> net = HopfieldNetwork([[1, 1, -1], [-1, -1, 1]])
    >>> net.summary()
    >>> states, energies = net.simulate([0.5, -0.3, 0.8], num_iter=10)
    >>> net.plot_energy_over_time(energies)
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, targets, alg='LSSM'):
        self.D = 0
        self.W = 0
        self.b = 0
        self.rng = np.random.default_rng()
        self.alg = alg
        self._targets = np.atleast_2d(np.asarray(targets, dtype=float))
        self._set_weights(self._targets.T, alg)
        self.tf = {
            'Hebb': np.sign,
            'LSSM': lambda x: np.clip(x, -1, 1),
        }.get(alg)
        self.int_inv_tf = {
            'Hebb': np.zeros_like,
            'LSSM': lambda x: x * x / 2,
        }.get(alg)

    def _set_weights(self, targets, alg):
        """Compute weight matrix and bias from target patterns."""
        self.D, T = targets.shape
        if alg == 'Hebb':
            rho = 0
            W = np.dot(targets - rho, targets.T - rho)
            np.fill_diagonal(W, 0)
            W /= T * self.D
            self.W = W
            self.b = np.zeros(self.D)
        elif alg == 'LSSM':
            Y = targets[:, :-1] - targets[:, [-1]]
            U, S, _ = np.linalg.svd(Y, full_matrices=True)
            Sigma = np.diag(S)
            k = np.linalg.matrix_rank(Sigma)
            Tp = np.dot(U[:, :k], U[:, :k].T)
            Tm = np.dot(U[:, k:], U[:, k:].T)
            tau = 10
            Ttau = Tp - tau * Tm
            Itau = targets[:, -1] - Ttau @ targets[:, -1]
            h = 0.15
            Dphi = np.concatenate([
                np.exp(h) * np.ones(k),
                np.exp(-tau * h) * np.ones(self.D - k),
            ])
            Dgamma = Dphi - 1
            Dgamma[k:] /= -tau
            self.W = U @ np.diag(Dphi) @ U.T
            self.b = U @ np.diag(Dgamma) @ U.T @ Itau
        else:
            raise ValueError(
                "Unknown learning algorithm. Supported: 'Hebb', 'LSSM'."
            )
        self.b = np.expand_dims(self.b, axis=0)

    # ------------------------------------------------------------------ #
    #  Core simulation                                                    #
    # ------------------------------------------------------------------ #

    def simulate(self, data, num_iter=20, sync=True):
        """
        Simulate the state evolution of the Hopfield network.

        Parameters
        ----------
        data : array_like
            Initial state(s).  Vector (D,) for one state or matrix (P, D).
        num_iter : int
            Number of update steps.
        sync : bool
            ``True`` for synchronous update, ``False`` for asynchronous.

        Returns
        -------
        states : ndarray
            All visited states -- shape (D, num_iter+1) for a single input,
            (P, D, num_iter+1) for multiple inputs.
        energies : ndarray
            Energies at each step -- shape (num_iter+1,) or (P, num_iter+1).
        """
        data = np.atleast_2d(np.asarray(data, dtype=float))
        states = np.empty((*data.shape, num_iter + 1))
        energies = np.empty((data.shape[0], num_iter + 1))

        s = np.copy(data)
        states[:, :, 0] = s
        energies[:, 0] = self.energy(s)
        for t in range(num_iter):
            s = self._step(s, sync)
            states[:, :, t + 1] = s
            energies[:, t + 1] = self.energy(s)
        return np.squeeze(states), np.squeeze(energies)

    def _step(self, s, sync):
        if sync:
            s = self.tf(s @ self.W + self.b)
        else:
            for d in self.rng.permutation(self.D):
                s[:, d] = self.tf(s @ self.W[:, d] + self.b[0, d])
        return s

    def _evolve(self, data, num_iter=50, sync=True):
        """Evolve states to their final configuration (no history stored)."""
        data = np.atleast_2d(np.asarray(data, dtype=float))
        s = np.copy(data)
        for _ in range(num_iter):
            s = self._step(s, sync)
        return s

    def energy(self, s):
        """Calculate the energy of the given state(s)."""
        s = np.atleast_2d(s)
        e = (
            -0.5 * np.sum((s @ self.W) * s, axis=1, keepdims=True)
            - s @ self.b.T
        )
        e += np.sum(self.int_inv_tf(s), axis=1, keepdims=True)
        return np.squeeze(e)

    # ------------------------------------------------------------------ #
    #  Static utilities                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize(data):
        """
        Normalize data to bipolar {-1, +1} range.

        Returns
        -------
        normalized : ndarray  (P, D)
        params : NormalizationParameters
        """
        data = np.asarray(data, dtype=float)
        P, *shape = data.shape
        reshaped = data.reshape((P, -1))
        m = np.min(reshaped)
        M = np.max(reshaped)
        normalized = np.sign(2 * (reshaped - m) / (M - m) - 1)
        return normalized, NormalizationParameters(shape, m, M)

    @staticmethod
    def rescale(normalized, params):
        """Rescale normalized data back to original range."""
        normalized = np.asarray(normalized, dtype=float)
        rescaled = (normalized + 1) * (params.M - params.m) / 2 + params.m
        return rescaled.reshape([-1, *params.shape])

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def targets(self):
        """Stored target patterns, shape (T, D)."""
        return self._targets

    @property
    def num_patterns(self):
        """Number of stored patterns."""
        return self._targets.shape[0]

    @property
    def theoretical_capacity(self):
        """Approximate Hebbian capacity  ~0.14 * D."""
        return int(0.14 * self.D)

    # ------------------------------------------------------------------ #
    #  Summary                                                            #
    # ------------------------------------------------------------------ #

    def summary(self):
        """Print a concise summary of the network configuration."""
        T = self.num_patterns
        cap = self.theoretical_capacity
        stable = sum(1 for p in self._targets if self.is_stable(p))
        print("=" * 54)
        print("  Hopfield Network Summary")
        print("=" * 54)
        print(f"  Dimension (D):            {self.D}")
        print(f"  Stored patterns (T):      {T}")
        print(f"  Algorithm:                {self.alg}")
        print(f"  Theoretical capacity:     ~{cap}  (0.14 x {self.D})")
        print(f"  Stable stored patterns:   {stable}/{T}")
        print(f"  Weight matrix norm:       {np.linalg.norm(self.W):.4f}")
        if T > cap and self.alg == 'Hebb':
            print(f"  WARNING: over capacity ({T} > {cap})")
        print("=" * 54)

    # ------------------------------------------------------------------ #
    #  Analysis                                                           #
    # ------------------------------------------------------------------ #

    def is_stable(self, pattern):
        """Return ``True`` if *pattern* is a fixed point (synchronous update)."""
        p = np.atleast_2d(np.asarray(pattern, dtype=float))
        updated = self.tf(p @ self.W + self.b)
        return bool(np.allclose(p, updated, atol=1e-6))

    def classify_fixed_points(self):
        """
        Discover all stable fixed points by enumeration (D <= 20) or sampling.

        Returns
        -------
        attractors : ndarray (K, D)
        """
        if self.D <= 20:
            candidates = np.array(list(product([-1.0, 1.0], repeat=self.D)))
        else:
            candidates = self.rng.choice(
                [-1.0, 1.0], size=(min(20000, 2 ** self.D), self.D)
            )
        attractors = []
        for c in candidates:
            if self.is_stable(c):
                if not any(np.allclose(c, a) for a in attractors):
                    attractors.append(c.copy())
        return np.array(attractors) if attractors else np.empty((0, self.D))

    def find_spurious_attractors(self):
        """
        Return attractors that are neither a stored pattern nor its negation.
        """
        attractors = self.classify_fixed_points()
        spurious = []
        for a in attractors:
            is_stored = any(np.allclose(a, t, atol=0.5) for t in self._targets)
            is_neg = any(np.allclose(a, -t, atol=0.5) for t in self._targets)
            if not is_stored and not is_neg:
                spurious.append(a)
        return np.array(spurious) if spurious else np.empty((0, self.D))

    def compute_basins(self, num_iter=50, initial=None, num_samples=None,
                       sync=True):
        """
        Map starting states to the attractors they converge to.

        Parameters
        ----------
        num_iter : int
            Convergence iterations.
        initial : ndarray, optional
            Custom initial states (N, D).  If *None*, binary states are
            enumerated (D <= 15) or sampled.
        num_samples : int, optional
            Number of random binary starting states when *initial* is None
            and D > 15.
        sync : bool
            Update mode.

        Returns
        -------
        initial : ndarray (N, D)
        labels : ndarray (N,)        -- index into *attractors*
        attractors : ndarray (K, D)
        fractions : ndarray (K,)     -- basin size fractions
        """
        if initial is None:
            if num_samples is None and self.D <= 15:
                initial = np.array(
                    list(product([-1.0, 1.0], repeat=self.D))
                )
            else:
                n = num_samples or min(5000, 2 ** self.D)
                initial = self.rng.choice([-1.0, 1.0], size=(n, self.D))

        final = self._evolve(initial, num_iter=num_iter, sync=sync)

        attractors = []
        labels = np.zeros(len(initial), dtype=int)
        for i, fs in enumerate(final):
            matched = False
            for j, a in enumerate(attractors):
                if np.allclose(fs, a, atol=0.5):
                    labels[i] = j
                    matched = True
                    break
            if not matched:
                labels[i] = len(attractors)
                attractors.append(fs.copy())

        attractors = np.array(attractors)
        fractions = np.array(
            [np.mean(labels == j) for j in range(len(attractors))]
        )
        return initial, labels, attractors, fractions

    @staticmethod
    def capacity_test(dim, max_patterns, trials=30, alg='Hebb', num_iter=30):
        """
        Measure retrieval success as number of stored patterns grows.

        Returns
        -------
        counts : ndarray   -- numbers of patterns tested
        rates  : ndarray   -- fraction of patterns successfully retrieved
        """
        counts = np.arange(1, max_patterns + 1)
        rates = np.zeros(len(counts))
        rng = np.random.default_rng(42)

        for idx, T in enumerate(counts):
            ok = 0
            total = T * trials
            for _ in range(trials):
                tgts = rng.choice([-1.0, 1.0], size=(T, dim))
                try:
                    net = HopfieldNetwork(tgts, alg=alg)
                    final = net._evolve(tgts, num_iter=num_iter)
                    for ti in range(T):
                        if np.allclose(np.sign(final[ti]), tgts[ti], atol=0.5):
                            ok += 1
                except Exception:
                    pass
            rates[idx] = ok / total if total > 0 else 0.0
        return counts, rates

    # ------------------------------------------------------------------ #
    #  Visualization                                                      #
    # ------------------------------------------------------------------ #

    def plot_weight_matrix(self, ax=None, cmap='RdBu_r'):
        """Display the weight matrix as a heatmap."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure
        vmax = np.max(np.abs(self.W)) or 1.0
        im = ax.imshow(self.W, cmap=cmap, aspect='equal',
                       vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Weight Matrix  W')
        ax.set_xlabel('Neuron j')
        ax.set_ylabel('Neuron i')
        plt.tight_layout()
        return fig, ax

    def plot_energy_over_time(self, energies, ax=None, labels=None):
        """
        Plot energy vs. iteration for one or more trajectories.

        Parameters
        ----------
        energies : ndarray
            Shape (T,) for one trajectory or (P, T) for several.
        labels : list of str, optional
        """
        import matplotlib.pyplot as plt

        energies = np.atleast_2d(energies)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure
        for i, e in enumerate(energies):
            lbl = labels[i] if labels else f'Trajectory {i}'
            ax.plot(e, 'o-', markersize=3, linewidth=1.5, label=lbl)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Evolution')
        if energies.shape[0] <= 12:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    def plot_state_evolution(self, states, targets=None, ax=None):
        """
        Visualize state trajectories.

        * D = 2 : 2-D scatter/trajectory in state space.
        * D = 3 : 3-D trajectory.
        * D > 3 : heatmap of neuron activations over iterations.
        """
        import matplotlib.pyplot as plt

        if states.ndim == 1:
            states = states.reshape(-1, 1)

        if self.D == 2 and states.ndim == 2:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 6))
            else:
                fig = ax.figure
            ax.plot(states[0], states[1], '-o', ms=4, lw=1.2, zorder=3,
                    label='Trajectory')
            ax.plot(states[0, 0], states[1, 0], 's', color='green', ms=10,
                    zorder=4, label='Start')
            ax.plot(states[0, -1], states[1, -1], '*', color='red', ms=14,
                    zorder=4, label='End')
            if targets is not None:
                tgt = np.atleast_2d(targets)
                ax.scatter(tgt[:, 0], tgt[:, 1], marker='D', s=100,
                           c='gold', edgecolors='black', zorder=5,
                           label='Targets')
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.set_xlabel('Neuron 1')
            ax.set_ylabel('Neuron 2')
            ax.set_title('State Evolution (2D)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            plt.tight_layout()
            return fig, ax

        if self.D == 3 and states.ndim == 2:
            if ax is None:
                fig = plt.figure(figsize=(8, 7))
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig = ax.figure
            ax.plot(states[0], states[1], states[2], '-o', ms=4, lw=1.2,
                    label='Trajectory')
            ax.scatter(*states[:, 0], s=100, c='green', marker='s',
                       depthshade=False, label='Start')
            ax.scatter(*states[:, -1], s=150, c='red', marker='*',
                       depthshade=False, label='End')
            if targets is not None:
                tgt = np.atleast_2d(targets)
                ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2], s=120,
                           c='gold', marker='D', edgecolors='black',
                           depthshade=False, label='Targets')
            ax.set_xlabel('N1')
            ax.set_ylabel('N2')
            ax.set_zlabel('N3')
            ax.set_title('State Evolution (3D)')
            ax.legend(fontsize=8)
            plt.tight_layout()
            return fig, ax

        # High-dimensional: heatmap
        display = states if states.ndim == 2 else states[0]
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, max(3, self.D * 0.12)))
        else:
            fig = ax.figure
        im = ax.imshow(display, aspect='auto', cmap='RdBu_r', vmin=-1,
                       vmax=1, interpolation='nearest')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Neuron index')
        ax.set_title('State Evolution')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        plt.tight_layout()
        return fig, ax

    # ---- energy landscape ------------------------------------------------

    def plot_energy_landscape(self, trajectories=None, resolution=80,
                              ax=None):
        """
        Contour/surface plot of the energy (D <= 3 only).

        Parameters
        ----------
        trajectories : list of ndarray, optional
            State trajectories to overlay, each shape (D, T).
        resolution : int
            Grid density per dimension.
        """
        import matplotlib.pyplot as plt

        if self.D > 3:
            print("Energy landscape plot requires D <= 3.")
            return None, None

        xy = np.linspace(-1, 1, resolution)

        if self.D == 2:
            X, Y = np.meshgrid(xy, xy)
            grid = np.column_stack([X.ravel(), Y.ravel()])
            E = self.energy(grid).reshape(resolution, resolution)

            if ax is None:
                fig, ax = plt.subplots(figsize=(7, 6))
            else:
                fig = ax.figure

            cont = ax.contourf(X, Y, E, levels=40, cmap='viridis')
            ax.contour(X, Y, E, levels=20, colors='white', linewidths=0.3,
                       alpha=0.5)
            fig.colorbar(cont, ax=ax, label='Energy')

            for i, t in enumerate(self._targets):
                ax.plot(t[0], t[1], 'D', color='gold', ms=12, mec='black',
                        mew=1.5, label='Target' if i == 0 else None)
                ax.annotate(
                    f'T{i}', (t[0], t[1]), xytext=(8, 8),
                    textcoords='offset points', fontsize=9,
                    fontweight='bold', color='white',
                )

            if trajectories:
                cmap_t = plt.colormaps['tab10']
                for j, traj in enumerate(trajectories):
                    traj = np.atleast_2d(traj)
                    if traj.shape[0] == self.D:
                        ax.plot(traj[0], traj[1], '-o', ms=3, lw=1.5,
                                color=cmap_t(j), label=f'Traj {j}')
                        ax.plot(traj[0, 0], traj[1, 0], 's',
                                color=cmap_t(j), ms=8)

            ax.set_xlabel('Neuron 1')
            ax.set_ylabel('Neuron 2')
            ax.set_title('Energy Landscape')
            ax.legend(fontsize=8, loc='upper right')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect('equal')
            plt.tight_layout()
            return fig, ax

        # D == 3 : show three slices at neuron3 = {-1, 0, 1}
        fig = plt.figure(figsize=(16, 5))
        X, Y = np.meshgrid(xy, xy)
        for si, sv in enumerate([-1.0, 0.0, 1.0]):
            ax_s = fig.add_subplot(1, 3, si + 1)
            grid = np.column_stack(
                [X.ravel(), Y.ravel(), np.full(resolution ** 2, sv)]
            )
            E = self.energy(grid).reshape(resolution, resolution)
            cont = ax_s.contourf(X, Y, E, levels=30, cmap='viridis')
            fig.colorbar(cont, ax=ax_s, fraction=0.046, pad=0.04)
            for t in self._targets:
                if abs(t[2] - sv) < 0.5:
                    ax_s.plot(t[0], t[1], 'D', color='gold', ms=10,
                              mec='black')
            ax_s.set_title(f'Neuron 3 = {sv:.0f}')
            ax_s.set_xlabel('Neuron 1')
            ax_s.set_ylabel('Neuron 2')
            ax_s.set_aspect('equal')

        fig.suptitle('Energy Landscape (3D slices)', fontsize=13)
        plt.tight_layout()
        return fig, None

    # ---- basins of attraction -------------------------------------------

    def plot_basins(self, num_iter=50, resolution=60, num_samples=None,
                    sync=True, ax=None):
        """
        Visualize basins of attraction colour-coded by convergent attractor.

        * D = 2 : dense 2-D image.
        * D = 3 : 3-D scatter.
        * D > 3 : PCA projection.
        """
        import matplotlib.pyplot as plt

        # Build initial state grid
        if self.D == 2:
            xy = np.linspace(-1, 1, resolution)
            X, Y = np.meshgrid(xy, xy)
            initial = np.column_stack([X.ravel(), Y.ravel()])
        elif self.D == 3:
            r3 = min(resolution, 20)
            xyz = np.linspace(-1, 1, r3)
            Xg, Yg, Zg = np.meshgrid(xyz, xyz, xyz)
            initial = np.column_stack(
                [Xg.ravel(), Yg.ravel(), Zg.ravel()]
            )
        else:
            n = num_samples or 5000
            initial = self.rng.choice([-1.0, 1.0], size=(n, self.D))

        final = self._evolve(initial, num_iter=num_iter, sync=sync)

        # Cluster into attractors
        attractors = []
        labels = np.zeros(len(initial), dtype=int)
        for i, fs in enumerate(final):
            matched = False
            for j, a in enumerate(attractors):
                if np.allclose(fs, a, atol=0.5):
                    labels[i] = j
                    matched = True
                    break
            if not matched:
                labels[i] = len(attractors)
                attractors.append(fs.copy())
        attractors = np.array(attractors)
        n_att = len(attractors)
        cmap = plt.colormaps.get_cmap('tab10').resampled(max(n_att, 1))
        fractions = np.array(
            [np.mean(labels == j) for j in range(n_att)]
        )

        # ---------- 2-D ----------
        if self.D == 2:
            if ax is None:
                fig, ax = plt.subplots(figsize=(7, 6))
            else:
                fig = ax.figure
            ax.imshow(
                labels.reshape(resolution, resolution),
                extent=[-1, 1, -1, 1], origin='lower', cmap=cmap,
                alpha=0.65, aspect='equal', interpolation='nearest',
            )
            seen_labels = set()
            for j, a in enumerate(attractors):
                is_tgt = any(
                    np.allclose(a, t, atol=0.5) for t in self._targets
                )
                mk = '*' if is_tgt else 'X'
                lbl_str = 'Stored attractor' if is_tgt else 'Spurious attractor'
                show_lbl = lbl_str if lbl_str not in seen_labels else None
                seen_labels.add(lbl_str)
                ax.scatter(
                    a[0], a[1], c=[cmap(j)], s=300, marker=mk,
                    edgecolors='black', linewidths=1.5, zorder=5,
                    label=show_lbl,
                )
                ax.annotate(
                    f'{fractions[j]:.0%}', (a[0], a[1]),
                    textcoords='offset points', xytext=(10, 10),
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black',
                              alpha=0.6),
                )
            ax.set_xlabel('Neuron 1')
            ax.set_ylabel('Neuron 2')
            ax.set_title('Basins of Attraction')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            return fig, ax

        # ---------- 3-D ----------
        if self.D == 3:
            if ax is None:
                fig = plt.figure(figsize=(9, 7))
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig = ax.figure
            colors_arr = [cmap(l) for l in labels]
            ax.scatter(
                initial[:, 0], initial[:, 1], initial[:, 2],
                c=colors_arr, s=8, alpha=0.3,
            )
            for j, a in enumerate(attractors):
                ax.scatter(
                    a[0], a[1], a[2], c=[cmap(j)], s=250, marker='*',
                    edgecolors='black', linewidths=1.5, depthshade=False,
                    zorder=5,
                )
            ax.set_xlabel('N1')
            ax.set_ylabel('N2')
            ax.set_zlabel('N3')
            ax.set_title('Basins of Attraction (3D)')
            plt.tight_layout()
            return fig, ax

        # ---------- high-D: PCA ----------
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        proj = pca.fit_transform(initial)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        ax.scatter(
            proj[:, 0], proj[:, 1], c=labels, cmap='tab10', s=8,
            alpha=0.6,
        )
        att_proj = pca.transform(attractors)
        for j in range(n_att):
            ax.scatter(
                att_proj[j, 0], att_proj[j, 1], s=200, marker='*',
                edgecolors='black', linewidths=1.5, c=[cmap(j)], zorder=5,
            )
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('Basins of Attraction (PCA projection)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    # ---- capacity --------------------------------------------------------

    @staticmethod
    def plot_capacity_analysis(dim, max_patterns=None, trials=30,
                               alg='Hebb', ax=None):
        """
        Run a capacity experiment and plot retrieval rate vs. pattern count.
        """
        import matplotlib.pyplot as plt

        if max_patterns is None:
            max_patterns = max(int(0.3 * dim), 5)

        counts, rates = HopfieldNetwork.capacity_test(
            dim, max_patterns, trials=trials, alg=alg,
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure
        ax.plot(counts, rates, 'o-', lw=2, ms=4, color='steelblue',
                label='Retrieval rate')
        cap = int(0.14 * dim)
        ax.axvline(x=cap, color='red', ls='--', lw=1.5,
                   label=f'Theoretical capacity ~ {cap}')
        ax.axhline(y=1.0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Number of Stored Patterns')
        ax.set_ylabel('Retrieval Success Rate')
        ax.set_title(f'Capacity Analysis  (D = {dim},  alg = {alg})')
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    # ---- pattern reconstruction -----------------------------------------

    @staticmethod
    def plot_pattern_reconstruction(original, noisy, reconstructed,
                                    shape=None, max_cols=10):
        """
        Show original / noisy / reconstructed patterns side by side.

        Parameters
        ----------
        original, noisy, reconstructed : array_like
            Each (P, D) or (D,).
        shape : tuple, optional
            2-D shape for image display, e.g. (28, 28).
        """
        import matplotlib.pyplot as plt

        original = np.atleast_2d(original)
        noisy = np.atleast_2d(noisy)
        reconstructed = np.atleast_2d(reconstructed)
        P = original.shape[0]
        ncols = min(P, max_cols)

        fig, axes = plt.subplots(3, ncols,
                                 figsize=(1.8 * ncols, 5.4))
        if ncols == 1:
            axes = axes[:, np.newaxis]

        row_labels = ['Original', 'Noisy', 'Reconstructed']
        for row, (data, lbl) in enumerate(
            zip([original, noisy, reconstructed], row_labels)
        ):
            for col in range(ncols):
                ax = axes[row, col]
                vec = data[col]
                D = len(vec)
                if shape is not None:
                    ax.imshow(vec.reshape(shape), cmap='gray',
                              vmin=-1, vmax=1)
                else:
                    side = int(np.sqrt(D))
                    if side * side == D:
                        ax.imshow(vec.reshape(side, side), cmap='gray',
                                  vmin=-1, vmax=1)
                    else:
                        ax.bar(range(D), vec, color='steelblue', width=1.0)
                        ax.set_ylim(-1.2, 1.2)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(lbl, fontsize=10)
                if row == 0:
                    ax.set_title(f'#{col}', fontsize=9)

        fig.suptitle('Pattern Reconstruction', fontsize=13)
        plt.tight_layout()
        return fig, axes

    # ---- comprehensive dashboard ----------------------------------------

    def plot_dashboard(self, data, num_iter=20, sync=True, shape=None):
        """
        All-in-one dashboard: energy decay, weight matrix, state evolution,
        and (for images) snapshots at selected iterations.

        Parameters
        ----------
        data : array_like
            Single initial state (D,).
        shape : tuple, optional
            Image shape, e.g. (28, 28).
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        data_arr = np.asarray(data, dtype=float)
        states, energies = self.simulate(data_arr, num_iter=num_iter,
                                         sync=sync)

        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

        # -- energy --
        ax_e = fig.add_subplot(gs[0, 0])
        ax_e.plot(energies, 'o-', color='steelblue', ms=4, lw=1.5)
        ax_e.set_xlabel('Iteration')
        ax_e.set_ylabel('Energy')
        ax_e.set_title('Energy Decay')
        ax_e.grid(True, alpha=0.3)

        # -- weight matrix --
        ax_w = fig.add_subplot(gs[0, 1])
        vmax = np.max(np.abs(self.W)) or 1.0
        im_w = ax_w.imshow(self.W, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        fig.colorbar(im_w, ax=ax_w, fraction=0.046, pad=0.04)
        ax_w.set_title('Weights')

        # -- state heatmap --
        ax_h = fig.add_subplot(gs[0, 2:])
        display = states if states.ndim == 2 else states[0]
        im_h = ax_h.imshow(display, aspect='auto', cmap='RdBu_r',
                           vmin=-1, vmax=1, interpolation='nearest')
        ax_h.set_xlabel('Iteration')
        ax_h.set_ylabel('Neuron')
        ax_h.set_title('State Evolution')
        fig.colorbar(im_h, ax=ax_h, fraction=0.02, pad=0.02)

        # -- bottom row --
        if shape is not None:
            n_snaps = 4
            step_idx = np.linspace(0, num_iter, n_snaps, dtype=int)
            for si, step in enumerate(step_idx):
                ax_s = fig.add_subplot(gs[1, si])
                snap = states[:, step] if states.ndim == 2 else states[step]
                ax_s.imshow(snap.reshape(shape), cmap='gray', vmin=-1,
                            vmax=1)
                ax_s.set_title(f'Step {step}  (E={energies[step]:.2f})',
                               fontsize=9)
                ax_s.axis('off')
        else:
            ax_bar = fig.add_subplot(gs[1, :])
            init_v = states[:, 0] if states.ndim == 2 else states[0]
            final_v = states[:, -1] if states.ndim == 2 else states[-1]
            x = np.arange(self.D)
            w = 0.35
            ax_bar.bar(x - w / 2, init_v, w, label='Initial', alpha=0.7,
                       color='coral')
            ax_bar.bar(x + w / 2, final_v, w, label='Final', alpha=0.7,
                       color='steelblue')
            ax_bar.set_xlabel('Neuron')
            ax_bar.set_ylabel('Activation')
            ax_bar.set_title('Initial vs Final State')
            ax_bar.legend()
            ax_bar.grid(True, alpha=0.3)

        fig.suptitle('Hopfield Network Dashboard', fontsize=14, y=1.01)
        plt.tight_layout()
        return fig

    # ---- multi-trajectory comparison ------------------------------------

    def plot_multi_trajectory(self, data_list, num_iter=20, sync=True,
                              labels=None):
        """
        Run several initial states and compare energy curves + final states.

        Parameters
        ----------
        data_list : list of array_like
            Each element is an initial state (D,).
        """
        import matplotlib.pyplot as plt

        n = len(data_list)
        all_energies = []
        all_finals = []
        for d in data_list:
            st, en = self.simulate(d, num_iter=num_iter, sync=sync)
            all_energies.append(en)
            final = st[:, -1] if st.ndim == 2 else st
            all_finals.append(final)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Energy curves
        for i, e in enumerate(all_energies):
            lbl = labels[i] if labels else f'Init {i}'
            axes[0].plot(e, 'o-', ms=3, lw=1.5, label=lbl)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Energy')
        axes[0].set_title('Energy Comparison')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Final states grouped bar
        x = np.arange(self.D)
        width = 0.8 / n
        cmap = plt.colormaps['tab10']
        for i, f in enumerate(all_finals):
            lbl = labels[i] if labels else f'Init {i}'
            axes[1].bar(x + i * width, f, width, label=lbl, alpha=0.8,
                        color=cmap(i))
        axes[1].set_xlabel('Neuron')
        axes[1].set_ylabel('Final Activation')
        axes[1].set_title('Final States')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

    # ---- Hamming distance helper ----------------------------------------

    @staticmethod
    def hamming_distance(a, b):
        """Hamming distance between two bipolar vectors."""
        a = np.sign(np.asarray(a, dtype=float))
        b = np.sign(np.asarray(b, dtype=float))
        return int(np.sum(a != b))

    def nearest_target(self, state):
        """Return (index, distance) of the closest stored target."""
        state = np.asarray(state, dtype=float)
        dists = [self.hamming_distance(state, t) for t in self._targets]
        idx = int(np.argmin(dists))
        return idx, dists[idx]
