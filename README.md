# Ornstein-Uhlenbeck Process Visualizer & Analyzer

An interactive, research-grade desktop simulation and analytical dashboard for studying the **Ornstein-Uhlenbeck (OU) process**—a fundamental continuous-time stochastic process widely utilized in statistical physics, computational finance, and mathematical biology.

The visualizer implements the **Euler-Maruyama integration scheme** to simulate stochastic paths and offers comparative analysis of simulated trajectories against theoretical statistical benchmarks.

---

## Mathematical Background

### 1. The Stochastic Differential Equation (SDE)
The Ornstein-Uhlenbeck process is defined by the following SDE:
\[ dX_t = \theta(\mu - X_t)dt + \sigma dW_t \]

where:
* **$\theta \ge 0$ (Reversion Speed / Rate):** Quantifies how forcefully the process is pulled back toward the equilibrium level.
* **$\mu$ (Long-term Mean / Equilibrium):** The long-term mean level that the process reverts to.
* **$\sigma > 0$ (Volatility / Diffusion Coefficient):** Governs the magnitude of random fluctuations driven by Gaussian noise.
* **$W_t$ (Wiener Process):** A standard Brownian motion representing continuous random shocks.

### 2. Numerical Integration (Euler-Maruyama Scheme)
To approximate solutions numerically, we discretize a continuous time horizon $T$ into $N$ steps of size $\Delta t = T/N$. The state transitions are defined by:
\[ X_{i+1} = X_i + \theta(\mu - X_i)\Delta t + \sigma \sqrt{\Delta t} Z_i \]

where $Z_i \sim N(0, 1)$ are independent and identically distributed (i.i.d.) standard normal random variables.

### 3. Transition Density
Conditioned on an initial state $X_0$ at $t=0$, the distribution at any future time $t > 0$ remains Gaussian:
\[ X_t \mid X_0 \sim N\left(X_0 e^{-\theta t} + \mu(1 - e^{-\theta t}),\ \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})\right) \]

As $t \to 0$, this distribution approaches a Dirac delta function centered at $X_0$.

### 4. Stationary Distribution
As time goes to infinity ($t \to \infty$), the process converges to a stationary (equilibrium) probability distribution if $\theta > 0$:
\[ X_\infty \sim N\left(\mu,\ \frac{\sigma^2}{2\theta}\right) \]

The balance between mean-reversion pull ($\theta$) and stochastic dispersion ($\sigma$) determines the stationary variance.

### 5. Autocorrelation Function (ACF)
For a stationary OU process, the autocorrelation between observations separated by a lag $\tau$ decays exponentially:
\[ \rho(\tau) = e^{-\theta \tau} \]

This demonstrates mean reversion with a characteristic correlation time scale of $\tau_{\text{corr}} = 1/\theta$.

---

## Key Features

1. **Ensemble Trajectory Simulator:**
   * Simulates up to 1,000 independent paths simultaneously.
   * Highlights the empirical mean path.
   * Renders shaded $1\sigma$ (68%) and $2\sigma$ (95%) confidence ribbons (empirical or theoretical).
2. **Stationary Distribution Fitting:**
   * Plots a normalized histogram of path endpoints ($X_T$) alongside the theoretical Gaussian PDF curve.
3. **Autocorrelation Analysis:**
   * Computes empirical ACF averaged across the ensemble and overlays the theoretical decay curve ($e^{-\theta \tau}$).
4. **Time-Slice / Transition Explorer:**
   * A scrub slider lets you select any step $t \in [0, T]$ and observe the cross-sectional histogram of the ensemble as it morphs from the initial condition toward the stationary distribution.
5. **Comparison Mode:**
   * Snapshot a run and overlay its mean and distribution limits on subsequent simulations to test parameter sensitivity.
6. **Presets Library:**
   * Includes typical model configurations: *Standard*, *Strong Reversion*, *High Volatility*, *Random Walk ($\theta=0$)*, *Vasicek Interest Rate Model*, and *Stationary Initial Conditions*.
7. **Research Utilities:**
   * Export all paths and summary statistics to a clean CSV spreadsheet.
   * Save publication-ready, high-resolution figures in PNG, PDF, or SVG formats.

---

## Installation & Running

### Requirements
* **Python 3.x**
* **NumPy**
* **Matplotlib**

The graphical interface is built using standard Python libraries (`tkinter`), meaning no heavy external UI framework (like Qt) is required.

### Launching the Application
Navigate to the project directory and run the entry script:
```bash
python3 main.py
```

---

## Usage Guide
* **Adjusting parameters:** Use the sliders in the sidebar to change model parameters. When you release a slider, the simulation runs automatically. You can also edit steps/paths in the spinboxes and press **Enter** or click **Run Simulation**.
* **Preserving runs:** Click **Save to Compare** to lock in your current configuration. Slide parameters to a new value to visualize the differences directly.
* **Inspecting evolution:** Go to the *Time-Slice Analysis* tab and scrub the slider at the bottom. Watch how a fixed starting state spreads out over time and eventually takes the shape of the stationary Gaussian PDF.