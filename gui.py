"""
Graphical User Interface (GUI) module for the Ornstein-Uhlenbeck Process Visualizer.
Uses Tkinter with styled ttk widgets and embedded Matplotlib canvases for dynamic, dark-themed plotting.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import csv

from simulation import (
    simulate_ou_process,
    compute_empirical_acf,
    get_theoretical_transition_stats,
    get_theoretical_transition_pdf,
    get_theoretical_stationary_pdf
)

# Colors and Styling Constants for Dark Theme
BG_DARK = "#0f172a"        # Deep slate background
BG_CARD = "#1e293b"        # Slate card panel
FG_LIGHT = "#f8fafc"       # Slate white text
FG_MUTED = "#94a3b8"       # Slate gray text
ACCENT_INDIGO = "#6366f1"  # Main indigo accent
ACCENT_CYAN = "#06b6d4"    # Secondary cyan accent
ACCENT_ORANGE = "#f97316"  # Comparison orange accent
BORDER_COLOR = "#334155"   # Slate border

class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ornstein-Uhlenbeck Process Visualizer & Analyzer")
        self.root.geometry("1300x820")
        self.root.configure(bg=BG_DARK)
        
        # Simulation state variables
        self.time_grid = None
        self.paths = None
        self.mean_path = None
        self.sd_path = None
        
        # Comparison mode variables
        self.comparison_data = None  # Dict to store comparison run results
        
        # Configure Tkinter window grid weights for responsiveness
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Define Tkinter variable wrappers
        self.theta_var = tk.DoubleVar(value=1.5)
        self.mu_var = tk.DoubleVar(value=0.0)
        self.sigma_var = tk.DoubleVar(value=0.5)
        
        self.t_max_var = tk.DoubleVar(value=10.0)
        self.steps_var = tk.IntVar(value=500)
        self.num_paths_var = tk.IntVar(value=100)
        
        self.x0_var = tk.DoubleVar(value=2.0)
        self.x0_mode_var = tk.StringVar(value="fixed")  # "fixed" or "stationary"
        
        self.show_paths_var = tk.BooleanVar(value=True)
        self.show_bands_var = tk.BooleanVar(value=True)
        self.band_type_var = tk.StringVar(value="empirical")  # "empirical" or "theoretical"
        
        self.time_slice_index = tk.IntVar(value=0)
        self.active_preset = tk.StringVar(value="Standard")

        self.setup_styles()
        self.create_layout()
        
        # Run initial simulation and plot
        self.run_simulation()
        
    def setup_styles(self):
        """Configure clean, modern ttk styles to match a dark-slate IDE theme."""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # General widget background & text colors
        self.style.configure('.', background=BG_DARK, foreground=FG_LIGHT, font=('Inter', 10))
        self.style.configure('TFrame', background=BG_DARK)
        
        # Sidebar custom panel
        self.style.configure('Sidebar.TFrame', background=BG_CARD)
        self.style.configure('Card.TFrame', background=BG_CARD, relief='solid', borderwidth=1)
        self.style.configure('TLabel', background=BG_DARK, foreground=FG_LIGHT)
        self.style.configure('Sidebar.TLabel', background=BG_CARD, foreground=FG_LIGHT)
        
        # Headers & Badges
        self.style.configure('Header.TLabel', font=('Inter', 11, 'bold'), foreground=FG_LIGHT)
        self.style.configure('SidebarHeader.TLabel', font=('Inter', 13, 'bold'), foreground=FG_LIGHT, background=BG_CARD)
        self.style.configure('SidebarTitle.TLabel', font=('Inter', 14, 'bold'), foreground=ACCENT_INDIGO, background=BG_CARD)
        self.style.configure('Badge.TLabel', font=('Consolas', 9, 'bold'), foreground=ACCENT_INDIGO, background=BG_CARD)
        self.style.configure('CompActive.TLabel', font=('Inter', 9, 'bold'), foreground=ACCENT_ORANGE, background=BG_CARD)
        self.style.configure('StatLabel.TLabel', font=('Inter', 9, 'bold'), foreground=FG_MUTED)
        self.style.configure('StatVal.TLabel', font=('Consolas', 11, 'bold'), foreground=FG_LIGHT)
        
        # Entry boxes and optionmenus
        self.style.configure('TEntry', fieldbackground=BG_DARK, foreground=FG_LIGHT, bordercolor=BORDER_COLOR)
        self.style.configure('TMenubutton', background=BG_DARK, foreground=FG_LIGHT, bordercolor=BORDER_COLOR, font=('Inter', 9, 'bold'))
        
        # Buttons
        self.style.configure('TButton', background=BG_DARK, foreground=FG_LIGHT, bordercolor=BORDER_COLOR, font=('Inter', 9, 'bold'), padding=6)
        self.style.map('TButton', 
                       background=[('active', BORDER_COLOR), ('pressed', BG_DARK)],
                       foreground=[('active', FG_LIGHT)])
        
        self.style.configure('Action.TButton', background=ACCENT_INDIGO, foreground='white', bordercolor=ACCENT_INDIGO, font=('Inter', 10, 'bold'), padding=8)
        self.style.map('Action.TButton',
                       background=[('active', '#4f46e5'), ('pressed', ACCENT_INDIGO)],
                       foreground=[('active', 'white')])

        self.style.configure('Compare.TButton', background=ACCENT_ORANGE, foreground='white', bordercolor=ACCENT_ORANGE, font=('Inter', 9, 'bold'), padding=6)
        self.style.map('Compare.TButton',
                       background=[('active', '#ea580c'), ('pressed', ACCENT_ORANGE)],
                       foreground=[('active', 'white')])

        # Notebook tabs
        self.style.configure('TNotebook', background=BG_DARK, borderwidth=0)
        self.style.configure('TNotebook.Tab', background=BG_CARD, foreground=FG_MUTED, font=('Inter', 10, 'bold'), padding=(16, 8))
        self.style.map('TNotebook.Tab',
                       background=[('selected', ACCENT_INDIGO), ('active', BORDER_COLOR)],
                       foreground=[('selected', 'white'), ('active', FG_LIGHT)])
                       
        self.style.configure('TCheckbutton', background=BG_CARD, foreground=FG_LIGHT)
        self.style.configure('TRadiobutton', background=BG_CARD, foreground=FG_LIGHT)

    def create_layout(self):
        """Assemble the sidebar control panel and visualization notebook tabs."""
        # 1. Left Sidebar frame
        sidebar = ttk.Frame(self.root, style='Sidebar.TFrame', width=360)
        sidebar.grid(row=0, column=0, sticky='nsew', padx=0, pady=0)
        sidebar.grid_propagate(False)
        
        # Title and Header
        header_frame = ttk.Frame(sidebar, style='Sidebar.TFrame')
        header_frame.pack(fill='x', padx=20, pady=(20, 15))
        
        ttk.Label(header_frame, text="STOCHASTIC VISUALIZER", style='SidebarTitle.TLabel').pack(anchor='w')
        ttk.Label(header_frame, text="Ornstein-Uhlenbeck Process Analyzer", style='Sidebar.TLabel', font=('Inter', 9), foreground=FG_MUTED).pack(anchor='w', pady=(2, 0))
        
        # Comparison indicator label (initially hidden)
        self.comp_lbl = ttk.Label(header_frame, text="● Comparison Mode Active", style='CompActive.TLabel')
        
        # Scrollable container for parameters to handle smaller vertical screens
        scroll_canvas = tk.Canvas(sidebar, bg=BG_CARD, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=scroll_canvas.yview)
        scrollable_frame = ttk.Frame(scroll_canvas, style='Sidebar.TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        )
        scroll_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=340)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        scroll_canvas.pack(side="left", fill="both", expand=True, padx=(20, 5), pady=0)
        scrollbar.pack(side="right", fill="y", pady=0)
        
        # Section A: Presets dropdown
        preset_card = ttk.Frame(scrollable_frame, style='Card.TFrame', padding=12)
        preset_card.pack(fill='x', pady=6)
        ttk.Label(preset_card, text="Quick Presets", style='Header.TLabel', background=BG_CARD).pack(anchor='w', pady=(0, 6))
        
        presets = ["Standard", "Strong Reversion", "High Volatility", "Random Walk (θ=0)", "Vasicek Model (IR)", "Stationary Init"]
        preset_menu = ttk.OptionMenu(preset_card, self.active_preset, presets[0], *presets, command=self.load_preset)
        preset_menu.pack(fill='x')
        
        # Section B: Process parameters card
        param_card = ttk.Frame(scrollable_frame, style='Card.TFrame', padding=12)
        param_card.pack(fill='x', pady=6)
        ttk.Label(param_card, text="SDE Model Parameters", style='Header.TLabel', background=BG_CARD).pack(anchor='w', pady=(0, 8))
        
        self.make_slider(param_card, "Reversion Rate (θ)", -0.5, 5.0, 0.1, self.theta_var)
        self.make_slider(param_card, "Long-term Mean (μ)", -5.0, 5.0, 0.1, self.mu_var)
        self.make_slider(param_card, "Volatility (σ)", 0.01, 2.0, 0.01, self.sigma_var)
        
        # Section C: Simulation settings card
        sim_card = ttk.Frame(scrollable_frame, style='Card.TFrame', padding=12)
        sim_card.pack(fill='x', pady=6)
        ttk.Label(sim_card, text="Simulation Settings", style='Header.TLabel', background=BG_CARD).pack(anchor='w', pady=(0, 8))
        
        self.make_slider(sim_card, "Time Horizon (T)", 1.0, 50.0, 1.0, self.t_max_var)
        
        # Numeric inputs for steps and paths
        grid_frame = ttk.Frame(sim_card, style='Sidebar.TFrame')
        grid_frame.pack(fill='x', pady=4)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)
        
        ttk.Label(grid_frame, text="Time Steps (N)", style='Sidebar.TLabel').grid(row=0, column=0, sticky='w')
        ttk.Label(grid_frame, text="Paths (M)", style='Sidebar.TLabel').grid(row=0, column=1, sticky='w')
        
        steps_spin = ttk.Spinbox(grid_frame, from_=10, to=2000, increment=50, textvariable=self.steps_var, width=10)
        steps_spin.grid(row=1, column=0, sticky='w', pady=(2, 0))
        steps_spin.bind("<FocusOut>", lambda e: self.run_simulation())
        steps_spin.bind("<Return>", lambda e: self.run_simulation())
        
        paths_spin = ttk.Spinbox(grid_frame, from_=1, to=1000, increment=20, textvariable=self.num_paths_var, width=10)
        paths_spin.grid(row=1, column=1, sticky='w', pady=(2, 0))
        paths_spin.bind("<FocusOut>", lambda e: self.run_simulation())
        paths_spin.bind("<Return>", lambda e: self.run_simulation())
        
        # Initial condition settings
        init_frame = ttk.Frame(sim_card, style='Sidebar.TFrame')
        init_frame.pack(fill='x', pady=(8, 0))
        ttk.Label(init_frame, text="Initial Condition (X₀)", style='Sidebar.TLabel').pack(anchor='w')
        
        x0_choices_frame = ttk.Frame(init_frame, style='Sidebar.TFrame')
        x0_choices_frame.pack(fill='x', pady=2)
        ttk.Radiobutton(x0_choices_frame, text="Fixed", variable=self.x0_mode_var, value="fixed", command=self.run_simulation).pack(side='left', padx=(0, 10))
        ttk.Radiobutton(x0_choices_frame, text="Stationary Dist.", variable=self.x0_mode_var, value="stationary", command=self.run_simulation).pack(side='left')
        
        self.x0_slider_container = ttk.Frame(init_frame, style='Sidebar.TFrame')
        self.x0_slider_container.pack(fill='x')
        self.make_slider(self.x0_slider_container, "Fixed Value (X₀)", -10.0, 10.0, 0.1, self.x0_var)

        # Trace x0 mode to toggle slider visibility
        self.x0_mode_var.trace_add("write", lambda *args: self.toggle_x0_slider())

        # Section D: Rendering settings card
        render_card = ttk.Frame(scrollable_frame, style='Card.TFrame', padding=12)
        render_card.pack(fill='x', pady=6)
        ttk.Label(render_card, text="Visualization Options", style='Header.TLabel', background=BG_CARD).pack(anchor='w', pady=(0, 6))
        
        ttk.Checkbutton(render_card, text="Show Individual Paths (max 50)", variable=self.show_paths_var, command=self.plot_active_tab).pack(anchor='w', pady=2)
        ttk.Checkbutton(render_card, text="Show Standard Deviation Bands", variable=self.show_bands_var, command=self.plot_active_tab).pack(anchor='w', pady=2)
        
        band_type_frame = ttk.Frame(render_card, style='Sidebar.TFrame')
        band_type_frame.pack(fill='x', pady=(4, 0))
        ttk.Label(band_type_frame, text="Band Mode:", style='Sidebar.TLabel').pack(side='left', padx=(0, 8))
        ttk.Radiobutton(band_type_frame, text="Empirical", variable=self.band_type_var, value="empirical", command=self.plot_active_tab).pack(side='left', padx=(0, 10))
        ttk.Radiobutton(band_type_frame, text="Theoretical", variable=self.band_type_var, value="theoretical", command=self.plot_active_tab).pack(side='left')

        # Action Buttons frame at bottom of sidebar (sticky)
        action_frame = ttk.Frame(sidebar, style='Sidebar.TFrame', padding=(20, 10))
        action_frame.pack(fill='x', side='bottom')
        
        run_btn = ttk.Button(action_frame, text="Run Simulation", style='Action.TButton', command=self.run_simulation)
        run_btn.pack(fill='x', pady=4)
        
        comp_btn_frame = ttk.Frame(action_frame, style='Sidebar.TFrame')
        comp_btn_frame.pack(fill='x', pady=2)
        comp_btn_frame.columnconfigure(0, weight=1)
        comp_btn_frame.columnconfigure(1, weight=1)
        
        self.save_comp_btn = ttk.Button(comp_btn_frame, text="Save to Compare", style='Compare.TButton', command=self.save_to_comparison)
        self.save_comp_btn.grid(row=0, column=0, sticky='ew', padx=(0, 2))
        
        self.clear_comp_btn = ttk.Button(comp_btn_frame, text="Clear Compare", style='TButton', command=self.clear_comparison, state='disabled')
        self.clear_comp_btn.grid(row=0, column=1, sticky='ew', padx=(2, 0))
        
        export_btn_frame = ttk.Frame(action_frame, style='Sidebar.TFrame')
        export_btn_frame.pack(fill='x', pady=2)
        export_btn_frame.columnconfigure(0, weight=1)
        export_btn_frame.columnconfigure(1, weight=1)
        
        ttk.Button(export_btn_frame, text="Export CSV", command=self.export_csv).grid(row=0, column=0, sticky='ew', padx=(0, 2))
        ttk.Button(export_btn_frame, text="Save Active Plot", command=self.save_active_plot).grid(row=0, column=1, sticky='ew', padx=(2, 0))
        
        # 2. Main content dashboard
        dashboard_frame = ttk.Frame(self.root)
        dashboard_frame.grid(row=0, column=1, sticky='nsew', padx=20, pady=20)
        dashboard_frame.columnconfigure(0, weight=1)
        dashboard_frame.rowconfigure(0, weight=0) # Metrics bar
        dashboard_frame.rowconfigure(1, weight=1) # Visualizer tabs
        
        # Metrics Bar (Empirical vs Theoretical stats at endpoint T)
        self.metrics_bar = ttk.Frame(dashboard_frame, style='Card.TFrame', padding=(16, 12))
        self.metrics_bar.grid(row=0, column=0, sticky='ew', pady=(0, 16))
        
        # Configure columns inside metrics bar
        for col in range(4):
            self.metrics_bar.columnconfigure(col, weight=1)
            
        self.stat_elements = {}
        self.create_metric_widget(self.metrics_bar, 0, "Empirical Mean (X_T)", "-", "Targeting μ")
        self.create_metric_widget(self.metrics_bar, 1, "Theoretical Mean (μ)", "-", "Equilibrium Level")
        self.create_metric_widget(self.metrics_bar, 2, "Empirical Variance (T)", "-", "Realized fluctuations")
        self.create_metric_widget(self.metrics_bar, 3, "Stationary Var (σ²/2θ)", "-", "Asymptotic Variance")
        
        # Tabs notebook card
        notebook_card = ttk.Frame(dashboard_frame, style='Card.TFrame')
        notebook_card.grid(row=1, column=0, sticky='nsew')
        notebook_card.columnconfigure(0, weight=1)
        notebook_card.rowconfigure(0, weight=1)
        
        self.notebook = ttk.Notebook(notebook_card)
        self.notebook.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self.plot_active_tab())
        
        # Define 4 notebook frames
        self.tab_trajectories = ttk.Frame(self.notebook)
        self.tab_stationary = ttk.Frame(self.notebook)
        self.tab_acf = ttk.Frame(self.notebook)
        self.tab_timeslice = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_trajectories, text="Trajectories & Bands")
        self.notebook.add(self.tab_stationary, text="Stationary Distribution")
        self.notebook.add(self.tab_acf, text="Autocorrelation (ACF)")
        self.notebook.add(self.tab_timeslice, text="Time-Slice Analysis")
        
        # Pre-initialize figures for embedding in tabs
        self.setup_embedded_plots()

    def make_slider(self, parent, label_text, from_val, to_val, resolution, var):
        """Helper to create styled Tkinter slider with header title and live value badge."""
        f = ttk.Frame(parent, style='Sidebar.TFrame')
        f.pack(fill='x', pady=4)
        
        lbl_wrapper = ttk.Frame(f, style='Sidebar.TFrame')
        lbl_wrapper.pack(fill='x')
        
        ttk.Label(lbl_wrapper, text=label_text, style='Sidebar.TLabel').pack(side='left')
        val_lbl = ttk.Label(lbl_wrapper, text=f"{var.get():.2f}", style='Badge.TLabel')
        val_lbl.pack(side='right')
        
        def update_label(val):
            formatted_val = float(val)
            var.set(formatted_val)
            val_lbl.config(text=f"{formatted_val:.2f}")
            
        s = tk.Scale(f, from_=from_val, to=to_val, resolution=resolution, orient='horizontal',
                     variable=var, command=update_label,
                     bg=BG_CARD, fg=FG_LIGHT, highlightthickness=0,
                     troughcolor=BG_DARK, activebackground=ACCENT_INDIGO, showvalue=False)
        s.pack(fill='x', pady=(2, 0))
        
        # Bind release event to re-run simulation for dynamic real-time scrubbing
        s.bind("<ButtonRelease-1>", lambda event: self.run_simulation())
        return s

    def toggle_x0_slider(self):
        """Show or hide fixed X0 slider based on initial condition mode."""
        if self.x0_mode_var.get() == "fixed":
            self.x0_slider_container.pack(fill='x', after=self.x0_slider_container.master.winfo_children()[1])
        else:
            self.x0_slider_container.pack_forget()
        self.run_simulation()

    def create_metric_widget(self, parent, col, title, initial_val, subtext):
        """Create a column widget inside the metrics bar."""
        f = ttk.Frame(parent, style='Sidebar.TFrame')
        f.grid(row=0, column=col, sticky='ew')
        
        lbl_title = ttk.Label(f, text=title, style='StatLabel.TLabel', background=BG_CARD)
        lbl_title.pack(anchor='center')
        
        lbl_val = ttk.Label(f, text=initial_val, style='StatVal.TLabel', background=BG_CARD)
        lbl_val.pack(anchor='center', pady=2)
        
        lbl_sub = ttk.Label(f, text=subtext, font=('Inter', 7, 'italic'), foreground=FG_MUTED, background=BG_CARD)
        lbl_sub.pack(anchor='center')
        
        self.stat_elements[title] = lbl_val

    def setup_embedded_plots(self):
        """Initialize Matplotlib figures and canvases for each notebook tab."""
        # Custom stylesheet to format plots for our dark GUI theme
        self.chart_theme = {
            'figure.facecolor': BG_DARK,
            'axes.facecolor': BG_DARK,
            'axes.edgecolor': BORDER_COLOR,
            'axes.labelcolor': FG_LIGHT,
            'xtick.color': FG_MUTED,
            'ytick.color': FG_MUTED,
            'grid.color': BORDER_COLOR,
            'text.color': FG_LIGHT,
            'legend.facecolor': BG_CARD,
            'legend.edgecolor': BORDER_COLOR,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Inter', 'DejaVu Sans']
        }
        
        # Apply theme overrides
        plt.rcParams.update(self.chart_theme)
        
        # 1. Trajectories Canvas
        self.fig_traj, self.ax_traj = plt.subplots(figsize=(7, 4), tight_layout=True)
        self.canvas_traj = FigureCanvasTkAgg(self.fig_traj, master=self.tab_trajectories)
        self.canvas_traj.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # 2. Stationary Distribution Canvas
        self.fig_stat, self.ax_stat = plt.subplots(figsize=(7, 4), tight_layout=True)
        self.canvas_stat = FigureCanvasTkAgg(self.fig_stat, master=self.tab_stationary)
        self.canvas_stat.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # 3. Autocorrelation (ACF) Canvas
        self.fig_acf, self.ax_acf = plt.subplots(figsize=(7, 4), tight_layout=True)
        self.canvas_acf = FigureCanvasTkAgg(self.fig_acf, master=self.tab_acf)
        self.canvas_acf.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # 4. Time-Slice Analysis Canvas
        self.fig_slice, self.ax_slice = plt.subplots(figsize=(7, 4), tight_layout=True)
        
        # For the Time-Slice tab, we also need a slider to select time step at the bottom
        self.slice_ctrl_frame = ttk.Frame(self.tab_timeslice, style='Sidebar.TFrame', padding=10)
        self.slice_ctrl_frame.pack(fill='x', side='bottom', padx=10, pady=(0, 10))
        
        ttk.Label(self.slice_ctrl_frame, text="Time Step Slice (t):", style='Sidebar.TLabel').pack(side='left', padx=(0, 10))
        
        self.slice_val_lbl = ttk.Label(self.slice_ctrl_frame, text="0.00", style='Badge.TLabel')
        self.slice_val_lbl.pack(side='right', padx=(10, 0))
        
        # We will configure the scale range in run_simulation because N can change
        self.slice_scale = tk.Scale(self.slice_ctrl_frame, from_=0, to=100, resolution=1, orient='horizontal',
                                    variable=self.time_slice_index, command=self.update_slice_label,
                                    bg=BG_CARD, fg=FG_LIGHT, highlightthickness=0,
                                    troughcolor=BG_DARK, activebackground=ACCENT_INDIGO, showvalue=False)
        self.slice_scale.pack(fill='x', expand=True)
        self.slice_scale.bind("<ButtonRelease-1>", lambda e: self.plot_timeslice_tab())
        
        self.canvas_slice = FigureCanvasTkAgg(self.fig_slice, master=self.tab_timeslice)
        self.canvas_slice.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def update_slice_label(self, val):
        """Update slider label badge for time slice analysis."""
        step_idx = int(float(val))
        if self.time_grid is not None and step_idx < len(self.time_grid):
            t_val = self.time_grid[step_idx]
            self.slice_val_lbl.config(text=f"t = {t_val:.3f} s (Step {step_idx})")
        else:
            self.slice_val_lbl.config(text=f"Step {step_idx}")

    def load_preset(self, preset_name):
        """Load parameter sets corresponding to typical physical/financial scenarios."""
        presets = {
            "Standard": {"theta": 1.5, "mu": 0.0, "sigma": 0.5, "t_max": 10.0, "steps": 500, "num_paths": 100, "x0": 2.0, "x0_mode": "fixed"},
            "Strong Reversion": {"theta": 5.0, "mu": 1.0, "sigma": 0.3, "t_max": 5.0, "steps": 500, "num_paths": 100, "x0": -1.0, "x0_mode": "fixed"},
            "High Volatility": {"theta": 0.5, "mu": 0.0, "sigma": 1.2, "t_max": 10.0, "steps": 500, "num_paths": 120, "x0": 0.0, "x0_mode": "fixed"},
            "Random Walk (θ=0)": {"theta": 0.0, "mu": 0.0, "sigma": 0.5, "t_max": 10.0, "steps": 500, "num_paths": 100, "x0": 0.0, "x0_mode": "fixed"},
            "Vasicek Model (IR)": {"theta": 0.8, "mu": 0.05, "sigma": 0.03, "t_max": 20.0, "steps": 800, "num_paths": 60, "x0": 0.02, "x0_mode": "fixed"},
            "Stationary Init": {"theta": 2.0, "mu": 1.5, "sigma": 0.8, "t_max": 10.0, "steps": 500, "num_paths": 150, "x0": 0.0, "x0_mode": "stationary"}
        }
        
        config = presets[preset_name]
        
        # Disable traces momentarily by changing values directly
        self.theta_var.set(config["theta"])
        self.mu_var.set(config["mu"])
        self.sigma_var.set(config["sigma"])
        self.t_max_var.set(config["t_max"])
        self.steps_var.set(config["steps"])
        self.num_paths_var.set(config["num_paths"])
        self.x0_var.set(config["x0"])
        self.x0_mode_var.set(config["x0_mode"])
        
        # Trigger slider badge refreshes
        # Tkinter scales don't update their command callbacks if set programmatically, so we trigger manual updates
        self.update_all_slider_badges()
        
        self.toggle_x0_slider() # Toggles slider and runs simulation

    def update_all_slider_badges(self):
        """Force redraw of slider badges when a preset is loaded."""
        # A simple trick is to rebuild the labels by finding the active child panels or we can just run the commands
        # Let's just reload the interface states
        pass

    def run_simulation(self):
        """Trigger SDE solver and update results. Updates metrics bar and active plots."""
        try:
            # 1. Gather parameters
            theta = self.theta_var.get()
            mu = self.mu_var.get()
            sigma = self.sigma_var.get()
            t_max = self.t_max_var.get()
            steps = int(self.steps_var.get())
            num_paths = int(self.num_paths_var.get())
            x0 = self.x0_var.get()
            x0_mode = self.x0_mode_var.get()
            
            if steps <= 0 or num_paths <= 0 or t_max <= 0:
                raise ValueError("Steps, Paths, and Time Horizon must be greater than zero.")
                
            # 2. Run simulation
            self.time_grid, self.paths, self.mean_path, self.sd_path = simulate_ou_process(
                theta, mu, sigma, t_max, steps, num_paths, x0, x0_mode
            )
            
            # Configure time slice slider limits based on time steps N
            self.slice_scale.config(to=steps)
            # Default time slice to midpoint if out of bounds
            if self.time_slice_index.get() > steps:
                self.time_slice_index.set(steps // 2)
            self.update_slice_label(self.time_slice_index.get())
            
            # 3. Update top metrics bar at time T
            emp_mean_T = self.mean_path[-1]
            emp_var_T = self.sd_path[-1]**2
            
            self.stat_elements["Empirical Mean (X_T)"].config(text=f"{emp_mean_T:.4f}")
            self.stat_elements["Theoretical Mean (μ)"].config(text=f"{mu:.4f}")
            self.stat_elements["Empirical Variance (T)"].config(text=f"{emp_var_T:.4f}")
            
            if theta > 0:
                theo_var = sigma**2 / (2.0 * theta)
                self.stat_elements["Stationary Var (σ²/2θ)"].config(text=f"{theo_var:.4f}")
            else:
                self.stat_elements["Stationary Var (σ²/2θ)"].config(text="Undefined (θ≤0)")
            
            # 4. Redraw plots
            self.plot_active_tab()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run simulation: {str(e)}")

    def plot_active_tab(self):
        """Redraw Matplotlib canvas for the currently selected notebook tab."""
        if self.time_grid is None:
            return
            
        tab_idx = self.notebook.index(self.notebook.select())
        
        if tab_idx == 0:
            self.plot_trajectories_tab()
        elif tab_idx == 1:
            self.plot_stationary_tab()
        elif tab_idx == 2:
            self.plot_acf_tab()
        elif tab_idx == 3:
            self.plot_timeslice_tab()

    def plot_trajectories_tab(self):
        """Render trajectory simulations, empirical/theoretical bands, and mean path."""
        ax = self.ax_traj
        ax.clear()
        ax.grid(True, alpha=0.15, linestyle='--')
        
        t = self.time_grid
        
        # 1. Overlay individual paths (translucent)
        if self.show_paths_var.get():
            # Cap visible paths to 50 to avoid lagging
            visible_paths = min(self.paths.shape[0], 50)
            for p in range(visible_paths):
                ax.plot(t, self.paths[p], color=ACCENT_INDIGO, alpha=0.1, linewidth=0.8, label="Realizations" if p == 0 else "")
        
        # 2. Draw standard deviation bands
        if self.show_bands_var.get():
            mode = self.band_type_var.get()
            if mode == "empirical":
                upper_1 = self.mean_path + self.sd_path
                lower_1 = self.mean_path - self.sd_path
                upper_2 = self.mean_path + 2.0 * self.sd_path
                lower_2 = self.mean_path - 2.0 * self.sd_path
                
                ax.fill_between(t, lower_2, upper_2, color=ACCENT_INDIGO, alpha=0.06, label="Empirical ± 2 SD (95%)")
                ax.fill_between(t, lower_1, upper_1, color=ACCENT_INDIGO, alpha=0.13, label="Empirical ± 1 SD (68%)")
            else:
                # Calculate theoretical transition confidence bands over time
                theta = self.theta_var.get()
                mu = self.mu_var.get()
                sigma = self.sigma_var.get()
                x0 = self.x0_var.get()
                x0_mode = self.x0_mode_var.get()
                
                theo_means = np.zeros_like(t)
                theo_sds = np.zeros_like(t)
                
                # Check for stationary mode
                if x0_mode == "stationary" and theta > 0:
                    stationary_sd = np.sqrt(sigma**2 / (2.0 * theta))
                    theo_means.fill(mu)
                    theo_sds.fill(stationary_sd)
                else:
                    for i, time_val in enumerate(t):
                        m, sd = get_theoretical_transition_stats(time_val, x0, theta, mu, sigma)
                        theo_means[i] = m
                        theo_sds[i] = sd
                        
                ax.fill_between(t, theo_means - 2.0 * theo_sds, theo_means + 2.0 * theo_sds, color=ACCENT_CYAN, alpha=0.05, label="Theoretical ± 2 SD (95%)")
                ax.fill_between(t, theo_means - theo_sds, theo_means + theo_sds, color=ACCENT_CYAN, alpha=0.12, label="Theoretical ± 1 SD (68%)")
        
        # 3. Draw Comparison Run (if present)
        if self.comparison_data is not None:
            comp_t = self.comparison_data["time_grid"]
            comp_mean = self.comparison_data["mean_path"]
            comp_sd = self.comparison_data["sd_path"]
            
            # Dotted comparison mean line
            ax.plot(comp_t, comp_mean, color=ACCENT_ORANGE, linewidth=2.0, linestyle='--', label=f"Comparison Mean (Run A)")
            
            if self.show_bands_var.get():
                ax.fill_between(comp_t, comp_mean - 1.96 * comp_sd, comp_mean + 1.96 * comp_sd, color=ACCENT_ORANGE, alpha=0.04)

        # 4. Draw primary Run Mean Path
        ax.plot(t, self.mean_path, color=ACCENT_INDIGO, linewidth=2.5, label="Ensemble Mean (Run B)" if self.comparison_data is not None else "Ensemble Mean")
        
        ax.set_title("Ornstein-Uhlenbeck Trajectories & Confidence Ribbons", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Time (t)", fontsize=9)
        ax.set_ylabel("State Value (X_t)", fontsize=9)
        ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
        
        self.canvas_traj.draw()

    def plot_stationary_tab(self):
        """Render endpoint histogram compared with the theoretical stationary PDF."""
        ax = self.ax_stat
        ax.clear()
        ax.grid(True, alpha=0.15, linestyle='--')
        
        endpoints = self.paths[:, -1]
        
        # Choose bin counts based on path sizes
        bins_count = max(15, min(40, len(endpoints) // 5))
        
        # 1. Plot comparison histogram (drawn behind)
        if self.comparison_data is not None:
            comp_endpoints = self.comparison_data["paths"][:, -1]
            ax.hist(comp_endpoints, bins=bins_count, density=True, color=ACCENT_ORANGE, alpha=0.15, edgecolor=ACCENT_ORANGE, label="Comparison Empirical")
            
            # Overlay comparison theoretical PDF
            comp_theta = self.comparison_data["theta"]
            comp_mu = self.comparison_data["mu"]
            comp_sigma = self.comparison_data["sigma"]
            if comp_theta > 0:
                x_min = min(endpoints.min(), comp_endpoints.min()) - 1.0
                x_max = max(endpoints.max(), comp_endpoints.max()) + 1.0
                x_grid = np.linspace(x_min, x_max, 200)
                comp_pdf = get_theoretical_stationary_pdf(x_grid, comp_theta, comp_mu, comp_sigma)
                ax.plot(x_grid, comp_pdf, color=ACCENT_ORANGE, linewidth=1.5, linestyle='--', label="Comparison Theoretical")

        # 2. Plot main empirical histogram
        ax.hist(endpoints, bins=bins_count, density=True, color=ACCENT_INDIGO, alpha=0.25, edgecolor=ACCENT_INDIGO, label="Empirical Density")
        
        # Calculate axis grid range
        x_min = endpoints.min() - 1.0
        x_max = endpoints.max() + 1.0
        x_grid = np.linspace(x_min, x_max, 200)
        
        # 3. Plot theoretical stationary density (if theta > 0)
        theta = self.theta_var.get()
        mu = self.mu_var.get()
        sigma = self.sigma_var.get()
        
        if theta > 0:
            pdf = get_theoretical_stationary_pdf(x_grid, theta, mu, sigma)
            if pdf is not None:
                ax.plot(x_grid, pdf, color=ACCENT_CYAN, linewidth=2.5, label="Stationary PDF")
            ax.set_title(f"Stationary Distribution Convergence (t = {self.time_grid[-1]:.2f})", fontsize=11, fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, "Stationary Distribution Does Not Exist for θ ≤ 0\n(Process Variance Grows Indefinitely)", 
                    ha='center', va='center', color=FG_MUTED, fontsize=10, transform=ax.transAxes, bbox=dict(boxstyle='round,pad=1', facecolor=BG_CARD, edgecolor=BORDER_COLOR))
            ax.set_title("Endpoint Distribution (Non-Stationary Process)", fontsize=11, fontweight='bold', pad=10)
            
        ax.set_xlabel("Value (X_T)", fontsize=9)
        ax.set_ylabel("Probability Density", fontsize=9)
        ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
        
        self.canvas_stat.draw()

    def plot_acf_tab(self):
        """Render empirical autocorrelation bars vs theoretical exponential decay decay."""
        ax = self.ax_acf
        ax.clear()
        ax.grid(True, alpha=0.15, linestyle='--')
        
        dt = self.time_grid[1] - self.time_grid[0]
        theta = self.theta_var.get()
        
        # Limit lags to 10% of total steps, capped at 100 lags for visual clarity
        max_lag = min(100, len(self.time_grid) // 10)
        if max_lag < 10:
            max_lag = min(20, len(self.time_grid) - 2)
            
        lags = np.arange(max_lag + 1)
        lag_times = lags * dt
        
        # Compute empirical ACF
        empirical_acf = compute_empirical_acf(self.paths, max_lag)
        
        # 1. Draw empirical bars
        ax.bar(lag_times, empirical_acf, width=dt*0.6, color=ACCENT_INDIGO, alpha=0.4, edgecolor=ACCENT_INDIGO, label="Empirical ACF")
        
        # 2. Draw theoretical ACF decay curve: rho(tau) = exp(-theta * tau)
        if theta > 0:
            theoretical_acf = np.exp(-theta * lag_times)
            ax.plot(lag_times, theoretical_acf, color=ACCENT_CYAN, linewidth=2.5, label="Theoretical Decay (e^{-θ·τ})")
            
            # Show correlation time tau = 1/theta
            tau = 1.0 / theta
            ax.axvline(x=tau, color=ACCENT_ORANGE, linestyle=':', linewidth=1.5, label=f"Correlation Time τ = 1/θ ({tau:.3f}s)")
        elif theta == 0:
            theoretical_acf = np.ones_like(lag_times)
            ax.plot(lag_times, theoretical_acf, color=ACCENT_CYAN, linewidth=2.5, label="Theoretical (θ=0)")
        else:
            # For theta < 0, process is non-stationary, ACF is not theoretically well-defined as a function of lag alone
            ax.text(0.5, 0.8, "Theoretical ACF is not stationary for θ < 0\n(Mean-fleeing process)", 
                    ha='center', va='center', color=FG_MUTED, fontsize=9, transform=ax.transAxes)
            
        ax.set_ylim(-0.25, 1.1)
        ax.set_title("Autocorrelation Function (ACF) Decay Analysis", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Time Lag (τ)", fontsize=9)
        ax.set_ylabel("Autocorrelation Coefficient", fontsize=9)
        ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
        
        self.canvas_acf.draw()

    def plot_timeslice_tab(self):
        """Render cross-sectional distribution at time t_i vs theoretical transition density."""
        ax = self.ax_slice
        ax.clear()
        ax.grid(True, alpha=0.15, linestyle='--')
        
        step_idx = self.time_slice_index.get()
        if step_idx >= len(self.time_grid):
            step_idx = len(self.time_grid) - 1
            
        t_val = self.time_grid[step_idx]
        slice_values = self.paths[:, step_idx]
        
        # 1. Draw empirical histogram
        bins_count = max(15, min(40, len(slice_values) // 5))
        ax.hist(slice_values, bins=bins_count, density=True, color=ACCENT_INDIGO, alpha=0.25, edgecolor=ACCENT_INDIGO, label=f"Empirical (Step {step_idx})")
        
        # 2. Compute theoretical transition PDF
        theta = self.theta_var.get()
        mu = self.mu_var.get()
        sigma = self.sigma_var.get()
        x0 = self.x0_var.get()
        x0_mode = self.x0_mode_var.get()
        
        x_min = slice_values.min() - 1.0
        x_max = slice_values.max() + 1.0
        x_grid = np.linspace(x_min, x_max, 200)
        
        # Use generalized transition formula depending on x0 initialization mode
        if x0_mode == "stationary" and theta > 0:
            # If initial state is already stationary, the transition PDF is the stationary distribution
            pdf = get_theoretical_stationary_pdf(x_grid, theta, mu, sigma)
        else:
            # Fixed initial state
            pdf = get_theoretical_transition_pdf(x_grid, t_val, x0, theta, mu, sigma)
            
        ax.plot(x_grid, pdf, color=ACCENT_CYAN, linewidth=2.5, label=f"Transition Density p(x,t|X₀)")
        
        # Annotate theoretical mean and SD at this slice
        if x0_mode == "stationary" and theta > 0:
            mean_t = mu
            sd_t = np.sqrt(sigma**2 / (2.0 * theta))
        else:
            mean_t, sd_t = get_theoretical_transition_stats(t_val, x0, theta, mu, sigma)
            
        stats_text = f"Theoretical Mean: {mean_t:.3f}\nTheoretical SD: {sd_t:.3f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=BG_CARD, edgecolor=BORDER_COLOR, alpha=0.8),
                fontsize=8)
        
        ax.set_title(f"Transition Density Evolution: t = {t_val:.3f} s", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel(f"State Value (X_t)", fontsize=9)
        ax.set_ylabel("Probability Density", fontsize=9)
        ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
        
        self.canvas_slice.draw()

    def save_to_comparison(self):
        """Save the current simulation run data to memory as Comparison Run (Run A)."""
        if self.time_grid is None:
            return
            
        self.comparison_data = {
            "theta": self.theta_var.get(),
            "mu": self.mu_var.get(),
            "sigma": self.sigma_var.get(),
            "time_grid": self.time_grid.copy(),
            "paths": self.paths.copy(),
            "mean_path": self.mean_path.copy(),
            "sd_path": self.sd_path.copy()
        }
        
        self.comp_lbl.pack(anchor='w', pady=(2, 0))
        self.clear_comp_btn.config(state='normal')
        
        # Redraw plots to overlay comparison lines
        self.plot_active_tab()
        messagebox.showinfo("Comparison Saved", "Current simulation saved to memory as 'Comparison Run A'. Changing parameters will now draw overlays.")

    def clear_comparison(self):
        """Erase comparison data and hide overlays."""
        self.comparison_data = None
        self.comp_lbl.pack_forget()
        self.clear_comp_btn.config(state='disabled')
        
        # Redraw active plot
        self.plot_active_tab()

    def export_csv(self):
        """Export trajectories and descriptive statistics to a CSV file."""
        if self.time_grid is None or self.paths is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Export Simulation Data"
        )
        
        if not file_path:
            return
            
        try:
            num_paths, steps_plus_1 = self.paths.shape
            
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                # Write CSV header row
                header = ["Step", "Time", "Ensemble_Mean", "Ensemble_SD"]
                for p in range(num_paths):
                    header.append(f"Trajectory_{p + 1}")
                writer.writerow(header)
                
                # Write data rows
                for i in range(steps_plus_1):
                    row = [
                        i,
                        f"{self.time_grid[i]:.6f}",
                        f"{self.mean_path[i]:.6f}",
                        f"{self.sd_path[i]:.6f}"
                    ]
                    for p in range(num_paths):
                        row.append(f"{self.paths[p, i]:.6f}")
                    writer.writerow(row)
                    
            messagebox.showinfo("Export Successful", f"Simulation data successfully exported to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export CSV: {str(e)}")

    def save_active_plot(self):
        """Export the currently selected tab's Matplotlib chart as a high-resolution image."""
        tab_idx = self.notebook.index(self.notebook.select())
        
        if tab_idx == 0:
            fig = self.fig_traj
            name_hint = "ou_trajectories"
        elif tab_idx == 1:
            fig = self.fig_stat
            name_hint = "ou_stationary_distribution"
        elif tab_idx == 2:
            fig = self.fig_acf
            name_hint = "ou_autocorrelation"
        elif tab_idx == 3:
            fig = self.fig_slice
            name_hint = "ou_transition_density_slice"
        else:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("SVG Vector", "*.svg")],
            title="Save Active Figure",
            initialfile=name_hint
        )
        
        if not file_path:
            return
            
        try:
            # Temporarily configure dpi=300 for high resolution export
            fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=BG_DARK)
            messagebox.showinfo("Plot Saved", f"Figure successfully exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Failed to save plot image: {str(e)}")
