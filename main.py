#!/usr/bin/env python3
"""
Main entry point for the Ornstein-Uhlenbeck Process Visualizer.
Launches the Tkinter application loop.
"""

import tkinter as tk
from gui import SimulationGUI

def main():
    # Create the Tkinter root window
    root = tk.Tk()
    
    # Initialize the GUI application
    app = SimulationGUI(root)
    
    # Start the Tkinter event mainloop
    root.mainloop()

if __name__ == "__main__":
    main()
