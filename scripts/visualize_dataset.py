import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import Dict, List, Optional

NUM_BINS = 4
BIN_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
OBJECT_ALPHA = 0.7
BIN_SIZE = 8
OBJECT_SIZE = 5


class DatasetVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Visualizer")
        self.root.geometry("1200x800")
        
        self.dataset: Optional[List[Dict]] = None
        self.current_scenario_idx = 0
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        self.setup_ui()
        self.load_default_dataset()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(control_frame, text="Dataset:").grid(row=0, column=0, padx=(0, 5))
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(control_frame, textvariable=self.dataset_var, 
                                          state="readonly", width=30)
        self.dataset_combo.grid(row=0, column=1, padx=5)
        self.dataset_combo.bind("<<ComboboxSelected>>", self.on_dataset_selected)
        
        ttk.Button(control_frame, text="Browse...", 
                  command=self.browse_dataset).grid(row=0, column=2, padx=5)
        
        ttk.Label(control_frame, text="Scenario:").grid(row=0, column=3, padx=(20, 5))
        self.scenario_var = tk.StringVar()
        self.scenario_spinbox = ttk.Spinbox(control_frame, from_=0, to=0, 
                                           textvariable=self.scenario_var, width=10,
                                           command=self.on_scenario_changed)
        self.scenario_spinbox.grid(row=0, column=4, padx=5)
        self.scenario_var.trace('w', lambda *args: self.on_scenario_changed())
        
        ttk.Label(control_frame, text="/").grid(row=0, column=5, padx=2)
        self.total_scenarios_label = ttk.Label(control_frame, text="0")
        self.total_scenarios_label.grid(row=0, column=6, padx=2)
        
        self.show_path_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Path", 
                       variable=self.show_path_var,
                       command=self.update_visualization).grid(row=0, column=7, padx=(20, 5))
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        legend_frame = ttk.Frame(main_frame)
        legend_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Label(legend_frame, text="Bin Colors:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5)
        for i in range(NUM_BINS):
            color_frame = ttk.Frame(legend_frame)
            color_frame.grid(row=0, column=i+1, padx=5)
            canvas = tk.Canvas(color_frame, width=30, height=20, highlightthickness=0)
            canvas.pack()
            canvas.create_rectangle(0, 0, 30, 20, fill=BIN_COLORS[i], outline="black")
            ttk.Label(color_frame, text=f"Bin {i}").pack()
    
    def load_default_dataset(self):
        if os.path.exists(self.data_dir):
            datasets = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
            datasets.sort()
            self.dataset_combo['values'] = datasets
            if datasets:
                self.dataset_combo.current(0)
                self.on_dataset_selected()
    
    def browse_dataset(self):
        filename = filedialog.askopenfilename(
            initialdir=self.data_dir,
            title="Select Dataset",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            rel_path = os.path.relpath(filename, self.data_dir)
            if rel_path in self.dataset_combo['values']:
                self.dataset_combo.current(list(self.dataset_combo['values']).index(rel_path))
            else:
                self.dataset_combo['values'] = list(self.dataset_combo['values']) + [rel_path]
                self.dataset_combo.current(len(self.dataset_combo['values']) - 1)
            self.on_dataset_selected()
    
    def on_dataset_selected(self, event=None):
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            return
        
        filepath = os.path.join(self.data_dir, dataset_name)
        if not os.path.exists(filepath):
            filepath = dataset_name
        
        try:
            with open(filepath, 'r') as f:
                self.dataset = json.load(f)
            
            self.current_scenario_idx = 0
            max_idx = len(self.dataset) - 1
            self.scenario_spinbox.config(from_=0, to=max_idx)
            self.scenario_var.set("0")
            self.total_scenarios_label.config(text=str(len(self.dataset)))
            self.update_visualization()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    
    def on_scenario_changed(self):
        try:
            idx = int(self.scenario_var.get())
            if self.dataset and 0 <= idx < len(self.dataset):
                self.current_scenario_idx = idx
                self.update_visualization()
        except ValueError:
            pass
    
    def update_visualization(self):
        if not self.dataset or self.current_scenario_idx >= len(self.dataset):
            return
        
        scenario = self.dataset[self.current_scenario_idx]
        self.ax.clear()
        
        objects = np.array(scenario["objects"])
        types = np.array(scenario["types"])
        bins = np.array(scenario["bins"])
        start = np.array(scenario["start"])
        
        for bin_idx in range(NUM_BINS):
            bin_pos = bins[bin_idx]
            self.ax.scatter(bin_pos[0], bin_pos[1], s=BIN_SIZE*100, 
                          c=BIN_COLORS[bin_idx], marker='s', 
                          edgecolors='black', linewidths=2, 
                          label=f'Bin {bin_idx}', zorder=3)
        
        for obj_type in range(NUM_BINS):
            mask = types == obj_type
            if np.any(mask):
                self.ax.scatter(objects[mask, 0], objects[mask, 1], 
                              s=OBJECT_SIZE*20, c=BIN_COLORS[obj_type], 
                              alpha=OBJECT_ALPHA, edgecolors='black', 
                              linewidths=0.5, zorder=2)
        
        self.ax.scatter(start[0], start[1], s=150, c='black', 
                       marker='*', edgecolors='white', linewidths=1, 
                       label='Start', zorder=4)
        
        if self.show_path_var.get() and "greedy_sequence" in scenario:
            seq = scenario["greedy_sequence"]
            path_x = [start[0]]
            path_y = [start[1]]
            
            for obj_idx in seq:
                obj_pos = objects[obj_idx]
                path_x.append(obj_pos[0])
                path_y.append(obj_pos[1])
                
                bin_idx = types[obj_idx]
                bin_pos = bins[bin_idx]
                path_x.append(bin_pos[0])
                path_y.append(bin_pos[1])
            
            self.ax.plot(path_x, path_y, 'k--', alpha=0.3, linewidth=1, zorder=1)
        
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        
        metadata = scenario.get("metadata", {})
        title = f"Scenario {self.current_scenario_idx}"
        if "object_count" in metadata:
            title += f" | {metadata['object_count']} objects"
        if "greedy_cost" in scenario:
            title += f" | Cost: {scenario['greedy_cost']:.2f}"
        self.ax.set_title(title)
        
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = DatasetVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

