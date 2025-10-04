
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.interpolate import CubicSpline

KW_PER_HP = 1 / 1.34102
HP_PER_KW = 1.34102
PS_PER_KW = 1.35962
LBFT_PER_NM = 0.737562
NM_PER_LBFT = 1 / LBFT_PER_NM

def nm_to_lbft(nm: float) -> float:
    return nm * LBFT_PER_NM

def lbft_to_nm(lbft: float) -> float:
    return lbft * NM_PER_LBFT

def kw_to_hp(kw: float) -> float:
    return kw * HP_PER_KW

def kw_to_ps(kw: float) -> float:
    return kw * PS_PER_KW

def hp_to_kw(hp: float) -> float:
    return hp * KW_PER_HP

def ps_to_kw(ps: float) -> float:
    return ps / PS_PER_KW

def power_kw_from_nm_rpm(torque_nm: float, rpm: float) -> float:
    # 9549 is the constant for kW = (Nm * RPM) / 9549
    return (torque_nm * rpm) / 9549.0

class TorquePowerTool:
    """
    Fixed + English version.
    Key fixes:
      1) Internal torque is **always** stored in Nm to avoid unit drift when switching units.
      2) Power is recomputed for **all points** whenever torque or units change.
      3) The power curve for a flat torque line is computed directly from torque via P = T * rpm / 9549
         (spline in torque[Nm] first, then derive power). This guarantees a straight, monotonically
         increasing power line for constant torque.
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Torque & Power Curve Tool")
        self.root.geometry("1200x800")
        
        # Settings
        self.rpm_step = tk.IntVar(value=500)
        self.max_rpm = tk.IntVar(value=8000)
        self.torque_unit = tk.StringVar(value="Nm")  # display unit
        self.power_unit = tk.StringVar(value="kW")   # display unit
        
        # Data (torque stored internally in Nm at all times)
        self.rpm_values = []
        self.torque_values_nm = []       # internal canonical storage
        self.power_values_display = []    # cached in selected display unit for labels
        self.torque_entries = []          # [(tk.DoubleVar, Entry), ...]
        self.torque_sliders = []
        self.torque_labels = []
        self.power_labels = []
        
        # UI
        self.create_widgets()
        self.update_rpm_range()
    
    # ---------------- UI Building ----------------
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls", width=400)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        chart_frame = ttk.LabelFrame(main_frame, text="Torque & Power Curves")
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_control_panel(control_frame)
        self.setup_chart_panel(chart_frame)
    
    def setup_control_panel(self, parent):
        # RPM Settings
        rpm_frame = ttk.LabelFrame(parent, text="RPM Settings")
        rpm_frame.pack(fill=tk.X, pady=(0, 10))
        
        step_frame = ttk.Frame(rpm_frame)
        step_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(step_frame, text="Step:").pack(side=tk.LEFT)
        ttk.Entry(step_frame, textvariable=self.rpm_step, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(step_frame, text="RPM").pack(side=tk.LEFT)
        
        max_frame = ttk.Frame(rpm_frame)
        max_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(max_frame, text="Max RPM:").pack(side=tk.LEFT)
        ttk.Entry(max_frame, textvariable=self.max_rpm, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(max_frame, text="RPM").pack(side=tk.LEFT)
        
        ttk.Button(rpm_frame, text="Apply", command=self.update_rpm_range).pack(pady=5)
        
        # Units
        unit_frame = ttk.LabelFrame(parent, text="Units")
        unit_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(unit_frame, text="Torque:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        torque_combo = ttk.Combobox(unit_frame, textvariable=self.torque_unit, 
                                    values=["Nm", "lb-ft"], state="readonly", width=10)
        torque_combo.grid(row=0, column=1, padx=5, pady=5)
        torque_combo.bind('<<ComboboxSelected>>', self.on_unit_changed)
        
        ttk.Label(unit_frame, text="Power:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        power_combo = ttk.Combobox(unit_frame, textvariable=self.power_unit, 
                                   values=["kW", "hp", "ps"], state="readonly", width=10)
        power_combo.grid(row=1, column=1, padx=5, pady=5)
        power_combo.bind('<<ComboboxSelected>>', self.on_unit_changed)
        
        # Sliders
        slider_frame = ttk.LabelFrame(parent, text="Torque / Power Adjustments")
        slider_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        canvas = tk.Canvas(slider_frame)
        scrollbar = ttk.Scrollbar(slider_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.sliders_frame = self.scrollable_frame
        
        # Output
        output_frame = ttk.LabelFrame(parent, text="Torque Table Output")
        output_frame.pack(fill=tk.X)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=8, width=50)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Button(output_frame, text="Copy Torque Table", command=self.copy_torque_table).pack(pady=5)
    
    def setup_chart_panel(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ax.set_xlabel("RPM")
        self.ax.set_ylabel("Torque (Nm)", color='blue')
        self.ax.tick_params(axis='y', labelcolor='blue')
        
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel("Power (kW)", color='red')
        self.ax2.tick_params(axis='y', labelcolor='red')
        
        self.torque_line, = self.ax.plot([], [], 'b-', label="Torque", linewidth=2)
        self.power_line, = self.ax2.plot([], [], 'r-', label="Power", linewidth=2)
        
        self.torque_points, = self.ax.plot([], [], 'bo', markersize=4)
        self.power_points, = self.ax2.plot([], [], 'ro', markersize=4)
        
        self.ax.legend(loc="upper left")
        self.ax2.legend(loc="upper right")
        self.fig.tight_layout()
    
    # ---------------- Event Handlers ----------------
    def on_unit_changed(self, event=None):
        # Do not modify internal torque in Nm; just refresh display and recompute power
        self.recompute_all_power_display()
        self.update_display()
        self.update_chart()
    
    def update_rpm_range(self):
        # Clear existing slider UI
        for widget in self.sliders_frame.winfo_children():
            widget.destroy()
        
        self.rpm_values = []
        self.torque_values_nm = []
        self.power_values_display = []
        self.torque_sliders = []
        self.torque_labels = []
        self.power_labels = []
        self.torque_entries = []
        
        step = self.rpm_step.get()
        max_rpm = self.max_rpm.get()
        
        for i, rpm in enumerate(range(0, max_rpm + 1, step)):
            self.rpm_values.append(rpm)
            # default torque = 100 Nm internally
            self.torque_values_nm.append(100.0)
            self.power_values_display.append(0.0)
            
            group = ttk.LabelFrame(self.sliders_frame, text=f"{rpm} RPM")
            group.pack(fill=tk.X, pady=5, padx=5)
            
            # Torque row
            torque_frame = ttk.Frame(group)
            torque_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(torque_frame, text="Torque:").pack(side=tk.LEFT)
            
            # Slider works in **current display unit**
            slider = ttk.Scale(
                torque_frame, from_=0, to=1000, orient=tk.HORIZONTAL,
                value=self._torque_display_from_nm(100.0), length=150,
                command=lambda value, idx=i: self.on_torque_slider_changed(idx, float(value))
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.torque_sliders.append(slider)
            
            # Entry reflects current display unit
            torque_var = tk.DoubleVar(value=round(self._torque_display_from_nm(100.0), 1))
            torque_entry = ttk.Entry(torque_frame, textvariable=torque_var, width=8)
            torque_entry.pack(side=tk.LEFT, padx=5)
            torque_entry.bind('<Return>', lambda e, idx=i: self.on_torque_entry_changed(idx))
            torque_entry.bind('<FocusOut>', lambda e, idx=i: self.on_torque_entry_changed(idx))
            self.torque_entries.append((torque_var, torque_entry))
            
            unit_label = ttk.Label(torque_frame, text=self.torque_unit.get())
            unit_label.pack(side=tk.LEFT, padx=5)
            self.torque_labels.append(unit_label)
            
            # Power row
            power_frame = ttk.Frame(group)
            power_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(power_frame, text="Power:").pack(side=tk.LEFT)
            
            power_label = ttk.Label(power_frame, text="0.0")
            power_label.pack(side=tk.LEFT, padx=5)
            self.power_labels.append(power_label)
            
            ttk.Label(power_frame, text=self.power_unit.get()).pack(side=tk.LEFT)
        
        # Now compute power for all with defaults
        self.recompute_all_power_display()
        self.update_display()
        self.update_chart()
    
    def on_torque_slider_changed(self, index: int, value_in_display_unit: float):
        # Convert slider value (display unit) to Nm for storage
        new_value_nm = self._nm_from_torque_display(value_in_display_unit)
        self.torque_values_nm[index] = new_value_nm
        
        # Keep entry in sync (in display unit)
        torque_var, _ = self.torque_entries[index]
        torque_var.set(round(value_in_display_unit, 1))
        
        self.recompute_single_power_display(index)
        self.update_display()
        self.update_chart()
    
    def on_torque_entry_changed(self, index: int):
        torque_var, _ = self.torque_entries[index]
        try:
            val_display = float(torque_var.get())
            # store as Nm
            self.torque_values_nm[index] = self._nm_from_torque_display(val_display)
            
            # Sync slider (slider range is fixed; cap visually at max if beyond)
            cap = 1000.0
            self.torque_sliders[index].set(val_display if val_display <= cap else cap)
            
            self.recompute_single_power_display(index)
            self.update_display()
            self.update_chart()
        except ValueError:
            # Restore previous value
            prev_display = self._torque_display_from_nm(self.torque_values_nm[index])
            torque_var.set(round(prev_display, 1))
    
    # ---------------- Computation ----------------
    def _torque_display_from_nm(self, nm_val: float) -> float:
        return nm_to_lbft(nm_val) if self.torque_unit.get() == "lb-ft" else nm_val
    
    def _nm_from_torque_display(self, display_val: float) -> float:
        return lbft_to_nm(display_val) if self.torque_unit.get() == "lb-ft" else display_val
    
    def _power_display_from_kw(self, kw_val: float) -> float:
        if self.power_unit.get() == "hp":
            return kw_to_hp(kw_val)
        elif self.power_unit.get() == "ps":
            return kw_to_ps(kw_val)
        return kw_val
    
    def recompute_single_power_display(self, index: int):
        rpm = self.rpm_values[index]
        torque_nm = self.torque_values_nm[index]
        kw = power_kw_from_nm_rpm(torque_nm, rpm)
        self.power_values_display[index] = self._power_display_from_kw(kw)
    
    def recompute_all_power_display(self):
        self.power_values_display = []
        for i in range(len(self.rpm_values)):
            rpm = self.rpm_values[i]
            torque_nm = self.torque_values_nm[i]
            kw = power_kw_from_nm_rpm(torque_nm, rpm)
            self.power_values_display.append(self._power_display_from_kw(kw))
    
    # ---------------- Display / Output ----------------
    def update_display(self):
        # Update per-row labels
        for i in range(len(self.rpm_values)):
            # torque entry shows in display unit
            disp_torque = self._torque_display_from_nm(self.torque_values_nm[i])
            torque_var, _ = self.torque_entries[i]
            torque_var.set(round(disp_torque, 1))
            
            # unit labels
            self.torque_labels[i].config(text=self.torque_unit.get())
            
            # power label in display unit
            self.power_labels[i].config(text=f"{self.power_values_display[i]:.1f}")
        
        self.update_output()
    
    def update_output(self):
        # Export torque table as pairs [rpm, torque_in_display_unit]
        lines = []
        for i, rpm in enumerate(self.rpm_values):
            disp_torque = self._torque_display_from_nm(self.torque_values_nm[i])
            lines.append(f"[{rpm}, {disp_torque:.1f}]")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "[\n " + ",\n ".join(lines) + "\n]")
    
    
    def _nice_tick_step(self, vmax: float, target_ticks: int = 8) -> float:
        if vmax <= 0:
            return 1.0
        raw = vmax / target_ticks
        power = 10 ** (int(np.floor(np.log10(raw))))
        for m in [1, 2, 2.5, 5, 10]:
            step = m * power
            if raw <= step:
                return step
        return 10 * power

    def update_chart(self):
        if len(self.rpm_values) < 2:
            return
        
        # Prepare arrays
        rpm_arr = np.array(self.rpm_values, dtype=float)
        torque_nm_arr = np.array(self.torque_values_nm, dtype=float)
        
        # Build dense RPM grid
        rpm_dense = np.linspace(float(np.min(rpm_arr)), float(np.max(rpm_arr)), 300)
        
        # Smooth torque in **Nm** to ensure power is derived from physics equation
        try:
            cs_torque_nm = CubicSpline(rpm_arr, torque_nm_arr)
            torque_nm_smooth = cs_torque_nm(rpm_dense)
        except Exception:
            # Fallback: linear
            torque_nm_smooth = np.interp(rpm_dense, rpm_arr, torque_nm_arr)
        
        # Display torque curve (convert to display unit AFTER smoothing)
        torque_display_smooth = np.array([self._torque_display_from_nm(v) for v in torque_nm_smooth])
        torque_points_display = np.array([self._torque_display_from_nm(v) for v in torque_nm_arr])
        self.torque_line.set_data(rpm_dense, torque_display_smooth)
        self.torque_points.set_data(rpm_arr, torque_points_display)
        
        # Derive power from torque (guarantees straight, monotonic line if torque is flat)
        power_kw_smooth = (torque_nm_smooth * rpm_dense) / 9549.0
        power_display_smooth = np.array([self._power_display_from_kw(v) for v in power_kw_smooth])
        power_points_display = np.array(self.power_values_display, dtype=float)
        
        self.power_line.set_data(rpm_dense, power_display_smooth)
        self.power_points.set_data(rpm_arr, power_points_display)
        
        # === Unified Y-scale for both left/right axes ===
        # Use the **display units** currently selected, so switching units will
        # rescale the axes and therefore visually change the curve shape.
        max_torque_disp = float(np.nanmax([1.0, *torque_points_display, *torque_display_smooth]))
        max_power_disp  = float(np.nanmax([1.0, *power_points_display, *power_display_smooth]))
        
        same_max = max(max_torque_disp, max_power_disp)
        same_max *= 1.10  # headroom
        
        # Compute a "nice" shared tick step and ticks
        step = self._nice_tick_step(same_max, target_ticks=8)
        # Snap the limit to a multiple of step >= same_max
        n_steps = int(np.ceil(same_max / step))
        y_max = n_steps * step
        ticks = np.arange(0.0, y_max + 0.5 * step, step)
        
        # Apply identical limits and ticks to both axes
        self.ax.set_xlim(0, float(np.max(rpm_arr)))
        self.ax.set_ylim(0, y_max)
        self.ax.set_yticks(ticks)
        self.ax2.set_ylim(0, y_max)
        self.ax2.set_yticks(ticks)
        
        # Axes labels reflect current units
        self.ax.set_ylabel(f"Torque ({self.torque_unit.get()})", color='blue')
        self.ax2.set_ylabel(f"Power ({self.power_unit.get()})", color='red')
        
        # Make sure tick label colors match axes
        self.ax.tick_params(axis='y', labelcolor='blue')
        self.ax2.tick_params(axis='y', labelcolor='red')
        
        self.canvas.draw()

    def copy_torque_table(self):
        torque_table = self.output_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(torque_table)
        messagebox.showinfo("Success", "Torque table copied to clipboard!")

def main():
    root = tk.Tk()
    app = TorquePowerTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()
