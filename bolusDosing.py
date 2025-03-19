import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Fixed parameters
Vd = 200.0            # Volume of distribution (L)
D_load = 100.0        # Fixed IV bolus loading dose (mg) – assumed too high
MTC = 100           # Minimum Toxic Concentration (ng/mL)
t_sim = 24.0          # Total simulation time (hours)
dt = 0.1              # Time step for simulation

def compute_concentration(D_maint, T_interval, t_half):
    """
    Computes the total plasma concentration (ng/mL) over time after a loading dose at time 0 
    and repeated maintenance doses given every T_interval hours.
    
    Parameters:
        D_maint    : Maintenance dose (mg)
        T_interval : Dosing interval (hours)
        t_half     : Elimination half-life (hours)
    
    Returns:
        time_points: Array of time points (hours)
        concentration_ng_per_mL: Corresponding plasma concentration (ng/mL)
    """
    # Compute elimination rate constant based on half-life
    k = np.log(2) / t_half  # Elimination rate constant (hr^-1)
    
    # Create simulation time vector from 0 to t_sim
    time_points = np.arange(0, t_sim + dt, dt)
    
    # Define maintenance dose times (starting at T_interval)
    maintenance_times = np.arange(T_interval, t_sim + dt, T_interval)
    
    # Combine loading dose at time 0 with maintenance doses
    dose_times = np.concatenate(([0.0], maintenance_times))
    dose_values = np.concatenate(([D_load], np.full(len(maintenance_times), D_maint)))
    
    # Calculate the total concentration at each time point by summing contributions
    concentration_mg_per_L = np.zeros_like(time_points)
    for i, t in enumerate(time_points):
        C_total = 0.0
        for t_d, D in zip(dose_times, dose_values):
            if t >= t_d:
                C_total += (D / Vd) * np.exp(-k * (t - t_d))
        concentration_mg_per_L[i] = C_total
    
    # Convert mg/L to ng/mL (1 mg/L = 1000 ng/mL)
    concentration_ng_per_mL = concentration_mg_per_L * 1000
    return time_points, concentration_ng_per_mL

# Initial slider values
initial_D_maint = 20.0   # mg
initial_T_interval = 2.0 # hours
initial_t_half = 1.0     # hours

# Compute initial concentration profile
time_points, concentration = compute_concentration(initial_D_maint, initial_T_interval, initial_t_half)

# Create the figure and main plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.35)  # leave space at the bottom for sliders
line, = ax.plot(time_points, concentration, lw=2, label='Plasma Concentration (ng/mL)')
ax.axhline(MTC, color='r', linestyle='--', label=f'MTC ({MTC} ng/mL)')
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Concentration (ng/mL)', fontsize=12)
ax.set_title('IV Bolus Dosing: Loading Dose and Maintenance Doses', fontsize=14)
ax.legend()
ax.grid(True)

# Create slider axes below the main plot
axcolor = 'lightgoldenrodyellow'
ax_D_maint = plt.axes([0.15, 0.25, 0.7, 0.03], facecolor=axcolor)
ax_T_interval = plt.axes([0.15, 0.18, 0.7, 0.03], facecolor=axcolor)
ax_t_half = plt.axes([0.15, 0.11, 0.7, 0.03], facecolor=axcolor)

# Create the sliders with precise increments
slider_D_maint = Slider(ax_D_maint, 'Maint. Dose (mg)', 0, 50, valinit=initial_D_maint, valstep=0.1)
slider_T_interval = Slider(ax_T_interval, 'Dosing Interval (hr)', 0.5, t_sim, valinit=initial_T_interval, valstep=0.1)
slider_t_half = Slider(ax_t_half, 'Elim. Half-Life (hr)', 0.5, 5, valinit=initial_t_half, valstep=0.1)

# Update function: recalculate and update the plot when slider values change
def update(val):
    D_maint = slider_D_maint.val
    T_interval = slider_T_interval.val
    t_half = slider_t_half.val
    time_points, concentration = compute_concentration(D_maint, T_interval, t_half)
    line.set_data(time_points, concentration)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# Connect the update function to each slider
slider_D_maint.on_changed(update)
slider_T_interval.on_changed(update)
slider_t_half.on_changed(update)

plt.show()
