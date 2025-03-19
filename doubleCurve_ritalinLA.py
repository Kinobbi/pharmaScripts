import numpy as np
import matplotlib.pyplot as plt

def concentration_immediate(t, Dose, ka, k, Vd):
    """
    Computes the plasma concentration from the immediate-release fraction.
    
    Parameters:
        t    : time (hours)
        Dose : dose for the immediate-release fraction (mg)
        ka   : absorption rate constant (hr^-1) for immediate release
        k    : elimination rate constant (hr^-1)
        Vd   : volume of distribution (L)
        
    Returns:
        Plasma concentration at time t in mg/L.
    """
    # For t < 0, concentration is zero.
    return (Dose * ka / (Vd * (ka - k))) * (np.exp(-k * t) - np.exp(-ka * t))

def concentration_delayed(t, Dose, ka, k, Vd, lag):
    """
    Computes the plasma concentration from the delayed-release fraction.
    Absorption starts after a lag time.
    
    Parameters:
        t    : time (hours)
        Dose : dose for the delayed-release fraction (mg)
        ka   : absorption rate constant (hr^-1) for delayed release
        k    : elimination rate constant (hr^-1)
        Vd   : volume of distribution (L)
        lag  : lag time before absorption begins (hours)
    
    Returns:
        Plasma concentration at time t in mg/L (zero if t < lag).
    """
    # For vectorized computation, shift time by the lag.
    tau = np.maximum(t - lag, 0)
    conc = (Dose * ka / (Vd * (ka - k))) * (np.exp(-k * tau) - np.exp(-ka * tau))
    # Ensure that before the lag, the concentration is zero.
    conc[t < lag] = 0.0
    return conc

def total_concentration(t, total_dose, F, ka1, ka2, k, Vd, lag):
    """
    Calculates the overall plasma concentration for a double-release formulation.
    
    The total dose is adjusted by bioavailability (F) and assumed to be split equally between
    the immediate and delayed fractions.
    
    Parameters:
        t          : time (hours)
        total_dose : total administered dose (mg)
        F          : absolute bioavailability (fraction)
        ka1        : absorption rate constant for immediate release (hr^-1)
        ka2        : absorption rate constant for delayed release (hr^-1)
        k          : elimination rate constant (hr^-1)
        Vd         : volume of distribution (L)
        lag        : lag time for the delayed-release fraction (hours)
        
    Returns:
        Total plasma concentration at time t in mg/L.
    """
    # Effective dose after bioavailability and split equally:
    effective_dose = total_dose * F / 2.0
    conc_immediate = concentration_immediate(t, effective_dose, ka1, k, Vd)
    conc_delayed = concentration_delayed(t, effective_dose, ka2, k, Vd, lag)
    return conc_immediate + conc_delayed

# Main simulation parameters based on the label information:
if __name__ == "__main__":
    # Dose and patient parameters
    total_dose = 20            # mg (Ritalin LA 20 mg capsule)
    weight = 80                # kg
    F = 0.22                   # Approximate absolute bioavailability (22%)
    
    # Pharmacokinetic parameters for adult males from the document:
    Vd = 2.65 * weight         # L (using Vd ~2.65 L/kg for d-methylphenidate, the active enantiomer)
    t_half = 3.3               # hours (elimination half-life)
    k = np.log(2) / t_half     # elimination rate constant (hr^-1)
    
    # Absorption parameters chosen to match observed Tmax values:
    ka1 = 1.0                # hr^-1 for immediate release (Tmax ~2 hours)
    ka2 = 1.5                # hr^-1 for delayed release (after lag, Tmax ~1.5 hours post-lag, ~5.5 hours overall)
    lag = 4                  # hours delay for the delayed-release fraction
    
    # Create a time array for simulation (0 to 12 hours)
    t = np.linspace(0, 12, 500)
    
    # Calculate concentrations (mg/L)
    C_total_mgL = total_concentration(t, total_dose, F, ka1, ka2, k, Vd, lag)
    C_immediate_mgL = concentration_immediate(t, total_dose * F / 2.0, ka1, k, Vd)
    C_delayed_mgL = concentration_delayed(t, total_dose * F / 2.0, ka2, k, Vd, lag)
    
    # Convert concentrations to ng/mL (1 mg/L = 1000 ng/mL)
    C_total = C_total_mgL * 1000
    C_immediate = C_immediate_mgL * 1000
    C_delayed = C_delayed_mgL * 1000
    
    # Plot the results
    plt.figure(figsize=(9, 6))
    plt.plot(t, C_total, label="Total Plasma Concentration", lw=2)
    plt.plot(t, C_immediate, '--', label="Immediate-Release Fraction", lw=1.5)
    plt.plot(t, C_delayed, '--', label="Delayed-Release Fraction", lw=1.5)
    plt.axvline(x=lag, color='gray', linestyle='--', label="Lag Time (Delayed Release)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Plasma Concentration (ng/mL)")
    plt.title("Simulated Double-Release Concentration Profile of Ritalin LA\n(20 mg dose, 80 kg male)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
