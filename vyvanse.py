import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Model Parameters ---
D_total = 30.0       # Total oral dose of lisdexamfetamine dimesilate (mg)
F_conv = 0.3         # Fraction of lisdexamfetamine dimesilate converted to dexamphetamine (30%)
ka = 0.7             # Absorption rate constant from the gut (hr^-1)

# Lisdexamfetamine dimesilate elimination (parent drug): assumed half-life = 0.5 hr
t_half_L = 0.5       
k_L = np.log(2) / t_half_L  # ~1.386 hr^-1

# Dexamphetamine elimination (active metabolite): assumed half-life ~10 hr
t_half_D = 10.0      
k_D = np.log(2) / t_half_D  # ~0.0693 hr^-1

Vd = 200.0           # Volume of distribution (L)

# --- ODE System Definition ---
# y = [A, L, D] where:
#   A: Amount in the gut (mg)
#   L: Amount of lisdexamfetamine dimesilate in plasma (mg)
#   D: Amount of dexamphetamine in plasma (mg)
def odes(y, t, ka, k_L, k_D, F_conv):
    A, L, D = y
    dA_dt = -ka * A                     # Absorption from the gut
    dL_dt = ka * A - k_L * L              # Lisdexamfetamine dimesilate: input from absorption, then rapidly eliminated
    dD_dt = F_conv * k_L * L - k_D * D    # Dexamphetamine: produced via conversion and eliminated slowly
    return [dA_dt, dL_dt, dD_dt]

# Initial conditions: all dose in the gut; none in plasma compartments
y0 = [D_total, 0.0, 0.0]

# --- Time Simulation ---
t = np.linspace(0, 48, 480)  # Simulate from 0 to 48 hours

# Solve the system of ODEs
sol = odeint(odes, y0, t, args=(ka, k_L, k_D, F_conv))
A_sol = sol[:, 0]  # Amount in gut
L_sol = sol[:, 1]  # Amount of lisdexamfetamine dimesilate in plasma
D_sol = sol[:, 2]  # Amount of dexamphetamine in plasma

# Convert amounts to plasma concentrations (mg/L) and then to ng/mL:
# (1 mg/L = 1000 ng/mL)
C_lisdex_mg_L = L_sol / Vd
C_dex_mg_L = D_sol / Vd
C_lisdex_ng_mL = C_lisdex_mg_L * 1000
C_dex_ng_mL = C_dex_mg_L * 1000

# Find Cmax and Tmax for each curve
max_index_lisdex = np.argmax(C_lisdex_ng_mL)
max_index_dex = np.argmax(C_dex_ng_mL)
Cmax_lisdex = C_lisdex_ng_mL[max_index_lisdex]
Tmax_lisdex = t[max_index_lisdex]
Cmax_dex = C_dex_ng_mL[max_index_dex]
Tmax_dex = t[max_index_dex]

print("Lisdexamfetamine dimesilate: Cmax = {:.2f} ng/mL at t = {:.2f} hr".format(Cmax_lisdex, Tmax_lisdex))
print("Dexamphetamine: Cmax = {:.2f} ng/mL at t = {:.2f} hr".format(Cmax_dex, Tmax_dex))

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(t, C_lisdex_ng_mL, label='Lisdexamfetamine dimesilate (ng/mL)', lw=2)
plt.plot(t, C_dex_ng_mL, label='Dexamphetamine (ng/mL)', lw=2)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Plasma Concentration (ng/mL)', fontsize=12)
plt.title('Plasma Concentration-Time Profiles\nfor Lisdexamfetamine Dimesilate and Dexamphetamine', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
