import numpy as np

# Given parameters
F = 0.22
weight = 80  # kg
CL_per_kg = 0.40  # L/h/kg
Vd_per_kg = 2.65  # L/kg
Cmax1 = 5.3
Tmax1 = 2
HL = 3.3

# Calculate total CL and Vd
CL = CL_per_kg * weight  # Total clearance in L/h
Vd = Vd_per_kg * weight  # Total volume of distribution in L

# Calculate elimination rate constant k_el
k_el = CL / Vd  # Elimination rate constant (1/h)
print(f"Elimination rate constant (k_el): {k_el:.4f} h^-1")

# Calculate absorption rate constant ka
ka = np.log(2) / HL
print(f"Absorption rate constant (ka): {ka:.4f} h^-1")
