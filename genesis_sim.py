import numpy as np
import matplotlib.pyplot as plt

"""
GENESIS SIMULATOR: Love-OS Evolutionary Dynamics
------------------------------------------------
This script simulates the evolution of agents based on the Love-OS framework.
It compares two scenarios:
1. Darwinian (Standard): Bandwidth (B) is fixed. Survival of the fittest.
2. Love-OS (Expanded): Bandwidth (B) and Alignment (R) evolve endogenously.

Key Variables:
- x: Frequency of species
- R: Alignment (Coherence)
- B: Bandwidth (Information Processing Capacity)
- M: Meaning Density
- C: Friction/Noise
"""

class LoveOS_Evolution:
    def __init__(self, mode='love_os'):
        self.mode = mode
        
        # --- System Parameters ---
        self.K = 3            # Number of species/strategies
        self.dt = 0.05        # Time step
        self.T_max = 1000     # Total simulation steps
        self.Phi = 1.2        # Energy Influx (Free Energy)
        
        # --- Weights for Fitness Function ---
        # F = theta_R*R + theta_B*B + theta_M*M - theta_C*C
        self.theta = {'R': 1.0, 'B': 1.5, 'M': 0.5, 'C': 0.8}
        
        # --- Dynamics Coefficients ---
        # dR = alpha*Phi*(1-R) + beta*B*(1-R) ...
        self.coeffs_R = {'alpha': 0.5, 'beta': 0.8, 'gamma': 0.4}
        # dB = rho_R*R ...
        self.coeffs_B = {'rho_R': 0.4, 'rho_Phi': 0.1, 'rho_C': 0.3, 'lambda': 0.02}
        
        # --- Criticality (Phase Transition) ---
        self.criticality = {'kappa': 5.0, 'tau_c': 1.0, 'gain_low': 1.0, 'gain_high': 3.0}

        # --- Initial State ---
        self.x = np.ones(self.K) / self.K  # Equal distribution
        self.R = np.array([0.1, 0.2, 0.15]) # Low initial alignment
        self.B = np.array([0.1, 0.1, 0.1])  # Low initial bandwidth
        
        # Fixed Environment Profiles per species (Genetic predisposition)
        self.M = np.array([0.5, 1.0, 1.5])  # Meaning potential
        self.C = np.array([0.8, 0.5, 0.3])  # Friction/Noise (Species 3 is efficient)

        # Mutation Matrix (Q)
        mu = 0.01
        self.Q = (1 - mu) * np.eye(self.K) + (mu / (self.K - 1)) * (np.ones((self.K, self.K)) - np.eye(self.K))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def step(self):
        # 1. Calculate Fitness Core
        F_core = (self.theta['R'] * self.R + 
                  self.theta['B'] * self.B + 
                  self.theta['M'] * self.M - 
                  self.theta['C'] * self.C)

        # 2. Phase Transition (The "Awakening" Gain)
        # Threshold depends on R (Alignment) and B (Bandwidth)
        threshold_val = self.R + self.B - 0.5 * self.C
        s = self.sigmoid(self.criticality['kappa'] * (threshold_val - self.criticality['tau_c']))
        gain = self.criticality['gain_low'] * (1 - s) + self.criticality['gain_high'] * s
        
        # Apply Gain to Fitness
        F_effective = F_core * gain

        # 3. Convert to Positive Fitness (Boltzmann/Softmax style for stability)
        # beta is selection pressure
        beta_sel = 2.0
        F_prob = np.exp(beta_sel * F_effective)

        # 4. Replicator-Mutator Dynamics
        # Growth = (Frequency * Fitness) distributed by Mutation Matrix
        growth = (self.x * F_prob) @ self.Q
        phi_bar = growth.sum()
        self.x = growth / phi_bar  # Normalize

        # 5. Endogenous Evolution (The Love-OS Difference)
        if self.mode == 'love_os':
            # Alignment (R) Dynamics: Reinforced by Energy and Bandwidth
            dR = (self.coeffs_R['alpha'] * self.Phi * (1 - self.R) + 
                  self.coeffs_R['beta'] * self.B * (1 - self.R) - 
                  self.coeffs_R['gamma'] * self.C * self.R)
            
            # Bandwidth (B) Dynamics: Driven by Alignment
            dB = (self.coeffs_B['rho_R'] * self.R + 
                  self.coeffs_B['rho_Phi'] * self.Phi - 
                  self.coeffs_B['rho_C'] * self.C - 
                  self.coeffs_B['lambda'] * self.B)
            
            self.R = np.clip(self.R + self.dt * dR, 0, 1)
            self.B = np.clip(self.B + self.dt * dB, 0.01, 5.0) # Bandwidth can grow > 1
        
        elif self.mode == 'darwinian':
            # In standard model, Capacity (B) and Alignment (R) are static traits
            pass

        return self.x, self.R.mean(), self.B.mean(), s.mean()

# --- Main Execution ---

def run_scenario(mode_name):
    sim = LoveOS_Evolution(mode=mode_name)
    history = {'x': [], 'R': [], 'B': [], 's': []}
    
    for _ in range(sim.T_max):
        x, R, B, s = sim.step()
        history['x'].append(x)
        history['R'].append(R)
        history['B'].append(B)
        history['s'].append(s)
        
    return history

# Run both scenarios
hist_love = run_scenario('love_os')
hist_darwin = run_scenario('darwinian')

# --- Plotting ---
t_axis = np.arange(1000) * 0.05
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Bandwidth Evolution Comparison
axes[0].plot(t_axis, hist_love['B'], label='Love-OS (Evolving Bandwidth)', color='#d62728', linewidth=2.5)
axes[0].plot(t_axis, hist_darwin['B'], label='Standard (Fixed Bandwidth)', color='#7f7f7f', linestyle='--')
axes[0].set_title('Evolution of Processing Capacity (Bandwidth)', fontsize=12)
axes[0].set_xlabel('Time (generations)')
axes[0].set_ylabel('System Bandwidth (B)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Phase Transition (Awakening) in Love-OS
s_curve = np.array(hist_love['s'])
axes[1].plot(t_axis, hist_love['R'], label='Alignment (R)', color='#1f77b4')
axes[1].plot(t_axis, s_curve, label='Phase Transition State (s)', color='#ff7f0e', linewidth=2)
axes[1].fill_between(t_axis, 0, s_curve, color='#ff7f0e', alpha=0.2)
axes[1].set_title('Love-OS Internal Dynamics: Alignment & Criticality', fontsize=12)
axes[1].set_xlabel('Time (generations)')
axes[1].set_ylabel('Index Value [0-1]')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('genesis_evolution_plot.png')
print("Simulation complete. 'genesis_evolution_plot.png' generated.")
