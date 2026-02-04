import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. CSV Template Generation (First time setup)
# ---------------------------------------------------------
def create_template():
    columns = [
        'Date',
        'Meaning_M (1-7)',          # Meaning Density (M_bar)
        'Alignment_R (0-1)',        # Alignment R (Purity of Attention)
        'Friction_C (1-7)',         # Friction C_bar (Interruptions/Doubt)
        'Flow_State (1-7)',         # Subjective Time Density Proxy
        'Action_Volume (1-10)',     # Magnitude of Action |dr|
        'Boundary_Note (Text)'      # Boundary Update (What was defined?)
    ]
    # Sample Data for 7 days
    data = [
        ['2023-10-01', 5, 0.6, 4, 3, 5, 'Defined project scope'],
        ['2023-10-02', 6, 0.8, 2, 5, 6, 'Aligned with key stakeholder'],
        ['2023-10-03', 4, 0.4, 5, 2, 4, 'Friction due to unplanned mtg'],
        ['2023-10-04', 7, 0.9, 1, 7, 8, 'Deep work: Core algorithm'],
        ['2023-10-05', 6, 0.8, 2, 6, 7, 'Documentation & Cleanup'],
        ['2023-10-06', 3, 0.3, 6, 2, 3, 'Recovering from fatigue'],
        ['2023-10-07', 7, 0.95, 1, 7, 9, 'Full integration achieved'],
    ]
    return pd.DataFrame(data, columns=columns)

# ---------------------------------------------------------
# 2. The Kernel Calculation (Love-OS Logic)
# ---------------------------------------------------------
def calculate_love_integral(df):
    # Parameters (Tuning the sensitivity)
    alpha_M, alpha_R, alpha_C = 1.0, 1.2, 0.5
    
    results = []
    accumulated_love = 0
    
    for index, row in df.iterrows():
        # Inputs & Normalization
        M = row['Meaning_M (1-7)'] / 7.0   # Normalize to 0-1
        R = row['Alignment_R (0-1)']       # Already 0-1
        C = row['Friction_C (1-7)'] / 7.0  # Normalize to 0-1
        dr = row['Action_Volume (1-10)']   # Magnitude of action |dr|
        
        # 1. Calculate Instantaneous LoveForce Density (L)
        # L = (Meaning * Alignment) - Friction
        # Logic: We treat (M * R) as the directional alignment (cos theta) of the vector field.
        L_density = (alpha_M * M + alpha_R * R) - (alpha_C * C)
        
        # 2. Calculate Daily Work (Integration step)
        # dI = L_density * |dr| (Path Integral)
        daily_integral = L_density * dr
        
        # 3. Accumulate (Time Integration)
        accumulated_love += daily_integral
        
        # 4. Phase Transition Check (Criticality)
        # If order parameter (theta) is high enough, the S-curve jumps (Awakening)
        theta = M + R - C
        # Sigmoid function for criticality detection
        criticality = 1 / (1 + np.exp(-5 * (theta - 0.5))) 
        
        results.append({
            'Date': row['Date'],
            'Daily_Integral': daily_integral,
            'Total_Integrated_Love': accumulated_love,
            'Criticality_S': criticality
        })
        
    return pd.DataFrame(results)

# ---------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------
def visualize_dashboard(df_log, df_calc):
    # Set style for cleaner look
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    dates = pd.to_datetime(df_calc['Date'])
    
    # Plot 1: The Accumulation (Integral) - The "Mountain of Order"
    ax1.fill_between(dates, df_calc['Total_Integrated_Love'], color='skyblue', alpha=0.4, label='Integrated Love (Accumulation)')
    ax1.plot(dates, df_calc['Total_Integrated_Love'], color='blue', linewidth=2)
    ax1.set_ylabel('Total Integrated Order', color='blue', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot 2: Daily Performance (Bar) - "Work Done Today"
    ax2 = ax1.twinx()
    # Green for positive work (Alignment), Red for negative work (Friction/Entropy)
    colors = ['green' if x > 0 else 'red' for x in df_calc['Daily_Integral']]
    ax2.bar(dates, df_calc['Daily_Integral'], color=colors, alpha=0.6, width=0.4, label='Daily Integration')
    ax2.set_ylabel('Daily Work (M $\cdot$ dr)', color='green', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Criticality Markers (The "Zone" Indicator)
    for i, row in df_calc.iterrows():
        if row['Criticality_S'] > 0.8:
            ax1.text(dates[i], row['Total_Integrated_Love'], '★', fontsize=18, color='gold', ha='center', 
                     path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground="black")])

    ax1.set_title('Love-Integration Dashboard: Tracking the Accumulation of Order', fontsize=14, pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# --- Execution ---
if __name__ == "__main__":
    df_raw = create_template()
    df_processed = calculate_love_integral(df_raw)
    visualize_dashboard(df_raw, df_processed)
