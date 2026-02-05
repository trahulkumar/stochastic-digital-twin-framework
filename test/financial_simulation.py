import simpy
import random
import pandas as pd
import os

# --- OUTPUT SETUP ---
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Define the Stochastic Environment
def financial_process(env, case_id, log_list):
    # Activity A: Submission
    log_list.append([case_id, "Invoice Submitted", env.now, "System"])
    
    # Stochastic Delay (Log-Normal Distribution)
    processing_time = random.lognormvariate(mu=1, sigma=0.5) 
    yield env.timeout(processing_time)
    
    # Activity B: Risk Check (Agentic Decision)
    risk_score = random.uniform(0, 1)
    if risk_score > 0.8:
        # High Risk Path
        log_list.append([case_id, "Manual Risk Review", env.now, "Risk_Officer_01"])
        yield env.timeout(random.uniform(5, 10)) # Long delay
        if random.random() > 0.5:
             log_list.append([case_id, "Invoice Rejected", env.now, "Risk_Officer_01"])
             return # End process
    
    # Activity C: Final Approval
    log_list.append([case_id, "Invoice Approved", env.now, "Finance_VP"])

# 2. Run the Generator
env = simpy.Environment()
log_data = []
print("Starting simulation for 5000 cases...")
for i in range(5000): # Generate 5,000 unique cases
    env.process(financial_process(env, f"CASE_{i}", log_data))
    
env.run()
print("Simulation complete.")

# 3. Create the Dataset
df = pd.DataFrame(log_data, columns=["CaseID", "Activity", "Timestamp", "Resource"])

# Save to output
csv_path = os.path.join(OUTPUT_DIR, "simulation_log.csv")
df.to_csv(csv_path, index=False)
print(f"Simulation data saved to {csv_path}")

# Optional: Print first few rows
print(df.head())
