import simpy
import random
import pandas as pd
import numpy as np
import os

# --- OUTPUT SETUP ---
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CONFIGURATION (The "Physics" of the Digital Twin) ---
NUM_CASES = 1000            # Total loan applications to simulate
INTER_ARRIVAL_MEAN = 10     # Average minutes between new applications
RANDOM_SEED = 42            # For reproducibility (Crucial for INFORMS)

# Resource Capacities (Bottlenecks)
NUM_CLERKS = 3              # Intake Clerks
NUM_UNDERWRITERS = 2        # Credit Underwriters (Intentionally scarce to create queues)
NUM_VPS = 1                 # VP of Finance (For high-value approvals)

# Data Containers
event_log = []

def log_event(env, case_id, activity, resource):
    """Helper to append event to the list"""
    event_log.append({
        'CaseID': case_id,
        'Activity': activity,
        'Timestamp': env.now,
        'Resource': resource
    })

def loan_process(env, case_id, clerk, underwriter, vp):
    # --- STEP 1: Application Receipt ---
    # Case arrives
    arrival_time = env.now
    
    # Request a Clerk
    with clerk.request() as req:
        yield req # Wait for clerk availability
        
        log_event(env, case_id, "Application Received", "System_Intake")
        
        # Determine Loan Amount (Contextual Data)
        # Using Pareto distribution to simulate few very high value loans
        loan_amount = (np.random.pareto(a=3) + 1) * 50000 
        
        # Processing: Data Entry (Fast, low variance)
        yield env.timeout(random.normalvariate(5, 1)) 
        log_event(env, case_id, "Data Check Complete", f"Clerk_{random.randint(1, NUM_CLERKS)}")

    # --- STEP 2: Credit Underwriting (The Bottleneck) ---
    with underwriter.request() as req:
        yield req # Wait for underwriter availability (Queue forms here)
        
        log_event(env, case_id, "Underwriting Started", "Credit_Dept")
        
        # Processing time depends on complexity (Loan Amount)
        # Log-normal distribution: Most are fast, some take forever
        base_time = 20
        if loan_amount > 150000:
            base_time = 60 # Harder to underwrite big loans
            
        proc_time = random.lognormvariate(np.log(base_time), 0.5)
        yield env.timeout(proc_time)
        
        decision_prob = random.random()
        
        if decision_prob < 0.15:
            log_event(env, case_id, "Application Rejected", f"Underwriter_{random.randint(1, NUM_UNDERWRITERS)}")
            return # Process End (Rejection)
        else:
            log_event(env, case_id, "Credit Approved", f"Underwriter_{random.randint(1, NUM_UNDERWRITERS)}")

    # --- STEP 3: Branching Logic (High Value Check) ---
    if loan_amount > 200000:
        # High value loans require VP approval
        with vp.request() as req:
            yield req # Wait for VP
            log_event(env, case_id, "Escalation to VP", "System_Escalation")
            
            # VP takes variable time depending on "busyness"
            yield env.timeout(random.uniform(120, 480)) # 2 to 8 hours
            log_event(env, case_id, "Final VP Approval", "VP_Finance")
    else:
        # Auto-finalize for standard loans
        log_event(env, case_id, "Final Approval Issued", "System_Auto")

    # --- Process End ---
    log_event(env, case_id, "Funds Disbursed", "Finance_Ops")

def run_simulation(env, num_cases, clerk, underwriter, vp):
    for i in range(num_cases):
        # Generate a new case
        env.process(loan_process(env, f"LOAN_{i:04d}", clerk, underwriter, vp))
        
        # Wait before next case arrives (Poisson arrival process)
        t = random.expovariate(1.0 / INTER_ARRIVAL_MEAN)
        yield env.timeout(t)

# --- EXECUTION ---
print("Starting Simulation...")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

env = simpy.Environment()

# Define Resources
clerk_res = simpy.Resource(env, capacity=NUM_CLERKS)
underwriter_res = simpy.Resource(env, capacity=NUM_UNDERWRITERS)
vp_res = simpy.Resource(env, capacity=NUM_VPS)

# Run
env.process(run_simulation(env, NUM_CASES, clerk_res, underwriter_res, vp_res))
env.run()

# --- EXPORT ---
df = pd.DataFrame(event_log)
df = df.sort_values(by=['CaseID', 'Timestamp'])
print(f"Simulation Complete. Generated {len(df)} events.")
print(df.head(10))

# Save to CSV for your Process Mining Tool
csv_path = os.path.join(OUTPUT_DIR, "financial_event_log_INFORMS.csv")
df.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}")
