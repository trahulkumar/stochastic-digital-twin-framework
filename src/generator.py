import simpy
import random
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
NUM_CASES = 100000
ARRIVAL_RATE = 5  # cases per hour
SEED = 42

# Resource Capacities (Baseline Staffing)
N_CLERKS = 8
N_UNDERWRITERS_PRIMARY = 5  # "Routine Lane"
N_UNDERWRITERS_RESERVE = 2  # "Exception Lane" (RADR Capacity)
N_VPS = 2

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'data' / 'raw'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_lognorm_duration(mu, sigma):
    return np.random.lognormal(mean=mu, sigma=sigma)

class FinancialProcess:
    def __init__(self, env, use_radr=False):
        self.env = env
        self.use_radr = use_radr
        self.clerk = simpy.Resource(env, capacity=N_CLERKS)
        # Split resource pools for RADR logic
        self.underwriter_primary = simpy.Resource(env, capacity=N_UNDERWRITERS_PRIMARY)
        self.underwriter_reserve = simpy.Resource(env, capacity=N_UNDERWRITERS_RESERVE)
        self.vp = simpy.Resource(env, capacity=N_VPS)
        self.log = []

    def log_event(self, case_id, activity, arrival, start, end, resource, mode):
        self.log.append({
            'CaseID': case_id,
            'Activity': activity,
            'Arrival_Time': arrival,
            'Start_Time': start,
            'End_Time': end,
            'Resource': resource,
            'Regime_Mode': mode  # 1=Routine, 2=Exception
        })

    def process_case(self, case_id):
        # Determine if this case is an Exception Mode (Bimodal Logic)
        is_exception = random.random() < 0.20  # 20% Exception Probability
        is_high_value = random.random() < 0.15
        mode_label = 2 if is_exception else 1

        # --- STEP 1: CLERK DATA ENTRY ---
        arrival = self.env.now
        with self.clerk.request() as req:
            yield req
            start = self.env.now
            yield self.env.timeout(get_lognorm_duration(2.1, 0.3))
        self.log_event(case_id, 'Data_Entry', arrival, start, self.env.now, 'Clerk', mode_label)

        # --- STEP 2: UNDERWRITING (With RADR Logic) ---
        arrival = self.env.now
        
        # RADR DECISION LOGIC: 
        # If active, route Exceptions to Reserve Lane to protect Primary Lane throughput.
        selected_resource = self.underwriter_primary
        resource_name = 'Primary_UW'
        
        if self.use_radr and is_exception:
            selected_resource = self.underwriter_reserve
            resource_name = 'Reserve_UW'

        with selected_resource.request() as req:
            yield req
            start = self.env.now
            # Duration based on mode (The "Fidelity" logic)
            duration = get_lognorm_duration(5.5, 0.6) if is_exception else get_lognorm_duration(3.8, 0.4)
            yield self.env.timeout(duration)
        
        self.log_event(case_id, 'Underwriting', arrival, start, self.env.now, resource_name, mode_label)

        # --- STEP 3: VP APPROVAL ---
        if is_high_value:
            arrival = self.env.now
            with self.vp.request() as req:
                yield req
                start = self.env.now
                yield self.env.timeout(get_lognorm_duration(4.1, 0.8))
            self.log_event(case_id, 'VP_Approval', arrival, start, self.env.now, 'VP', mode_label)

def run_simulation(use_radr=False, filename='financial_log.csv'):
    random.seed(SEED)
    np.random.seed(SEED)
    env = simpy.Environment()
    process = FinancialProcess(env, use_radr=use_radr)
    
    def generator():
        case_id = 0
        while case_id < NUM_CASES:
            env.process(process.process_case(case_id))
            interarrival = random.expovariate(ARRIVAL_RATE / 60.0)
            yield env.timeout(interarrival)
            case_id += 1

    env.process(generator())
    env.run()
    df = pd.DataFrame(process.log)
    df.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"Simulation complete. Saved to {filename}")

if __name__ == "__main__":
    # Generate Passive Log (Figure 5 context)
    run_simulation(use_radr=False, filename='passive_log_100k.csv')
    # Generate Active RADR Log (Figure 6 context)
    run_simulation(use_radr=True, filename='active_radr_log_100k.csv')
