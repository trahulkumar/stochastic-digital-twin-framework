import simpy
import random
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
NUM_CASES = 100000
ARRIVAL_RATE = 5  # cases per hour
SEED = 42

# Resource Capacities
N_CLERKS = 8
N_UNDERWRITERS = 5
N_VPS = 2

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'data' / 'raw'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / 'financial_log_100k.csv'

def get_lognorm_duration(mu, sigma):
    return np.random.lognormal(mean=mu, sigma=sigma)

class FinancialProcess:
    def __init__(self, env):
        self.env = env
        self.clerk = simpy.Resource(env, capacity=N_CLERKS)
        self.underwriter = simpy.Resource(env, capacity=N_UNDERWRITERS)
        self.vp = simpy.Resource(env, capacity=N_VPS)
        self.log = []

    def log_event(self, case_id, activity, arrival, start, end, resource):
        self.log.append({
            'CaseID': case_id,
            'Activity': activity,
            'Arrival_Time': arrival, # NEW: When they entered the queue
            'Start_Time': start,     # NEW: When work actually started
            'End_Time': end,
            'Resource': resource
        })

    def process_case(self, case_id):
        loan_amount = random.expovariate(1/50000)
        is_high_value = loan_amount > 200000
        is_exception = random.random() < 0.15

        # --- STEP 1: INTAKE ---
        arrival = self.env.now
        with self.clerk.request() as req:
            yield req
            start = self.env.now
            duration = max(0, np.random.normal(2.5, 0.2))
            yield self.env.timeout(duration)
        self.log_event(case_id, 'App_Intake', arrival, start, self.env.now, 'Clerk')

        # --- STEP 2: CREDIT CHECK ---
        arrival = self.env.now
        yield self.env.timeout(0.5) # Automated, no queue
        self.log_event(case_id, 'Credit_Check', arrival, arrival, self.env.now, 'System')

        # --- STEP 3: UNDERWRITING (The Bottleneck) ---
        arrival = self.env.now
        with self.underwriter.request() as req:
            yield req
            start = self.env.now
            if not is_exception:
                duration = get_lognorm_duration(3.8, 0.4)
            else:
                duration = get_lognorm_duration(5.5, 0.6)
            yield self.env.timeout(duration)
        self.log_event(case_id, 'Underwriting', arrival, start, self.env.now, 'Underwriter')

        # --- STEP 4: VP APPROVAL ---
        if is_high_value:
            arrival = self.env.now
            with self.vp.request() as req:
                yield req
                start = self.env.now
                duration = get_lognorm_duration(4.1, 0.8)
                yield self.env.timeout(duration)
            self.log_event(case_id, 'VP_Approval', arrival, start, self.env.now, 'VP')

def run_simulation():
    random.seed(SEED)
    np.random.seed(SEED)
    env = simpy.Environment()
    process = FinancialProcess(env)
    
    def generator():
        case_id = 0
        while case_id < NUM_CASES:
            env.process(process.process_case(case_id))
            interarrival = random.expovariate(ARRIVAL_RATE / 60.0)
            yield env.timeout(interarrival)
            case_id += 1

    print(f"Starting Simulation ({NUM_CASES} cases)...")
    env.process(generator())
    env.run()
    
    df = pd.DataFrame(process.log)
    df.sort_values(by=['CaseID', 'Start_Time'], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_simulation()
