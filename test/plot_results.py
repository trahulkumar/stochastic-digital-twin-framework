import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- PATH SETUP ---
OUTPUT_DIR = "outputs"
INPUT_FILE = os.path.join(OUTPUT_DIR, 'financial_event_log_INFORMS.csv')

# Load the data
print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# --- PRE-PROCESSING ---
# Convert Timestamp to numeric just in case
df['Timestamp'] = pd.to_numeric(df['Timestamp'])

# Sort by Case and Time
df = df.sort_values(by=['CaseID', 'Timestamp'])

# Calculate Cycle Time per Case (End Time - Start Time)
case_stats = df.groupby('CaseID')['Timestamp'].agg(['min', 'max'])
case_stats['CycleTime'] = case_stats['max'] - case_stats['min']

# Identify Case Type (Did it go to VP?)
def identify_type(group):
    if group['Resource'].str.contains('VP_Finance').any():
        return 'Escalated (High Value)'
    elif group['Activity'].str.contains('Rejected').any():
        return 'Rejected'
    else:
        return 'Standard'

case_types = df.groupby('CaseID').apply(identify_type).reset_index(name='CaseType')
merged_df = pd.merge(case_stats, case_types, on='CaseID')

# --- FIGURE 1: Cycle Time Distribution (The "Heavy Tail" Proof) ---
print("Generating Figure 1...")
plt.figure(figsize=(10, 6))
sns.histplot(merged_df['CycleTime'], kde=True, color='#2C3E50', bins=30)
plt.title('Distribution of Loan Application Processing Times', fontsize=14)
plt.xlabel('Cycle Time (Minutes)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
output_fig1 = os.path.join(OUTPUT_DIR, 'Fig1_CycleTime_Distribution.png')
plt.savefig(output_fig1, dpi=300)
print(f"Saved {output_fig1}")
# plt.show()

# --- FIGURE 2: Bottleneck Analysis (Phase Durations) ---
print("Generating Figure 2...")
# We need to calculate duration of specific segments.
# "Underwriting" is roughly time between "Data Check Complete" and "Credit Approved/Rejected"

def get_segment_duration(group):
    try:
        start = group[group['Activity'] == 'Data Check Complete']['Timestamp'].values[0]
        # End could be Approved or Rejected
        if 'Credit Approved' in group['Activity'].values:
            end = group[group['Activity'] == 'Credit Approved']['Timestamp'].values[0]
        elif 'Application Rejected' in group['Activity'].values:
            end = group[group['Activity'] == 'Application Rejected']['Timestamp'].values[0]
        else:
            return None
        return end - start
    except:
        return None

underwriting_times = df.groupby('CaseID').apply(get_segment_duration).reset_index(name='Underwriting_Duration')
underwriting_times = underwriting_times.dropna()

plt.figure(figsize=(8, 6))
sns.boxplot(y=underwriting_times['Underwriting_Duration'], color='#E74C3C', width=0.4)
plt.title('Bottleneck Analysis: Underwriting Phase Duration', fontsize=14)
plt.ylabel('Duration (Minutes)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Annotate the median
median_val = underwriting_times['Underwriting_Duration'].median()
plt.text(0.25, median_val, f'Median: {median_val:.1f} min', verticalalignment='center')
plt.tight_layout()
output_fig2 = os.path.join(OUTPUT_DIR, 'Fig2_Bottleneck_Boxplot.png')
plt.savefig(output_fig2, dpi=300)
print(f"Saved {output_fig2}")
# plt.show()

# --- FIGURE 3: Impact of Escalation (Variant Comparison) ---
print("Generating Figure 3...")
plt.figure(figsize=(10, 6))
sns.boxplot(x='CaseType', y='CycleTime', data=merged_df, palette='Set2')
plt.title('Impact of Process Variants on Cycle Time', fontsize=14)
plt.xlabel('Process Path', fontsize=12)
plt.ylabel('Total Cycle Time (Minutes)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
output_fig3 = os.path.join(OUTPUT_DIR, 'Fig3_Variant_Comparison.png')
plt.savefig(output_fig3, dpi=300)
print(f"Saved {output_fig3}")
# plt.show()
