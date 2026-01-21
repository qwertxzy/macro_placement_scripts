"""
Script to read the stats json file and display a selected set of statistics.
"""

import json
import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np

def parse_range_input(range_str) -> tuple[float, float]:
    """Turns something like 0.0-1.0 into a tuple(0.0, 1.0)"""
    t = tuple(float(x.strip()) for x in range_str.split('-') if x.strip())
    
    return t if len(t) == 2 else (t[0], t[0])


# Parse command line arguments
parser = ArgumentParser(description="Display hyperparameter tuning statistics")
parser.add_argument('--stats_file', type=str, default="hyperparameter_tuning_results.json")
parser.add_argument('--overlap_force_range', '-o', type=str, required=False)
parser.add_argument('--spring_force_range', '-s', type=str, required=False)
parser.add_argument('--damping_factor_range', '-d', type=str, required=False)
parser.add_argument('--boundary_force_range', '-b', type=str, required=False)
args = parser.parse_args()

# Read stats from JSON file
with open(args.stats_file, 'r') as f:
    stats_data = json.load(f)

# Check if ranges were given and prompt user if not
if args.overlap_force_range:
    overlap_force = args.overlap_force_range
else:
    print("Possible overlap_force values in data: ")
    unique_of = sorted(set(entry['overlap_force'] for entry in stats_data))
    print(unique_of)
    overlap_force = input("Enter overlap_force values to display: ")
overlap_force = parse_range_input(overlap_force)

if args.spring_force_range:
    spring_force = args.spring_force_range
else:
    print("Possible spring_force values in data: ")
    unique_sf = sorted(set(entry['spring_force'] for entry in stats_data))
    print(unique_sf)
    spring_force = input("Enter spring_force values to display: ")
spring_force = parse_range_input(spring_force)

if args.damping_factor_range:
    damping_factor = args.damping_factor_range
else:
    print("Possible damping_factor values in data: ")
    unique_df = sorted(set(entry['damping_factor'] for entry in stats_data))
    print(unique_df)
    damping_factor = input("Enter damping_factor values to display: ")
damping_factor = parse_range_input(damping_factor)

if args.boundary_force_range:
    boundary_force = args.boundary_force_range
else:
    print("Possible boundary_force values in data: ")
    unique_bf = sorted(set(entry['boundary_force'] for entry in stats_data))
    print(unique_bf)
    boundary_force = input("Enter boundary_force values to display: ")
boundary_force = parse_range_input(boundary_force)


# Find and display the matching statistics
stat_runs = []
for entry in stats_data:
  if (overlap_force[0] <= entry['overlap_force'] <= overlap_force[1] and
      spring_force[0] <= entry['spring_force'] <= spring_force[1] and
      damping_factor[0] <= entry['damping_factor'] <= damping_factor[1] and
      boundary_force[0] <= entry['boundary_force'] <= boundary_force[1]):    
        stat_runs.append(entry)
    
if not stat_runs:
    print("No matching statistics found for the given parameters.")
    sys.exit(1)
else:
    print(f"Found {len(stat_runs)} matching runs for the given parameters.")

# Calculate which run hit 0 overlaps the fastest
fastest_run = None
fastest_iterations = float('inf')
for run in stat_runs:
    for stat in run['stats']:
        if stat['overlap_count'] == 0 and stat['iteration'] < fastest_iterations:
            fastest_iterations = stat['iteration']
            fastest_run = run

if fastest_run:
    print("Fastest run to reach 0 overlaps:")
    print(f"  Overlap Force: {fastest_run['overlap_force']}")
    print(f"  Spring Force: {fastest_run['spring_force']}")
    print(f"  Damping Factor: {fastest_run['damping_factor']}")
    print(f"  Boundary Force: {fastest_run['boundary_force']}")
    print(f"  Iterations to 0 overlaps: {fastest_iterations}")


# Setup plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.set_ylabel('Number of Overlaps')
ax1.grid(True)

ax2.set_xlabel('Iteration')
ax2.set_ylabel('Total Overlap Area')
ax2.grid(True)

colors = plt.cm.rainbow(np.linspace(0, 1, len(stat_runs)))

# Plot stats for every matching run
for i, run in enumerate(stat_runs):
  iterations = [s["iteration"] for s in run["stats"]]
  overlap_counts = [s["overlap_count"] for s in run["stats"]]
  overlap_areas = [s["total_overlap_area"] for s in run["stats"]]
  label = (f"OF={run['overlap_force']}, SF={run['spring_force']}, "
           f"DF={run['damping_factor']}, BF={run['boundary_force']}")

  # Plot overlap count
  ax1.plot(iterations, overlap_counts, color=colors[i], marker='o', linewidth=1, label=label)

  # Plot overlap area
  ax2.plot(iterations, overlap_areas, color=colors[i], marker='o', linewidth=1, label=label)

# Shrink axes to fit legend outside
box1 = ax1.get_position()
ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
ax1.legend(loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))

# plt.tight_layout(rect=[0, 0, 0.65, 1])
plt.show()
plt.close('all')
