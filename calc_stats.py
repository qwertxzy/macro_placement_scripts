import json
import numpy as np
import copy
import gc
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial

from force_legalize import clip_macro_position

JSON_NAME = 'ariane133.json'
MAX_ITER = 125

# Set macro dimensions
MACRO_WIDTH = 155420
MACRO_HEIGHT = 81200
MACRO_HALO = 20000

class MacroPlacementOptimizer:
    def __init__(self, parsed_data, macro_width=100, macro_height=100, macro_halo=0, folder_prefix=""):
        """
        Initialize the optimizer with parsed DEF data.
        
        Args:
            parsed_data (dict): Parsed DEF file data
            macro_width (int): Width of each macro in DEF units
            macro_height (int): Height of each macro in DEF units
        """
        self.macro_width = macro_width
        self.macro_height = macro_height
        self.macro_halo = macro_halo
        self.iterations = []
        self.folder_prefix = folder_prefix
        
        # Create working copy
        self.current_data = copy.deepcopy(parsed_data)

        # Clip all macros to ensure they (with halo) fit within die area
        for name in self.current_data['macros']:
            coords = np.array(self.current_data['macros'][name]['coordinates'])
            
            # Clip using same logic as C++ clipInstBoundingBox
            clipped_coords = clip_macro_position(coords, macro_width, macro_height, macro_halo, self.current_data['die_area'])
            
            self.current_data['macros'][name]['coordinates'] = (int(clipped_coords[0]), int(clipped_coords[1]))
        
        # Save clipped positions as original data for spring forces
        self.original_data = copy.deepcopy(self.current_data)

        # Initialize the original data as the first iteration
        self.iterations.append(copy.deepcopy(self.original_data))

    def modify_placement(self, modification_func, **kwargs):
        """
        Apply a modification function to the current placement.
        
        Args:
            modification_func (function): Function that modifies the placement data
            **kwargs: Additional arguments to pass to the modification function
        """        
        # Apply the modification function
        modified_data, velocities = modification_func(self.current_data, **kwargs)
        
        # Store the modified data
        self.current_data = modified_data
        
        # Store a copy of this iteration
        self.iterations.append(copy.deepcopy(modified_data))
        
        # Periodic garbage collection to prevent memory buildup
        if len(self.iterations) % 10 == 0:
            gc.collect()
        
        return modified_data, velocities

    
    def get_overlap_statistics(self):
        """
        Calculate and return overlap statistics for each iteration.
        
        Returns:
            list: List of dictionaries with overlap statistics for each iteration
        """
        stats = []
        
        for i, iteration_data in enumerate(self.iterations):
            overlap_count = 0
            total_overlap_area = 0
            macro_names = list(iteration_data['macros'].keys())
            n_macros = len(macro_names)
            
            for n in range(n_macros):
                name_i = macro_names[n]
                coords_i = np.array(iteration_data['macros'][name_i]['coordinates'])
                
                # Create bloated rectangle for i
                rect_i_min = coords_i - MACRO_HALO
                rect_i_max = coords_i + np.array([MACRO_WIDTH, MACRO_HEIGHT]) + MACRO_HALO
                
                for j in range(n+1, n_macros):
                    name_j = macro_names[j]
                    coords_j = np.array(iteration_data['macros'][name_j]['coordinates'])
                    
                    # Create bloated rectangle for j
                    rect_j_min = coords_j - MACRO_HALO
                    rect_j_max = coords_j + np.array([MACRO_WIDTH, MACRO_HEIGHT]) + MACRO_HALO
                    
                    # Check for rectangle intersection
                    overlap_x = rect_i_max[0] > rect_j_min[0] and rect_j_max[0] > rect_i_min[0]
                    overlap_y = rect_i_max[1] > rect_j_min[1] and rect_j_max[1] > rect_i_min[1]
                    
                    if overlap_x and overlap_y:
                        overlap_count += 1
                        
                        # Calculate intersection rectangle area
                        overlap_min = np.maximum(rect_i_min, rect_j_min)
                        overlap_max = np.minimum(rect_i_max, rect_j_max)
                        overlap_size = overlap_max - overlap_min
                        
                        area = overlap_size[0] * overlap_size[1] / 1000  # Divide to prevent overflow
                        total_overlap_area += area
            
            stats.append({
                'iteration': i,
                'overlap_count': overlap_count,
                'total_overlap_area': total_overlap_area
            })
        
        return stats     
    
    def save_final_result(self):
        """
        Save the final placement to a JSON file.
        """
        with open('final_placement.json', 'w') as f:
            json.dump(self.current_data, f, indent=2)
        
        print("Saved final placement to final_placement.json")


def run_single_parameter_set(params_tuple, parsed_data, original_data, macro_width, macro_height, macro_halo):
    """
    Run optimization for a single set of hyperparameters.
    This function is designed to be called by multiprocessing.Pool.
    
    Args:
        params_tuple: Tuple of (overlap_force, spring_force, damping_factor, boundary_force)
        parsed_data: The parsed DEF data
        original_data: Copy of original data for reference
        macro_width: Width of macros
        macro_height: Height of macros
        macro_halo: Halo size for macros
    
    Returns:
        dict: Results for this parameter combination
    """
    overlap_force, spring_force, damping_factor, boundary_force = params_tuple
    
    print(f"Process starting: overlap_force={overlap_force}, spring_force={spring_force}, "
          f"damping_factor={damping_factor}, boundary_force={boundary_force}")
    
    # Initialize the optimizer
    optimizer = MacroPlacementOptimizer(
        parsed_data,
        macro_width,
        macro_height,
        macro_halo,
        folder_prefix=f"{overlap_force}_{spring_force}_{damping_factor}_{boundary_force}"
    )

    # Import force-based placement function
    from force_legalize import force_based_placement
    
    # Init velocities to None to get started
    velocities = None

    for _ in range(MAX_ITER):
        _, velocities = optimizer.modify_placement(
            force_based_placement,
            original_data=original_data,
            overlap_force=overlap_force,
            spring_force=spring_force,
            boundary_force=boundary_force,
            damping_factor=damping_factor,
            halo_size=MACRO_HALO,
            velocities=velocities
        )

    result = {
        'overlap_force': overlap_force,
        'spring_force': spring_force,
        'damping_factor': damping_factor,
        'boundary_force': boundary_force,
        'stats': optimizer.get_overlap_statistics()
    }
    
    print(f"Process completed: overlap_force={overlap_force}, spring_force={spring_force}, "
          f"damping_factor={damping_factor}, boundary_force={boundary_force}")
    
    # Clean up
    del optimizer
    gc.collect()
    
    return result


def main():
    # Load the parsed DEF data from a JSON file
    with open(JSON_NAME, 'r') as f:
        parsed_data = json.load(f)
    
    # Copy data into original_data for reference
    original_data = copy.deepcopy(parsed_data)

    # Define ranges for hyperparameters:
    # overlap_force_range = [x / 10 for x in range(1, 16, 1)]  # 0.1 - 1.5
    # spring_force_range = [x / 100 for x in range(0, 41, 5)]  # 0.00 - 0.4
    # damping_factor_range = [x / 10 for x in range(0, 11, 1)]  # 0.0 - 1.0
    # boundary_force_range = [x / 100 for x in range(0, 12, 1)]  # 0.01 - 0.12

    overlap_force_range = [0.9]
    spring_force_range = [0.001]
    damping_factor_range = [0.5]
    boundary_force_range = [0.0]

    # Generate all parameter combinations
    parameter_combinations = list(product(
        overlap_force_range, 
        spring_force_range, 
        damping_factor_range, 
        boundary_force_range
    ))

    print(f"Total hyperparameter combinations to test: {len(parameter_combinations)}")
    num_processes = cpu_count()
    print(f"Using {num_processes} CPU cores for parallel processing")
    input("Press Enter to start...")

    # Create a partial function with the fixed arguments
    worker_func = partial(
        run_single_parameter_set,
        parsed_data=parsed_data,
        original_data=original_data,
        macro_width=MACRO_WIDTH,
        macro_height=MACRO_HEIGHT,
        macro_halo=MACRO_HALO
    )

    results = []
    
    try:
        # Use multiprocessing Pool to parallelize
        with Pool(processes=num_processes) as pool:
            # Use imap_unordered for better performance and progress tracking
            for i, result in enumerate(pool.imap_unordered(worker_func, parameter_combinations), 1):
                results.append(result)
                print(f"Completed {i}/{len(parameter_combinations)} parameter combinations")
                
                # Periodically save intermediate results
                if i % 10 == 0:
                    with open('hyperparameter_tuning_results_partial.json', 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"Saved intermediate results ({i} combinations)")
    
    except KeyboardInterrupt:
        print(f"\nProcess interrupted by user. Saving {len(results)} results collected so far.")
    
    # Save final results to JSON
    with open('hyperparameter_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted! Saved {len(results)} results to hyperparameter_tuning_results.json")
    
    # Clean up
    gc.collect()


if __name__ == '__main__':
    main()