def force_based_placement(data, original_data=None, overlap_force=0.2, spring_force=0.05, halo_size=0):
    """
    Apply force-based placement modification.
    
    Args:
        data (dict): Current placement data
        original_data (dict): Original placement data to attract macros back to
        iterations (int): Number of force iterations to perform
        overlap_force (float): Strength of repulsive force between overlapping macros
        spring_force (float): Strength of attractive force towards original positions
    
    Returns:
        dict: Modified placement data
    """
    import copy
    import numpy as np
    
    # Make a copy of the data
    modified_data = copy.deepcopy(data)
    
    # Use the input data as original if not provided
    if original_data is None:
        original_data = copy.deepcopy(data)
    
    # Get die area dimensions
    die_area = modified_data['die_area']
    die_lower_left = die_area['lower_left']
    die_upper_right = die_area['upper_right']
    die_width = die_upper_right[0] - die_lower_left[0]
    die_height = die_upper_right[1] - die_lower_left[1]
    
    # Get macro names for easier indexing
    macro_names = list(modified_data['macros'].keys())
    n_macros = len(macro_names)
    
    # Get macro dimensions (assuming all macros have the same size)
    macro_width = 155420
    macro_height = 81200        
    
    # Calculate forces for each macro
    forces = {name: np.array([0.0, 0.0]) for name in macro_names}
    
    # Calculate overlap repulsion forces
    for i in range(n_macros):
        name_i = macro_names[i]
        coords_i = np.array(modified_data['macros'][name_i]['coordinates']) - (halo_size, halo_size)
        
        for j in range(i+1, n_macros):
            name_j = macro_names[j]
            coords_j = np.array(modified_data['macros'][name_j]['coordinates']) - (halo_size, halo_size)
            
            # Check for overlap
            overlap_x = (abs(coords_i[0] - coords_j[0]) < macro_width + halo_size * 2)
            overlap_y = (abs(coords_i[1] - coords_j[1]) < macro_height + halo_size * 2)
            
            if overlap_x and overlap_y:
                # Calculate center points
                center_i = coords_i + np.array([macro_width/2, macro_height/2])
                center_j = coords_j + np.array([macro_width/2, macro_height/2])
                
                # Vector from i to j
                direction = center_j - center_i
                
                # Avoid division by zero
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.array([1.0, 0.0])  # Default direction if centers are too close
                
                # Normalize
                direction = direction / np.linalg.norm(direction)
                
                # Calculate overlap distances
                overlap_dist_x = macro_width - abs(coords_i[0] - coords_j[0]) + halo_size * 2
                overlap_dist_y = macro_height - abs(coords_i[1] - coords_j[1]) + halo_size * 2
                overlap_dist = min(overlap_dist_x, overlap_dist_y)
                
                # Apply repulsive force proportional to overlap
                force_magnitude = overlap_force * overlap_dist
                
                # Apply forces in opposite directions
                forces[name_i] -= direction * force_magnitude
                forces[name_j] += direction * force_magnitude
    
    # Calculate spring forces to original positions
    for name in macro_names:
        current_pos = np.array(modified_data['macros'][name]['coordinates'])
        original_pos = np.array(original_data['macros'][name]['coordinates'])
        
        # Vector from current to original position
        direction = original_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:  # Avoid division by zero
            direction = direction / distance
            
            # Spring force proportional to distance
            force_magnitude = spring_force * distance
            
            # Apply spring force towards original position
            forces[name] += direction * force_magnitude
    
    # Apply forces to update positions
    for name in macro_names:
        current_pos = np.array(modified_data['macros'][name]['coordinates'])
        
        # Update position with force
        new_pos = current_pos + forces[name]
        
        # Constrain to die area boundaries
        new_pos[0] = max(die_lower_left[0], min(die_upper_right[0] - macro_width, new_pos[0]))
        new_pos[1] = max(die_lower_left[1], min(die_upper_right[1] - macro_height, new_pos[1]))
        
        # Update the coordinates
        modified_data['macros'][name]['coordinates'] = (int(new_pos[0]), int(new_pos[1]))

    return modified_data