import copy
import numpy as np

def clip_macro_position(coords, macro_width, macro_height, halo, die_area):
    """
    Clip macro position to ensure macro + halo stays within die boundaries.
    Matches C++ clipInstBoundingBox logic.
    
    Args:
        coords: [x, y] current position (origin/lower-left corner)
        macro_width, macro_height: macro dimensions
        halo: halo width
        die_area: dict with 'lower_left' and 'upper_right'
    
    Returns:
        Clipped [x, y] coordinates
    """
    core_min_x = die_area['lower_left'][0]
    core_min_y = die_area['lower_left'][1]
    core_max_x = die_area['upper_right'][0]
    core_max_y = die_area['upper_right'][1]
    
    new_x = coords[0]
    new_y = coords[1]
    
    # Create bloated bounding box (origin + dimensions + halo on all sides)
    bloated_min_x = coords[0] - halo
    bloated_min_y = coords[1] - halo
    bloated_max_x = coords[0] + macro_width + halo
    bloated_max_y = coords[1] + macro_height + halo
    
    # Clip X axis (if-elif, not both)
    if bloated_min_x < core_min_x:
        new_x = core_min_x + halo
    elif bloated_max_x > core_max_x:
        new_x = core_max_x - macro_width - halo
    
    # Clip Y axis (if-elif, not both)
    if bloated_min_y < core_min_y:
        new_y = core_min_y + halo
    elif bloated_max_y > core_max_y:
        new_y = core_max_y - macro_height - halo
    
    return np.array([new_x, new_y])

def force_based_placement(
    data,
    original_data,
    overlap_force,
    spring_force,
    boundary_force,
    damping_factor,
    halo_size,
    velocities
):
    """
    Iteration-based force legalization with velocity damping (Erleben-style).
    """

    modified_data = copy.deepcopy(data)

    if original_data is None:
        original_data = copy.deepcopy(data)

    # Initialize pseudo-velocities if not provided
    if velocities is None:
        velocities = {
            name: np.array([0.0, 0.0])
            for name in modified_data['macros']
        }

    die_area = modified_data['die_area']
    die_lower_left = die_area['lower_left']
    die_upper_right = die_area['upper_right']
    center_die = (np.array(die_lower_left) + np.array(die_upper_right)) / 2

    macro_names = list(modified_data['macros'].keys())
    n_macros = len(macro_names)

    macro_width = 155420
    macro_height = 81200

    # --- FORCE ACCUMULATION ---
    forces = {name: np.array([0.0, 0.0]) for name in macro_names}

    # Overlap repulsion
    for i in range(n_macros):
        name_i = macro_names[i]
        coords_i = np.array(modified_data['macros'][name_i]['coordinates'])
        
        # Create bloated rectangle for i
        rect_i_min = coords_i - halo_size
        rect_i_max = coords_i + np.array([macro_width, macro_height]) + halo_size
        center_i = (rect_i_min + rect_i_max) / 2

        for j in range(i + 1, n_macros):
            name_j = macro_names[j]
            coords_j = np.array(modified_data['macros'][name_j]['coordinates'])
            
            # Create bloated rectangle for j
            rect_j_min = coords_j - halo_size
            rect_j_max = coords_j + np.array([macro_width, macro_height]) + halo_size
            center_j = (rect_j_min + rect_j_max) / 2

            # Check for rectangle intersection
            overlap_x = rect_i_max[0] > rect_j_min[0] and rect_j_max[0] > rect_i_min[0]
            overlap_y = rect_i_max[1] > rect_j_min[1] and rect_j_max[1] > rect_i_min[1]

            if overlap_x and overlap_y:
                # Calculate intersection rectangle
                overlap_min = np.maximum(rect_i_min, rect_j_min)
                overlap_max = np.minimum(rect_i_max, rect_j_max)
                overlap_size = overlap_max - overlap_min
                
                # Overlap distance is the minimum of width and height of intersection
                overlap_dist = min(overlap_size[0], overlap_size[1])

                # Get direction from center i to center j
                direction = center_j - center_i
                norm = np.linalg.norm(direction)

                if norm < 1e-6:
                    # Centers are identical - use horizontal direction as fallback
                    direction = np.array([1.0, 0.0])
                else:
                    direction /= norm
                    
                    # Add perpendicular perturbation for perfectly aligned macros
                    # Scale by 1/spring_force to overcome spring pulling them back
                    perturb_strength = 1.0 / spring_force if spring_force > 0 else 5.0
                    
                    if abs(direction[0]) < 1e-6:
                        # Vertically aligned - add horizontal component
                        direction = np.array([perturb_strength, direction[1]])
                        direction /= np.linalg.norm(direction)
                    elif abs(direction[1]) < 1e-6:
                        # Horizontally aligned - add vertical component
                        direction = np.array([direction[0], perturb_strength])
                        direction /= np.linalg.norm(direction)

                force_vector = direction * overlap_dist * overlap_force

                # print(f"{name_i=} / {name_j=} / {coords_i=} / {coords_j=} / {overlap_size=} / {overlap_dist=} / direction: {center_j - center_i} / direction after normalize: {direction} / {force_vector=}")

                forces[name_i] -= force_vector
                forces[name_j] += force_vector

    # Spring forces to original positions
    for name in macro_names:
        current_pos = np.array(modified_data['macros'][name]['coordinates'])
        original_pos = np.array(original_data['macros'][name]['coordinates'])

        direction = original_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance > 1e-6:
            forces[name] += spring_force * direction

    # Boundary push forces (from center of area to the edge)
    for name in macro_names:
        # Use non-bloated bbox center for boundary forces
        current_pos = np.array(modified_data['macros'][name]['coordinates'])
        current_center = current_pos + np.array([macro_width / 2, macro_height / 2])

        direction = current_center - center_die
        distance = np.linalg.norm(direction)

        # Additional factor which is smallest when at boundary and larger when near center
        max_distance = np.linalg.norm((np.array(die_upper_right) - np.array(die_lower_left)) / 2)
        # pow by 3 to have stronger fall off
        boundary_factor = ((max_distance - distance) / max_distance) ** 3

        forces[name] += boundary_force * direction * boundary_factor

    # --- DAMPED ITERATION UPDATE ---
    for name in macro_names:
        velocities[name] = (1.0 - damping_factor) * velocities[name] + forces[name]

        # print(f"{name} : {velocities[name]}")

        new_pos = np.array(modified_data['macros'][name]['coordinates']) + velocities[name]

        new_pos = clip_macro_position(new_pos, macro_width, macro_height, halo_size, modified_data['die_area'])

        modified_data['macros'][name]['coordinates'] = (int(new_pos[0]), int(new_pos[1]))

    return modified_data, velocities