import copy
import numpy as np

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
        coords_i = np.array(modified_data['macros'][name_i]['coordinates']) - (halo_size, halo_size)

        for j in range(i + 1, n_macros):
            name_j = macro_names[j]
            coords_j = np.array(modified_data['macros'][name_j]['coordinates']) - (halo_size, halo_size)

            overlap_x = abs(coords_i[0] - coords_j[0]) < macro_width + halo_size * 2
            overlap_y = abs(coords_i[1] - coords_j[1]) < macro_height + halo_size * 2

            if overlap_x and overlap_y:
                center_i = coords_i + np.array([macro_width / 2, macro_height / 2])
                center_j = coords_j + np.array([macro_width / 2, macro_height / 2])

                direction = center_j - center_i
                norm = np.linalg.norm(direction)

                if norm < 1e-6:
                    direction = np.array([1.0, 0.0])
                else:
                    direction /= norm

                overlap_dist_x = macro_width - abs(coords_i[0] - coords_j[0]) + halo_size * 2
                overlap_dist_y = macro_height - abs(coords_i[1] - coords_j[1]) + halo_size * 2
                overlap_dist = min(overlap_dist_x, overlap_dist_y)

                # Not needed anymore, just make overlap_force larger
                # force_vector = direction * overlap_force * overlap_dist
                force_vector = direction * overlap_dist

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
        current_pos = np.array(modified_data['macros'][name]['coordinates']) + np.array([macro_width / 2, macro_height / 2])

        direction = current_pos - center_die
        distance = np.linalg.norm(direction)

        # Additional factor which is smallest when at boundary and larger when near center
        max_distance = np.linalg.norm((np.array(die_upper_right) - np.array(die_lower_left)) / 2)
        # pow by 2 to have stronger fall off
        boundary_factor = ((max_distance - distance) / max_distance) ** 2

        forces[name] += boundary_force * direction * boundary_factor

    # --- DAMPED ITERATION UPDATE ---
    for name in macro_names:
        velocities[name] = (1.0 - damping_factor) * velocities[name] + overlap_force * forces[name]

        new_pos = np.array(modified_data['macros'][name]['coordinates']) + velocities[name]

        new_pos[0] = max(die_lower_left[0], min(die_upper_right[0] - macro_width, new_pos[0]))
        new_pos[1] = max(die_lower_left[1], min(die_upper_right[1] - macro_height, new_pos[1]))

        modified_data['macros'][name]['coordinates'] = (
            int(new_pos[0]),
            int(new_pos[1])
        )

    return modified_data, velocities
