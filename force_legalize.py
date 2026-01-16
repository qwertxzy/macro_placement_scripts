import copy
import numpy as np

def force_based_placement(
    data,
    original_data=None,
    overlap_force=0.2,
    spring_force=0.05,
    damping_factor=0.0,
    halo_size=0,
    velocities=None
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

                force_vector = direction * overlap_force * overlap_dist

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
