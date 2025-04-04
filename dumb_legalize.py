def legalize_placement(data, macro_width, macro_height, macro_halo, iteration):
    """
    Legalizes macro placement to remove overlaps while preserving relative positioning.
    
    Args:
        data (dict): Parsed DEF file data
        macro_width (int): Width of each macro
        macro_height (int): Height of each macro
        
    Returns:
        dict: Modified data with legalized placement
    """
    # Convert from dict to list of tuples (k, v)
    macros = [{"name": k} | v for k, v in data['macros'].items()]

    # Sort macros by distance from the origin (0, 0)
    macros.sort(key=lambda x: (x['coordinates'][0]**2 + x['coordinates'][1]**2)**0.5)

    # Sort macros by their coordinates
    # macros.sort(key=lambda x: (x['coordinates'][0], x['coordinates'][1]))

    # Current macro is the ont at index current_iteration
    current_macro = macros[iteration]
    print(f"Current macro: {current_macro['name']} at {current_macro['coordinates']}")

    # Highlight current macro
    data["macros"][current_macro['name']]["highlighted"] = True

    # Get overlapping macros for current macro
    overlapping_macros = []
    for macro in macros[iteration + 1:]:
      # Check for overlap
      x1 = current_macro['coordinates'][0] - macro_halo
      y1 = current_macro['coordinates'][1] - macro_halo
      x2 = macro['coordinates'][0] - macro_halo
      y2 = macro['coordinates'][1] - macro_halo

      # Calculate potential overlap
      x_overlap = min(x1 + macro_width + 2 * macro_halo, x2 + macro_width + 2 * macro_halo) - max(x1, x2)
      y_overlap = min(y1 + macro_height + 2 * macro_halo, y2 + macro_height + 2 * macro_halo) - max(y1, y2)

      # Check if there's an actual overlap
      if x_overlap > 0 and y_overlap > 0:
        print(f"Overlap found between {current_macro['name']} and {macro['name']}")
        overlapping_macros.append((macro, x_overlap, y_overlap))
                
    # If there are no overlapping macros, return the data
    if not overlapping_macros:
      return data
    
    # Move all overlapping macros either right or down, depending which is less far
    for macro, x_overlap, y_overlap in overlapping_macros:
      # Calculate the distance to move
      if x_overlap < y_overlap:
        # Move right
        macro['coordinates'][0] += x_overlap
        print(f"Moving {macro['name']} right by {x_overlap}")
      else:
        # Move down
        macro['coordinates'][1] += y_overlap
        print(f"Moving {macro['name']} down by {y_overlap}")

    # Update macro coordinates in data with macro list
    for macro in macros:
      data['macros'][macro['name']]['coordinates'] = macro['coordinates']

    # Return for this iteration
    return data