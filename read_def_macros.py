import json
import re

def parse_def_file(filename: str) -> tuple[dict, dict]:
    """
    Parse a DEF file to extract die area and macro placements.
    
    Args:
        filename (str): Path to the DEF file
    
    Returns:
        dict: A dictionary containing die area and macro placements
    """
    die_area = None
    macro_placements = {}

    with open(filename, 'r') as f:
        in_components_section = False
        for line in f:
            # Parse Die Area
            if die_area is None and 'DIEAREA' in line:
                # Extract die area coordinates (assumes rectangular die area)
                coords_match = re.findall(r'\(\s*(\d+)\s+(\d+)\s*\)', line)
                if coords_match:
                    if die_area is None:
                        die_area = {
                            'lower_left': (int(coords_match[0][0]), int(coords_match[0][1])),
                            'upper_right': (int(coords_match[1][0]), int(coords_match[1][1]))
                        }
                
            # Parse Component (Macro) Placements
            if line.startswith("COMPONENTS"):
                in_components_section = True
                continue
            
            if in_components_section:
                parts = line.split()
                if "FIXED" in parts or "PLACED" in parts:
                    if len(parts) != 11: breakpoint()
                    _, inst_name, macro_type, _, placement_type, _, x, y, _, orientation, _ = parts
                    macro_placements[inst_name] = {
                        'type': macro_type,
                        'coordinates': (int(x), int(y)),
                        'status': placement_type,
                        'orientation': orientation
                    }
                    
            # End of COMPONENTS section
            if 'END COMPONENTS\n' == line:
                break
    
    return {
        'die_area': die_area,
        'macros': macro_placements
    }

def main():
    filename = 'shitty_macros.def'  # Replace with your DEF file path
    parsed_data = parse_def_file(filename)
    
    # Save parsed data to JSON for visualization
    with open('shitty_macros.json', 'w') as f:
        json.dump(parsed_data, f, indent=2)
    
    # Print summary
    print(f"Total macros found: {len(parsed_data['macros'])}")
    print(f"Die area: {parsed_data['die_area']}")
    print("Parsed data saved to 'parsed_def_data.json'")

if __name__ == '__main__':
    main()