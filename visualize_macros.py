import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import numpy as np
import copy
import os
import shutil
from matplotlib.animation import FuncAnimation

MACRO_HALO_SIZE = 10000

class MacroPlacementOptimizer:
    def __init__(self, parsed_data, macro_width=100, macro_height=100):
        """
        Initialize the optimizer with parsed DEF data.
        
        Args:
            parsed_data (dict): Parsed DEF file data
            macro_width (int): Width of each macro in DEF units
            macro_height (int): Height of each macro in DEF units
        """
        self.original_data = copy.deepcopy(parsed_data)
        self.current_data = copy.deepcopy(parsed_data)
        self.macro_width = macro_width
        self.macro_height = macro_height
        self.iterations = []
        
        # Initialize the original data as the first iteration
        self.iterations.append(copy.deepcopy(self.original_data))
        
        # Create output directory
        if os.path.exists('placement_iterations'):
            shutil.rmtree('placement_iterations')
        os.makedirs('placement_iterations')
        
        # Visualize initial state
        self._visualize_current(0)

    def modify_placement(self, modification_func, **kwargs):
        """
        Apply a modification function to the current placement.
        
        Args:
            modification_func (function): Function that modifies the placement data
            **kwargs: Additional arguments to pass to the modification function
        """
        # Create a copy of the current data for the modification function to work with
        new_data = copy.deepcopy(self.current_data)
        
        # Apply the modification function
        modified_data = modification_func(new_data, **kwargs)
        
        # Store the modified data
        self.current_data = modified_data
        
        # Store a copy of this iteration
        self.iterations.append(copy.deepcopy(modified_data))
        
        # Visualize this iteration
        self._visualize_current(len(self.iterations)-1)
        
        return modified_data
    
    def _visualize_current(self, iteration_number):
        """
        Visualize the current placement.
        
        Args:
            iteration_number (int): The iteration number for the filename
        """
        # Extract die area coordinates
        die_area = self.current_data['die_area']
        lower_left = die_area['lower_left']
        upper_right = die_area['upper_right']
        
        # Create the plot
        plt.figure(figsize=(15, 10))
        
        # Plot die area
        die_width = upper_right[0] - lower_left[0]
        die_height = upper_right[1] - lower_left[1]
        plt.gca().add_patch(plt.Rectangle(
            lower_left, 
            die_width, 
            die_height, 
            fill=False, 
            edgecolor='red', 
            linewidth=2, 
            label='Die Area'
        ))
        
        # Colors for different macro types
        macro_types_set = set(macro['type'] for macro in self.current_data['macros'].values())
        color_map = {}
        colors = plt.cm.tab10(np.linspace(0, 1, len(macro_types_set)))
        for i, macro_type in enumerate(macro_types_set):
            color_map[macro_type] = colors[i]
        
        # Plot individual macros as rectangles
        for name, macro in self.current_data['macros'].items():
            coords = macro['coordinates']
            macro_type = macro['type']
            
            # Get color based on macro type
            color = color_map.get(macro_type, 'blue')
            
            # Maybe this macro has a highlighted field
            if 'highlighted' in macro and macro['highlighted']:
                color = 'yellow'


            # Add rectangle for each macro (origin at lower-left corner)
            rect = patches.Rectangle(
                coords, 
                self.macro_width, 
                self.macro_height, 
                linewidth=1, 
                edgecolor='black', 
                facecolor=color, 
                alpha=0.5,
                label=macro_type if macro_type not in plt.gca().get_legend_handles_labels()[1] else ""
            )
            plt.gca().add_patch(rect)

            # Add another rectangle for each macro representing its halo
            halo_rect = patches.Rectangle(
                (coords[0] - MACRO_HALO_SIZE, coords[1] - MACRO_HALO_SIZE), 
                self.macro_width + MACRO_HALO_SIZE * 2, 
                self.macro_height + MACRO_HALO_SIZE * 2, 
                linewidth=1, 
                edgecolor='green', 
                facecolor=color, 
                alpha=0.5
            )
            plt.gca().add_patch(halo_rect)
            
            # Add a label in the center of the macro
            label_x = coords[0] + self.macro_width/2
            label_y = coords[1] + self.macro_height/2
            short_name = name.split('/')[-1]  # Get just the last part of the hierarchical name
            plt.text(label_x, label_y, short_name, 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    fontsize=6)
            
            # Draw a line to original position if not the first iteration
            if iteration_number > 0:
                original_coords = self.original_data['macros'][name]['coordinates']
                original_center_x = original_coords[0] + self.macro_width/2
                original_center_y = original_coords[1] + self.macro_height/2
                current_center_x = coords[0] + self.macro_width/2
                current_center_y = coords[1] + self.macro_height/2
                
                plt.plot([original_center_x, current_center_x], 
                         [original_center_y, current_center_y], 
                         'k--', alpha=0.2)
        
        # Highlight overlaps
        self._highlight_overlaps()
        
        # Add legend for macro types
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="Macro Types", 
                loc='upper left', bbox_to_anchor=(1, 1))
        
        # Formatting
        plt.title(f'Macro Placement - Iteration {iteration_number}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Set axis limits with some padding
        plt.xlim(lower_left[0] - die_width*0.05, upper_right[0] + die_width*0.05)
        plt.ylim(lower_left[1] - die_height*0.05, upper_right[1] + die_height*0.05)
        
        # Make the plot square-ish
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Save the plot
        plt.tight_layout()
        filename = f'placement_iterations/iteration_{iteration_number:03d}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        
        print(f"Saved iteration {iteration_number} to {filename}")
    
    def _highlight_overlaps(self):
        """
        Highlight overlapping macros on the current plot.
        """
        macro_names = list(self.current_data['macros'].keys())
        n_macros = len(macro_names)
        
        for i in range(n_macros):
            name_i = macro_names[i]
            coords_i = np.array(self.current_data['macros'][name_i]['coordinates'])
            
            for j in range(i+1, n_macros):
                name_j = macro_names[j]
                coords_j = np.array(self.current_data['macros'][name_j]['coordinates'])
                
                # Check for overlap
                overlap_x = (abs(coords_i[0] - coords_j[0]) < self.macro_width)
                overlap_y = (abs(coords_i[1] - coords_j[1]) < self.macro_height)
                
                if overlap_x and overlap_y:
                    # Calculate overlap rectangle
                    left = max(coords_i[0], coords_j[0])
                    bottom = max(coords_i[1], coords_j[1])
                    right = min(coords_i[0] + self.macro_width, coords_j[0] + self.macro_width)
                    top = min(coords_i[1] + self.macro_height, coords_j[1] + self.macro_height)
                    
                    # Draw overlap rectangle
                    overlap_rect = patches.Rectangle(
                        (left, bottom),
                        right - left,
                        top - bottom,
                        linewidth=1,
                        edgecolor='red',
                        facecolor='red',
                        alpha=0.3
                    )
                    plt.gca().add_patch(overlap_rect)
    
    def create_animation(self, fps=1):
        """
        Create an animated GIF of all iterations.
        
        Args:
            fps (int): Frames per second for the animation
        """
        # Similar to before, but now with overlap highlighting and original position lines
        # Code is omitted for brevity but would be very similar to _visualize_current
        # with the necessary adaptations for FuncAnimation
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        def update(frame):
            ax.clear()
            
            # Extract die area coordinates
            die_area = self.iterations[frame]['die_area']
            lower_left = die_area['lower_left']
            upper_right = die_area['upper_right']
            
            # Plot die area
            die_width = upper_right[0] - lower_left[0]
            die_height = upper_right[1] - lower_left[1]
            ax.add_patch(plt.Rectangle(
                lower_left, 
                die_width, 
                die_height, 
                fill=False, 
                edgecolor='red', 
                linewidth=2, 
                label='Die Area'
            ))
            
            # Colors for different macro types
            macro_types_set = set(macro['type'] for macro in self.iterations[frame]['macros'].values())
            color_map = {}
            colors = plt.cm.tab10(np.linspace(0, 1, len(macro_types_set)))
            for i, macro_type in enumerate(macro_types_set):
                color_map[macro_type] = colors[i]
            
            # Plot individual macros as rectangles
            for name, macro in self.iterations[frame]['macros'].items():
                coords = macro['coordinates']
                macro_type = macro['type']
                
                # Get color based on macro type
                color = color_map.get(macro_type, 'blue')
                
                # Maybe this macro has a highlighted field
                if 'highlighted' in macro and macro['highlighted']:
                    color = 'yellow'

                # Add rectangle for each macro (origin at lower-left corner)
                rect = patches.Rectangle(
                    coords, 
                    self.macro_width, 
                    self.macro_height, 
                    linewidth=1, 
                    edgecolor='black', 
                    facecolor=color, 
                    alpha=0.5,
                    label=macro_type if macro_type not in ax.get_legend_handles_labels()[1] else ""
                )
                ax.add_patch(rect)
                
                # Add a label in the center of the macro
                label_x = coords[0] + self.macro_width/2
                label_y = coords[1] + self.macro_height/2
                short_name = name.split('/')[-1]  # Get just the last part of the hierarchical name
                ax.text(label_x, label_y, short_name, 
                       horizontalalignment='center', 
                       verticalalignment='center',
                       fontsize=6)
                
                # Draw a line to original position if not the first iteration
                if frame > 0:
                    original_coords = self.original_data['macros'][name]['coordinates']
                    original_center_x = original_coords[0] + self.macro_width/2
                    original_center_y = original_coords[1] + self.macro_height/2
                    current_center_x = coords[0] + self.macro_width/2
                    current_center_y = coords[1] + self.macro_height/2
                    
                    ax.plot([original_center_x, current_center_x], 
                           [original_center_y, current_center_y], 
                           'k--', alpha=0.2)
            
            # Highlight overlaps
            self._highlight_overlaps_for_animation(ax, frame)
            
            # Set axis limits with some padding
            ax.set_xlim(lower_left[0] - die_width*0.05, upper_right[0] + die_width*0.05)
            ax.set_ylim(lower_left[1] - die_height*0.05, upper_right[1] + die_height*0.05)
            
            # Formatting
            ax.set_title(f'Macro Placement - Iteration {frame}')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            
            # Make the plot square-ish
            ax.set_aspect('equal', adjustable='box')
            
            return ax
        
        anim = FuncAnimation(fig, update, frames=len(self.iterations), interval=1000/fps)
        anim.save('placement_animation.gif', writer='pillow', fps=fps, dpi=300)
        plt.close()
        
        print("Created animation of all iterations: placement_animation.gif")
    
    def _highlight_overlaps_for_animation(self, ax, frame):
        """
        Highlight overlapping macros for a specific frame.
        
        Args:
            ax: The matplotlib axis
            frame: The frame number
        """
        macro_names = list(self.iterations[frame]['macros'].keys())
        n_macros = len(macro_names)
        
        for i in range(n_macros):
            name_i = macro_names[i]
            coords_i = np.array(self.iterations[frame]['macros'][name_i]['coordinates'])
            
            for j in range(i+1, n_macros):
                name_j = macro_names[j]
                coords_j = np.array(self.iterations[frame]['macros'][name_j]['coordinates'])
                
                # Check for overlap
                overlap_x = (abs(coords_i[0] - coords_j[0]) < self.macro_width)
                overlap_y = (abs(coords_i[1] - coords_j[1]) < self.macro_height)
                
                if overlap_x and overlap_y:
                    # Calculate overlap rectangle
                    left = max(coords_i[0], coords_j[0])
                    bottom = max(coords_i[1], coords_j[1])
                    right = min(coords_i[0] + self.macro_width, coords_j[0] + self.macro_width)
                    top = min(coords_i[1] + self.macro_height, coords_j[1] + self.macro_height)
                    
                    # Draw overlap rectangle
                    overlap_rect = patches.Rectangle(
                        (left, bottom),
                        right - left,
                        top - bottom,
                        linewidth=1,
                        edgecolor='red',
                        facecolor='red',
                        alpha=0.3
                    )
                    ax.add_patch(overlap_rect)
    
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
                
                for j in range(n+1, n_macros):
                    name_j = macro_names[j]
                    coords_j = np.array(iteration_data['macros'][name_j]['coordinates'])
                    
                    # Check for overlap
                    overlap_x = (abs(coords_i[0] - coords_j[0]) < self.macro_width)
                    overlap_y = (abs(coords_i[1] - coords_j[1]) < self.macro_height)
                    
                    if overlap_x and overlap_y:
                        overlap_count += 1
                        
                        # Calculate overlap area
                        left = max(coords_i[0], coords_j[0])
                        bottom = max(coords_i[1], coords_j[1])
                        right = min(coords_i[0] + self.macro_width, coords_j[0] + self.macro_width)
                        top = min(coords_i[1] + self.macro_height, coords_j[1] + self.macro_height)
                        
                        area = (right - left) * (top - bottom)  / 1000 # Too large otherwise..
                        total_overlap_area += area
            
            stats.append({
                'iteration': i,
                'overlap_count': overlap_count,
                'total_overlap_area': total_overlap_area
            })
        
        return stats
    
    def plot_overlap_statistics(self):
        """
        Plot the overlap statistics over iterations.
        """
        stats = self.get_overlap_statistics()
        
        iterations = [s['iteration'] for s in stats]
        overlap_counts = [s['overlap_count'] for s in stats]
        overlap_areas = [s['total_overlap_area'] for s in stats]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot overlap count
        ax1.plot(iterations, overlap_counts, 'b-o', linewidth=2)
        ax1.set_ylabel('Number of Overlaps')
        ax1.set_title('Overlap Statistics Over Iterations')
        ax1.grid(True)
        
        # Plot overlap area
        ax2.plot(iterations, overlap_areas, 'r-o', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Total Overlap Area')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('overlap_statistics.png', dpi=300)
        plt.close()
        
        print("Overlap statistics plot saved as 'overlap_statistics.png'")
    
    def save_final_result(self):
        """
        Save the final placement to a JSON file.
        """
        with open('final_placement.json', 'w') as f:
            json.dump(self.current_data, f, indent=2)
        
        print("Saved final placement to final_placement.json")

# Example usage demonstration
def main():
    # Load the parsed DEF data from a JSON file
    with open('shitty_macros.json', 'r') as f:
        parsed_data = json.load(f)
    
    # Set macro dimensions
    macro_width = 155420
    macro_height = 81200
    
    # Initialize the optimizer
    optimizer = MacroPlacementOptimizer(parsed_data, macro_width, macro_height)
    
    # Import force-based placement function from the other file
    # from force_legalize import force_based_placement
    
    # # Run multiple iterations of force-based placement
    # overlap_force = 0.8
    # spring_force = 0.05
    
    # for i in range(50):
    #     print(f"Running force-based iteration {i+1}")

    #     optimizer.modify_placement(
    #         force_based_placement,
    #         original_data=optimizer.original_data,
    #         overlap_force=overlap_force,
    #         spring_force=spring_force,
    #         halo_size=MACRO_HALO_SIZE
    #     )

    from dumb_legalize import legalize_placement

    for i in range(len(parsed_data['macros'])):
        try:
            optimizer.modify_placement(
                legalize_placement,
                macro_width=macro_width,
                macro_height=macro_height,
                macro_halo=MACRO_HALO_SIZE,
                iteration=i)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping the optimization process.")
            break
    
    # Create an animation of all iterations
    optimizer.create_animation(fps=2)
    
    # Plot statistics about the optimization process
    optimizer.plot_overlap_statistics()
    
    # Save the final result
    optimizer.save_final_result()

if __name__ == '__main__':
    main()