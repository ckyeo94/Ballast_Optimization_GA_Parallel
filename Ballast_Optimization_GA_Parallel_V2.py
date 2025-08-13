import pandas as pd
import numpy as np
import random
import configparser
import sys
import os
import functools
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

# --- CORE PHYSICS AND CALCULATION FUNCTIONS (Unchanged) ---

def calculate_ballast_inertia_at_com(mass, L, W, H):
    Ixx = mass / 12.0 * (W**2 + H**2); Iyy = mass / 12.0 * (L**2 + H**2); Izz = mass / 12.0 * (L**2 + W**2)
    return Ixx, Iyy, Izz

def convert_com_to_origin_properties(mass, x_com, y_com, z_com, Ixx_com, Iyy_com, Izz_com):
    Ixx_origin = Ixx_com + mass * (y_com**2 + z_com**2); Iyy_origin = Iyy_com + mass * (x_com**2 + z_com**2); Izz_origin = Izz_com + mass * (x_com**2 + y_com**2)
    Mx_origin = mass * x_com; My_origin = mass * y_com; Mz_origin = mass * z_com
    return Mx_origin, My_origin, Mz_origin, Ixx_origin, Iyy_origin, Izz_origin

def convert_origin_to_com_properties(total_mass, total_Mx, total_My, total_Mz, total_Ixx_origin, total_Iyy_origin, total_Izz_origin):
    if total_mass == 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    x_com = total_Mx / total_mass; y_com = total_My / total_mass; z_com = total_Mz / total_mass
    Ixx_com = total_Ixx_origin - total_mass * (y_com**2 + z_com**2); Iyy_com = total_Iyy_origin - total_mass * (x_com**2 + z_com**2); Izz_com = total_Izz_origin - total_mass * (x_com**2 + y_com**2)
    return total_mass, x_com, y_com, z_com, Ixx_com, Iyy_com, Izz_com

def calculate_properties_from_config(ballast_config, initial_props, ballast_props, ballast_H, unique_xy_z_base_df):
    ballast_mass, ballast_Ixx_com, ballast_Iyy_com, ballast_Izz_com = ballast_props
    current_total_mass = initial_props['Mass']
    current_Mx_origin, current_My_origin, current_Mz_origin, current_Ixx_origin, current_Iyy_origin, current_Izz_origin = convert_com_to_origin_properties(initial_props['Mass'], initial_props['X'], initial_props['Y'], initial_props['Z'], initial_props['Ixx'], initial_props['Iyy'], initial_props['Izz'])
    for (x_base, y_base), num_stacked in ballast_config.items():
        base_z_row = unique_xy_z_base_df[(unique_xy_z_base_df['x'] == x_base) & (unique_xy_z_base_df['y'] == y_base)]
        if base_z_row.empty: continue
        initial_z_at_xy = base_z_row['z'].iloc[0]
        for i in range(num_stacked):
            z_base_of_ballast = initial_z_at_xy + i * ballast_H; ballast_com_z_at_pos = z_base_of_ballast + ballast_H / 2.0
            ballast_Mx_origin, ballast_My_origin, ballast_Mz_origin, ballast_Ixx_origin_at_pos, ballast_Iyy_origin_at_pos, ballast_Izz_origin_at_pos = convert_com_to_origin_properties(ballast_mass, x_base, y_base, ballast_com_z_at_pos, ballast_Ixx_com, ballast_Iyy_com, ballast_Izz_com)
            current_total_mass += ballast_mass; current_Mx_origin += ballast_Mx_origin; current_My_origin += ballast_My_origin; current_Mz_origin += ballast_Mz_origin; current_Ixx_origin += ballast_Ixx_origin_at_pos; current_Iyy_origin += ballast_Iyy_origin_at_pos; current_Izz_origin += ballast_Izz_origin_at_pos
    return convert_origin_to_com_properties(current_total_mass, current_Mx_origin, current_My_origin, current_Mz_origin, current_Ixx_origin, current_Iyy_origin, current_Izz_origin)

# --- FITNESS FUNCTION FOR GA (Accepts all data as arguments) ---

def calculate_fitness(ballast_config, initial_props, ballast_props, ballast_H, unique_xy_z_base_df, target_props_list, weights, stacking_weight):
    """
    This function calculates the fitness of a single 'chromosome' (a ballast configuration).
    It's designed to be called by multiple processes in parallel.
    """
    props = calculate_properties_from_config(ballast_config, initial_props, ballast_props, ballast_H, unique_xy_z_base_df)
    
    mass_property_score = 0.0
    prop_keys = ['Mass', 'X', 'Y', 'Z', 'Ixx', 'Iyy', 'Izz']
    # The mass property is now handled by a fixed ballast count, so we start the property index from 1
    for i, key in enumerate(prop_keys[1:]):
        target_val = target_props_list[i+1] # Offset by 1 to skip Mass
        current_val = props[i+1]           # Offset by 1 to skip Mass
        
        # Calculate percentage error for all properties
        if abs(target_val) < 1e-9: # Handle true zero targets to avoid division by zero
            percentage_error = abs(current_val - target_val) # Use absolute error for true zero targets
        else:
            percentage_error = abs((current_val - target_val) / target_val)

        # Define tolerance for CG and MOI
        if key in ['X', 'Y', 'Z']:
            tolerance = 0.01 # 1% for CG
        elif key in ['Ixx', 'Iyy', 'Izz']:
            tolerance = 0.03 # 3% for MOI
        else:
            tolerance = 1.0 # Default for other properties (e.g., Mass, which is handled separately)

        # Penalize if error exceeds tolerance
        if percentage_error > tolerance:
            # Use a squared penalty that increases significantly beyond tolerance
            penalty = (percentage_error - tolerance)**2
        else:
            # Reward for being within tolerance (or slightly penalize for not being exactly zero)
            penalty = percentage_error**2 # Still penalize for non-zero error within tolerance

        mass_property_score += weights.get(key, 1.0) * penalty
        
    # The stacking penalty encourages fewer, larger stacks
    stacking_penalty = stacking_weight * len(ballast_config)
    
    score = mass_property_score + stacking_penalty
    return 1.0 / (score + 1e-9) # Return fitness (higher is better)

# --- GENETIC ALGORITHM OPERATORS (IMPROVED) ---

def create_individual(available_xy_coords, target_num_ballasts):
    """Creates a simple, single-stack individual with the target number of ballasts."""
    individual = {}
    loc_to_add = random.choice(available_xy_coords)
    individual[loc_to_add] = target_num_ballasts
    return individual

def crossover(parent1, parent2):
    """Performs a crossover that strictly preserves the total number of ballasts."""
    # Create a flat list of all ballast locations from both parents
    p1_ballasts = [loc for loc, count in parent1.items() for _ in range(count)]
    p2_ballasts = [loc for loc, count in parent2.items() for _ in range(count)]
    combined_pool = p1_ballasts + p2_ballasts

    # The child will have the same number of ballasts as the parents
    num_ballasts = len(p1_ballasts)
    child_ballasts = random.sample(combined_pool, num_ballasts)

    # Recreate the dictionary representation for the child
    child = {}
    for loc in child_ballasts:
        child[loc] = child.get(loc, 0) + 1
    return child

def mutate(individual, available_xy_coords, mutation_rate):
    """Mutates the placement of ballasts without changing the total count."""
    if random.random() < mutation_rate and individual:
        # Choose a random stack to remove a ballast from
        loc_from = random.choice(list(individual.keys()))
        
        # Choose a random location to add the ballast to
        loc_to = random.choice(available_xy_coords)
        
        # Move the ballast
        individual[loc_from] -= 1
        if individual[loc_from] == 0:
            del individual[loc_from]
        individual[loc_to] = individual.get(loc_to, 0) + 1
        
    return individual

def tournament_selection(population_with_fitness, k=3):
    """Selects a parent from the population using tournament selection."""
    tournament = random.sample(population_with_fitness, k)
    return max(tournament, key=lambda item: item[1])[0]

# --- NEW: VISUALIZATION FUNCTION ---
def generate_3d_plot(final_placements, all_coords_df, output_path):
    """Generates and saves a 3D scatter plot of ballast placements."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all possible locations
    ax.scatter(all_coords_df['x'], all_coords_df['y'], all_coords_df['z'], c='gray', marker='.', alpha=0.1, label='Available Locations')

    # Plot the final optimized placements
    if final_placements:
        final_placements_df = pd.DataFrame(final_placements, columns=['x', 'y', 'z'])
        ax.scatter(final_placements_df['x'], final_placements_df['y'], final_placements_df['z'], c='blue', marker='o', s=50, label='Placed Ballasts')

    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_zlabel('Z Coordinate (m)')
    ax.set_title('Optimized Ballast Placement')
    ax.legend()
    plt.savefig(output_path)
    print(f"\n3D visualization saved to '{output_path}'")

# --- MAIN ORCHESTRATION FUNCTION ---

def optimize_ballast_placement_ga(config_path='config.ini'):
    """
    Optimizes ballast placement using a parallel Genetic Algorithm.
    """
    config = configparser.ConfigParser()
    if not config.read(config_path):
        print(f"Error: Configuration file '{config_path}' not found."); sys.exit(1)

    # --- 1. Read Configuration ---
    mass_properties_path = config.get('FILES', 'mass_properties')
    ballast_coords_path = config.get('FILES', 'ballast_coordinates')
    output_csv_path = config.get('FILES', 'output_placements')
    output_plot_path = config.get('FILES', 'output_plot', fallback='ballast_placement_3d.png') # New

    ballast_mass = config.getfloat('BALLAST', 'mass'); ballast_L = config.getfloat('BALLAST', 'length'); ballast_W = config.getfloat('BALLAST', 'width'); ballast_H = config.getfloat('BALLAST', 'height')
    
    population_size = config.getint('OPTIMIZATION', 'population_size'); num_generations = config.getint('OPTIMIZATION', 'num_generations')
    initial_mutation_rate = config.getfloat('OPTIMIZATION', 'mutation_rate')
    adaptive_mutation = config.getboolean('OPTIMIZATION', 'adaptive_mutation', fallback=True) # New
    tournament_size = config.getint('OPTIMIZATION', 'tournament_size', fallback=3) # New
    elite_size = int(0.1 * population_size)

    prop_keys = ['Mass', 'X', 'Y', 'Z', 'Ixx', 'Iyy', 'Izz']
    weights = {key: config.getfloat('WEIGHTS', f'{key.lower()}_w') for key in prop_keys[1:]}
    stacking_weight = config.getfloat('WEIGHTS', 'stacking_w')

    # --- 2. Load and Prepare Data ---
    try:
        mass_df = pd.read_csv(mass_properties_path, index_col=0); mass_df.columns = mass_df.columns.str.strip(); mass_df.index = mass_df.index.str.split('(').str[0].str.strip()
        ballast_coords_df = pd.read_csv(ballast_coords_path)
        for col in ['x', 'y', 'z']:
            if ballast_coords_df[col].mean() > 10: ballast_coords_df[col] /= 1000.0
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e.filename}"); sys.exit(1)
    
    missing_keys = [key for key in prop_keys if key not in mass_df.index]
    if missing_keys:
        print(f"--- Configuration Error ---"); print(f"Error: The '{mass_properties_path}' file is missing required data: {missing_keys}"); sys.exit(1)

    initial_props = {key: mass_df.loc[key, 'Current Model'] for key in prop_keys}
    target_dict = {key: mass_df.loc[key, 'Target Value'] for key in prop_keys}
    target_props_list = [target_dict[key] for key in prop_keys]

    ballast_Ixx_com, ballast_Iyy_com, ballast_Izz_com = calculate_ballast_inertia_at_com(ballast_mass, ballast_L, ballast_W, ballast_H)
    ballast_props = (ballast_mass, ballast_Ixx_com, ballast_Iyy_com, ballast_Izz_com)
    unique_xy_z_base_df = ballast_coords_df.groupby(['x', 'y'])['z'].min().reset_index()
    available_xy_coords = list(zip(unique_xy_z_base_df['x'], unique_xy_z_base_df['y']))
    
    mass_increase_needed = target_dict['Mass'] - initial_props['Mass']
    target_num_ballasts = int(mass_increase_needed / ballast_mass) if mass_increase_needed > 0 else 0

    # --- 3. Genetic Algorithm Execution ---
    num_workers = os.cpu_count()
    print(f"--- Starting Parallel Genetic Algorithm using {num_workers} CPU cores ---")
    
    population = [create_individual(available_xy_coords, target_num_ballasts) for _ in range(population_size)]
    best_overall_config = None; best_overall_fitness = -1

    fitness_calculator = functools.partial(
        calculate_fitness,
        initial_props=initial_props,
        ballast_props=ballast_props,
        ballast_H=ballast_H,
        unique_xy_z_base_df=unique_xy_z_base_df,
        target_props_list=target_props_list,
        weights=weights,
        stacking_weight=stacking_weight
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for gen in range(num_generations):
            mutation_rate = initial_mutation_rate * (1 - gen / num_generations) if adaptive_mutation else initial_mutation_rate

            fitness_scores = list(executor.map(fitness_calculator, population))
            
            pop_with_fitness = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            
            if pop_with_fitness[0][1] > best_overall_fitness:
                best_overall_fitness = pop_with_fitness[0][1]
                best_overall_config = pop_with_fitness[0][0].copy()
                print(f"Generation {gen+1}/{num_generations} | New Best Fitness: {best_overall_fitness:.4f} | Mutation Rate: {mutation_rate:.4f}")

            next_population = [p[0] for p in pop_with_fitness[:elite_size]]

            while len(next_population) < population_size:
                p1 = tournament_selection(pop_with_fitness, k=tournament_size)
                p2 = tournament_selection(pop_with_fitness, k=tournament_size)
                child = crossover(p1, p2)
                child = mutate(child, available_xy_coords, mutation_rate)
                next_population.append(child)
            
            population = next_population

    # --- 4. Report Results ---
    print("\nOptimization complete.")
    final_ballast_placements = []
    if best_overall_config:
        for (x, y), num_stacked in best_overall_config.items():
            base_z_row = unique_xy_z_base_df[(unique_xy_z_base_df['x'] == x) & (unique_xy_z_base_df['y'] == y)]
            if not base_z_row.empty:
                initial_z = base_z_row['z'].iloc[0]
                for i in range(num_stacked):
                    final_ballast_placements.append((x, y, initial_z + i * ballast_H))

    print(f"\nPlaced {len(final_ballast_placements)} ballasts in {len(best_overall_config)} stacks.")
    final_props_com = calculate_properties_from_config(best_overall_config, initial_props, ballast_props, ballast_H, unique_xy_z_base_df)
    final_props_dict = dict(zip(prop_keys, final_props_com))
    
    print("\n--- Mass Properties Deviation Report ---")
    print("| Property | Target Value | Final Value | Deviation | Error (%) |")
    print("|:---------|:-------------|:------------|:----------|:----------|")
    for key in prop_keys:
        target_val = target_dict[key]; final_val = final_props_dict[key]
        deviation = final_val - target_val; error_pct = (abs(deviation) / abs(target_val)) * 100 if target_val != 0 else 0
        print(f"| {key:<8} | {target_val:12.4f} | {final_val:11.4f} | {deviation:9.4f} | {error_pct:8.2f}% |")

    selected_ballasts_df = pd.DataFrame(final_ballast_placements, columns=['x', 'y', 'z'])
    selected_ballasts_df.to_csv(output_csv_path, index=False)
    print(f"\nOptimized ballast placements saved to '{output_csv_path}'")

    # --- 5. Generate Visualization ---
    if config.getboolean('VISUALIZATION', 'generate_plot', fallback=True):
        generate_3d_plot(final_ballast_placements, ballast_coords_df, output_plot_path)

if __name__ == '__main__':
    # Important: This guard is necessary for multiprocessing to work correctly on Windows/macOS
    optimize_ballast_placement_ga('config.ini')
