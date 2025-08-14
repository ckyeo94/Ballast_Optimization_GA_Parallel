import pandas as pd
import configparser

# --- CORE PHYSICS AND CALCULATION FUNCTIONS (from Ballast_Optimization_GA_Parallel_V2.py) ---

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

def main():
    # --- 1. Load Configuration and Data ---
    config = configparser.ConfigParser()
    config.read('config.ini')

    mass_properties_path = config.get('FILES', 'mass_properties')
    ballast_coords_path = config.get('FILES', 'ballast_coordinates')
    optimized_placements_path = config.get('FILES', 'output_placements')

    ballast_mass = config.getfloat('BALLAST', 'mass')
    ballast_L = config.getfloat('BALLAST', 'length')
    ballast_W = config.getfloat('BALLAST', 'width')
    ballast_H = config.getfloat('BALLAST', 'height')

    mass_df = pd.read_csv(mass_properties_path, index_col=0)
    mass_df.columns = mass_df.columns.str.strip()
    mass_df.index = mass_df.index.str.split('(').str[0].str.strip()

    ballast_coords_df = pd.read_csv(ballast_coords_path)
    # The main script converts units if they are large (mm to m). We replicate that.
    for col in ['x', 'y', 'z']:
        if ballast_coords_df[col].mean() > 10:
            ballast_coords_df[col] /= 1000.0

    optimized_df = pd.read_csv(optimized_placements_path)

    # --- 2. Prepare Data for Calculation ---
    prop_keys = ['Mass', 'X', 'Y', 'Z', 'Ixx', 'Iyy', 'Izz']
    initial_props = {key: mass_df.loc[key, 'InitialValue'] for key in prop_keys}
    target_dict = {key: mass_df.loc[key, 'TargetValue'] for key in prop_keys}

    ballast_Ixx_com, ballast_Iyy_com, ballast_Izz_com = calculate_ballast_inertia_at_com(ballast_mass, ballast_L, ballast_W, ballast_H)
    ballast_props = (ballast_mass, ballast_Ixx_com, ballast_Iyy_com, ballast_Izz_com)

    unique_xy_z_base_df = ballast_coords_df.groupby(['x', 'y'])['z'].min().reset_index()

    # Convert the flat CSV of placements into the { (x, y): count } format
    ballast_config = optimized_df.groupby(['x', 'y']).size().to_dict()

    # --- 3. Calculate Final Properties ---
    final_props_com = calculate_properties_from_config(ballast_config, initial_props, ballast_props, ballast_H, unique_xy_z_base_df)
    final_props_dict = dict(zip(prop_keys, final_props_com))

    # --- 4. Report Results ---
    print("\n--- Manually Calculated Mass Properties Deviation Report ---")
    print("| Property | Target Value | Final Value | Deviation | Error (%) |")
    print("|:---------|:-------------|:------------|:----------|:----------|")
    for key in prop_keys:
        target_val = target_dict[key]
        final_val = final_props_dict.get(key, 0.0)
        deviation = final_val - target_val
        error_pct = (abs(deviation) / abs(target_val)) * 100 if abs(target_val) > 1e-9 else 0
        print(f"| {key:<8} | {target_val:12.4f} | {final_val:11.4f} | {deviation:9.4f} | {error_pct:8.2f}% |")

if __name__ == '__main__':
    main()
