# This script makes a new csv file with additional variables from the NASA exoplanet archive

import pandas as pd
import numpy as np
import requests
from io import StringIO
import os

#constants
EARTH_RADIUS = 6.371e6  # meters
EARTH_MASS = 5.972e24  # kg
EARTH_DENSITY = 5514  # kg/m^3
EARTH_ESCAPE_VELOCITY = 11186  # m/s
EARTH_SURFACE_TEMP_K = 288  # Kelvin (approx. 15 C)
ASTRONOMICAL_UNIT = 1.496e11 # meters
SOLAR_RADIUS = 6.957e8 # meters
GRAVITATIONAL_CONSTANT = 6.674e-11 # m^3 kg^-1 s^-2

# physical property calculations

def calculate_equilibrium_temp(star_temp, star_radius_solar, semi_major_axis_au, albedo=0.3):
    """Calculates the estimated equilibrium temperature of a planet."""
    if pd.isna(star_temp) or pd.isna(star_radius_solar) or pd.isna(semi_major_axis_au):
        return np.nan
    
    star_radius_m = star_radius_solar * SOLAR_RADIUS
    semi_major_axis_m = semi_major_axis_au * ASTRONOMICAL_UNIT
    
    # simplified formula
    temp = star_temp * (star_radius_m / (2 * semi_major_axis_m))**0.5 * (1 - albedo)**0.25
    return temp

def calculate_density(mass_earth, radius_earth):
    """Calculates the bulk density of a planet in kg/m^3."""
    if pd.isna(mass_earth) or pd.isna(radius_earth) or radius_earth == 0:
        return np.nan
    
    mass_kg = mass_earth * EARTH_MASS
    radius_m = radius_earth * EARTH_RADIUS
    volume_m3 = (4/3) * np.pi * radius_m**3
    return mass_kg / volume_m3

def calculate_escape_velocity(mass_earth, radius_earth):
    """Calculates the escape velocity of a planet in m/s."""
    if pd.isna(mass_earth) or pd.isna(radius_earth) or radius_earth == 0:
        return np.nan
        
    mass_kg = mass_earth * EARTH_MASS
    radius_m = radius_earth * EARTH_RADIUS
    return (2 * GRAVITATIONAL_CONSTANT * mass_kg / radius_m)**0.5


def calculate_esi(row):
    """Calculates the ESI based on radius, density, esc. velocity, and temp."""
    
    # ESI formula for an individual property x
    def esi_val(x, x_earth, weight):
        if pd.isna(x): return 0
        return (1 - abs((x - x_earth) / (x + x_earth))) ** weight

    # weights are chosen based on standard ESI formulations
    esi_radius = esi_val(row['pl_rade'], 1.0, 0.57)
    esi_density = esi_val(row['density_si'], EARTH_DENSITY, 1.07)
    esi_velocity = esi_val(row['escape_vel_si'], EARTH_ESCAPE_VELOCITY, 0.70)
    esi_temp = esi_val(row['eq_temp_k'], EARTH_SURFACE_TEMP_K, 5.58)
    
    # final ESI is the geometric mean of the property ESIs
    return (esi_radius * esi_density * esi_velocity * esi_temp) ** 0.25



if __name__ == '__main__':
    
    # New query with all variables
    TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    query = """
    SELECT
        pl_name, hostname, pl_radj, pl_bmassj, pl_orbsmax,
        st_spectype, st_age, st_mass, st_rad, st_teff,
        sy_snum, cb_flag, pul_flag, ptv_flag, pl_dens, pl_insol, pl_eqt, st_lum
    FROM
        pscomppars
    WHERE
        sy_dist < 15.3
    """
    
    # Load the dataset from the API
    try:
        print("Querying NASA Exoplanet Archive for detailed data...")
        response = requests.get(TAP_URL + query + "&format=csv")
        df = pd.read_csv(StringIO(response.text))
        print(f"Acquired data for {len(df)} exoplanets.")
    except Exception as e:
        print(f"Error during API query: {e}")
        exit()

    
    # Convert base units to Earth-relative or SI
    df['pl_rade'] = df['pl_radj'] * 11.209
    df['pl_bmasse'] = df['pl_bmassj'] * 317.83

    # Calculate derived physical properties needed for ESI
    print("Calculating derived physical properties (temperature, density, etc.)...")
    df['eq_temp_k'] = df.apply(lambda row: calculate_equilibrium_temp(row['st_teff'], row['st_rad'], row['pl_orbsmax']), axis=1)
    df['density_si'] = df.apply(lambda row: calculate_density(row['pl_bmasse'], row['pl_rade']), axis=1)
    df['escape_vel_si'] = df.apply(lambda row: calculate_escape_velocity(row['pl_bmasse'], row['pl_rade']), axis=1)

    # Add a check to compare calculated values with API values (pl_dens and pl_eqt)
    df['eq_temp_diff'] = df['eq_temp_k'] - df['pl_eqt']
    df['density_diff'] = df['density_si'] - (df['pl_dens'] * 1000) # Convert g/cm^3 to kg/m^3

    # Remove rows where essential calculations failed
    df.dropna(subset=['eq_temp_k', 'density_si', 'escape_vel_si'], inplace=True)
    
    print("Calculating Earth Similarity Index (ESI) for remaining planets...")
    df['esi_score'] = df.apply(calculate_esi, axis=1)

    # Display results and save to CSV
    df_sorted = df.sort_values(by='esi_score', ascending=False)
    
    output_columns = [
        'pl_name',
        'hostname',
        'esi_score',
        'pl_eqt', # official equilibrium temperature from the API
        'eq_temp_k', # calculated temperature
        'pl_dens', # official density from the API
        'density_si', # calculated density
        'pl_rade'  # shows radius in Earth radii
    ]
    
    df_display = df_sorted[output_columns].head(30).round(2)
    
    
    output_dir = "esi comparison"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the top 30 results to a new CSV file
    df_display.to_csv(os.path.join(output_dir, "narrowing_hscore_30.csv"), index=False)
    print("\nSuccessfully saved the top 30 ESI candidates to 'esi comparison/narrowing_hscore_30.csv'.")
    
    # Kelvin to Celsius for interpretation
    df_display['official_eq_temp_c'] = (df_display['pl_eqt'] - 273.15).round(1)
    df_display['calculated_eq_temp_c'] = (df_display['eq_temp_k'] - 273.15).round(1)

    print("\nTop 10 Exoplanet Candidates by Earth Similarity Index (ESI)")
    print("Note: The ESI score is based on your calculated values.")
    print(df_display[['pl_name', 'hostname', 'esi_score', 'official_eq_temp_c', 'calculated_eq_temp_c', 'pl_dens', 'density_si', 'pl_rade']].head(10))

    # compare with the provided ESI dataset
    # check if the esi.csv file exists before trying to read it
    esi_csv_path = os.path.join(output_dir, "esi.csv")
    if os.path.exists(esi_csv_path):
        try:
            # Load the ESI dataset for comparison
            df_esi = pd.read_csv(esi_csv_path)

            # Extract the planet names into sets for fast comparison
            hscore_planets = set(df_display['pl_name'])
            esi_planets = set(df_esi['Name'])

            # Find the intersection (planets common to both sets)
            common_planets = hscore_planets.intersection(esi_planets)

            # Calculate the number of planets in each set
            num_hscore_planets = len(hscore_planets)
            num_common_planets = len(common_planets)

            # Calculate the percentage of overlap
            if num_hscore_planets > 0:
                percentage_overlap = (num_common_planets / num_hscore_planets) * 100
                print(f"\n--- Comparison with ESI Dataset ---")
                print(f"Number of planets in your dataset: {num_hscore_planets}")
                print(f"Number of common planets found in the ESI dataset: {num_common_planets}")
                print(f"Percentage of your planets included in the ESI dataset: {percentage_overlap:.2f}%")
            else:
                print("\nYour dataset is empty, cannot calculate overlap.")

            
            if num_common_planets > 0:
                print("\nCommon planets found:")
                for planet in sorted(list(common_planets)):
                    print(f"- {planet}")
        except Exception as e:
            print(f"Error reading or processing 'esi.csv': {e}")
    else:
        print("\nCould not find 'esi.csv' for comparison. Please place it in the 'esi comparison' folder.")