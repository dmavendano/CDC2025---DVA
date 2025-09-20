import pandas as pd
import numpy as np
import requests
from io import StringIO

# Constants
EARTH_RADIUS = 6.371e6  # meters
EARTH_MASS = 5.972e24  # kg
EARTH_DENSITY = 5514  # kg/m^3
EARTH_ESCAPE_VELOCITY = 11186  # m/s
EARTH_SURFACE_TEMP_K = 288  # Kelvin (approx. 15 C)
EARTH_INSOLATION_FLUX = 1.0 # Earth's insolation flux in Earth Flux units
GRAVITATIONAL_CONSTANT = 6.674e-11 # m^3 kg^-1 s^-2

def calculate_density(mass_earth, radius_earth):
    #Calculates the bulk density of a planet in kg/m^3
    if pd.isna(mass_earth) or pd.isna(radius_earth) or radius_earth == 0:
        return np.nan
    
    mass_kg = mass_earth * EARTH_MASS
    radius_m = radius_earth * EARTH_RADIUS
    volume_m3 = (4/3) * np.pi * radius_m**3
    return mass_kg / volume_m3

def calculate_escape_velocity(mass_earth, radius_earth):
    #Calculates the escape velocity of a planet in m/s
    if pd.isna(mass_earth) or pd.isna(radius_earth) or radius_earth == 0:
        return np.nan
        
    mass_kg = mass_earth * EARTH_MASS
    radius_m = radius_earth * EARTH_RADIUS
    return (2 * GRAVITATIONAL_CONSTANT * mass_kg / radius_m)**0.5

def calculate_esi(row):
    #Calculates the ESI based on radius, density, esc. velocity, and temp
    
    # ESI formula for an individual property 'x'
    def esi_val(x, x_earth, weight):
        if pd.isna(x): return 0
        return (1 - abs((x - x_earth) / (x + x_earth))) ** weight

    # weights are chosen based on standard ESI formulations
    esi_radius = esi_val(row['pl_rade'], 1.0, 0.57)
    esi_density = esi_val(row['density_si'], EARTH_DENSITY, 1.07)
    esi_velocity = esi_val(row['escape_vel_si'], EARTH_ESCAPE_VELOCITY, 0.70)
    esi_temp = esi_val(row['pl_eqt'], EARTH_SURFACE_TEMP_K, 5.58)
    
    # New factors for habitability
    esi_insolation = esi_val(row['pl_insol'], EARTH_INSOLATION_FLUX, 0.3)
    
    # Stellar Luminosity: We can use the st_lum variable, but convert it back from log to linear.
    esi_stellar_lum = esi_val(10**row['st_lum'], 1.0, 0.5) 
    
    # Number of stars in the system: A single star system is considered more stable.
    esi_system_stability = 1.0 if row['sy_snum'] == 1 else 0.5 
    
    # Circumbinary flag: We can apply a penalty to planets orbiting binary systems.
    esi_circumbinary = 1.0 if row['cb_flag'] == 0 else 0.5

    # final ESI is the geometric mean of all properties
    return (esi_radius * esi_density * esi_velocity * esi_temp * esi_insolation * esi_stellar_lum * esi_system_stability * esi_circumbinary) ** (1/8) # Note: 8 factors

if __name__ == '__main__':
    
    # Updated query to include pl_eqt and pl_insol
    TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    query = """
    SELECT
        pl_name, hostname, pl_radj, pl_bmassj, 
        sy_snum, cb_flag, pul_flag, ptv_flag, pl_dens, pl_insol, pl_eqt, st_lum
    FROM
        pscomppars
    WHERE
        sy_dist < 15.3
    """
    
    # load the dataset from the API
    try:
        print("Querying NASA Exoplanet Archive for detailed data...")
        response = requests.get(TAP_URL + query + "&format=csv")
        df = pd.read_csv(StringIO(response.text))
        print(f"Acquired data for {len(df)} exoplanets.")
    except Exception as e:
        print(f"Error during API query: {e}")
        exit()

    # convert base units to Earth-relative or SI
    df['pl_rade'] = df['pl_radj'] * 11.209
    df['pl_bmasse'] = df['pl_bmassj'] * 317.83

    # calculate derived physical properties needed for ESI
    print("Calculating derived physical properties (density, etc.)...")
    df['density_si'] = df.apply(lambda row: calculate_density(row['pl_bmasse'], row['pl_rade']), axis=1)
    df['escape_vel_si'] = df.apply(lambda row: calculate_escape_velocity(row['pl_bmasse'], row['pl_rade']), axis=1)

    # remove rows where essential calculations failed or official data is missing
    df.dropna(subset=['pl_eqt', 'pl_insol', 'density_si', 'escape_vel_si'], inplace=True)
    
    print("Calculating Earth Similarity Index (ESI) for remaining planets...")
    df['esi_score'] = df.apply(calculate_esi, axis=1)

    # display results
    df_sorted = df.sort_values(by='esi_score', ascending=False)
    
    output_columns = [
        'pl_name',
        'hostname',
        'esi_score',
        'pl_eqt', 
        'pl_insol',
        'pl_dens', 
        'pl_rade'
    ]
    
    df_display = df_sorted[output_columns].head(10).round(2)
    df_display['official_eq_temp_c'] = (df_display['pl_eqt'] - 273.15).round(1)
    
    print("\nTop 10 Exoplanet Candidates by Earth Similarity Index (ESI) with added factors")
    print(df_display[['pl_name', 'hostname', 'esi_score', 'official_eq_temp_c', 'pl_insol', 'pl_dens', 'pl_rade']])