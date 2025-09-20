# This script makes a new csv file with additional variables from the NASA exoplanet archive

import pandas as pd
import numpy as np
import requests
from io import StringIO
import os # Import the os module for checking file paths

# --- Constants for Calculations ---
EARTH_RADIUS = 6.371e6  # meters
EARTH_MASS = 5.972e24  # kg
EARTH_DENSITY = 5514  # kg/m^3
EARTH_ESCAPE_VELOCITY = 11186  # m/s
EARTH_SURFACE_TEMP_K = 288  # Kelvin (approx. 15 C)
ASTRONOMICAL_UNIT = 1.496e11 # meters
SOLAR_RADIUS = 6.957e8 # meters
GRAVITATIONAL_CONSTANT = 6.674e-11 # m^3 kg^-1 s^-2

# --- Functions for Physical Property Calculations ---

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
    
    # ESI formula for an individual property 'x'
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


# --- Main Script ---
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