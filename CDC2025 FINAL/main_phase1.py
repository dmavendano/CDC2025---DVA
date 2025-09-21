import pandas as pd
import requests
from io import StringIO
import numpy as np

# url api
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="

# tap query to select specific columns (filtering variables)
query = """
SELECT 
    pl_name, hostname, pl_radj, pl_bmassj, sy_dist, st_spectype, st_age,st_mass,pl_orbper
FROM 
    pscomppars
WHERE   
    sy_dist < 15.3"""


# status update for troubleshooting
print("querying NASA exoplanet archive")
response = requests.get(TAP_URL + query + "&format=csv")
print("query successful")

# prepares the text data to be read by pandas
csv_data = StringIO(response.text) 

# reads the data into a pandas dataframe
data = pd.read_csv(csv_data)

#verification step
print(f'\nSuccessfully aquired data for {len(data)} exoplanets within parameter')
print("Data preview:")
print(data.head())

# save data to csv
#data.to_csv('CDC2025/exoplanet_data.csv', index=False)
#print("\nData saved to 'CDC2025/exoplanet_data.csv'")
#print("Process completed.")



