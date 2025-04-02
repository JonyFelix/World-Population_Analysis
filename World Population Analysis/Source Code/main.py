
'''

TO RUN IN PROGRAM AND OBTAIN VISUALZATIONS
STEP 1 DOWNLOAD THE FILE FROM DATA SET FOLDER
STEP 2 CHANGE THE df PATH TO YOUR FILE LOCATION
STEP 3 RUN THE CODE (COLAB,JUPYTER NOTEBOOK OR ANY OTHER ENVIROMENT)

'''

Run Codes as per Steps 

1 Loading
2 Forecasting
3 Geo Pandas
4 Folium


'Loading The cleaned World Population Data Set(1990-2025)'

%pip install openpyxl
import pandas as pd

wpop = pd.read_excel("World pop/Final Pop.xlsx")

print(df.tail())

'Confirmation Check to Ensure the data set if Fine'

wpop.info()

wpop.describe()

mvalues = wpop.isnull().sum()
print('Missing Values',mvalues)


'''
FORECASTING THE WORLD POPULATION PARAMETERS TO 2030

FULL CODE  ↡↡↡↡ '''

# Step 1: Install necessary libraries
%pip install pandas openpyxl matplotlib prophet

# Step 2: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# Step 3: Load the dataset
wpop = pd.read_excel("World Pop/Final Pop.xlsx") #Upload the File in the IDE your using and Change the Path that you have stored.

# Step 4: Convert Year to datetime format
wpop['YEAR'] = pd.to_datetime(wpop['YEAR'], format='%Y')

# Step 5: Get unique countries
countries = wpop['COUNTRY'].unique()

# Step 6: Create a directory to save results
output_folder = "Forecast_Results"
os.makedirs(output_folder, exist_ok=True)

# Step 7: Initialize an empty dictionary to store all forecasts
all_forecasts = []

# Step 8: Function to forecast each factor for a given country
def forecast_country(country_name):
    country_data = wpop[wpop['COUNTRY'] == country_name]

    for column in ['MALE TOTAL', 'FEMALE TOTAL', 'TOTAL POPULATION', 'BIRTH RATE', 'DEATH RATE', 'FERTILIY RATE', 'IMMIGRATION RATE']:
        df = country_data[['YEAR', column]].rename(columns={'YEAR': 'ds', column: 'y'})

        model = Prophet()
        model.fit(df)

        # ✅ FIX: Use 'YE' instead of 'Y'
        future = model.make_future_dataframe(periods=5, freq='YE')  # Predicting 2026-2030
        forecast = model.predict(future)

        # Store forecasted data
        forecast['COUNTRY'] = country_name
        forecast['FACTOR'] = column
        all_forecasts.append(forecast[['COUNTRY', 'ds', 'FACTOR', 'yhat', 'yhat_lower', 'yhat_upper']])

        # Step 9: Save graph for the forecast
        fig = model.plot(forecast)
        plt.title(f"{column} Forecast for {country_name}")
        plt.xlabel("Year")
        plt.ylabel(column)
        plt.grid(True)

        # Save the graph
        graph_filename = f"{output_folder}/{country_name}_{column}_forecast.png"
        plt.savefig(graph_filename)
        plt.close()

# Step 10: Run the forecast function for each country
for country in countries:
    print(f"📊 Forecasting,,,,,, for {country}...")
    forecast_country(country)

# Step 11: Save all forecasted data to an Excel sheet
forecast_df = pd.concat(all_forecasts, ignore_index=True)
forecast_filename = f"{output_folder}/Population_Forecast_2026_2030.xlsx"
forecast_df.to_excel(forecast_filename, index=False)

print(f"✅ Forecasts saved successfully: {forecast_filename}")
print(f"✅ Graphs saved in folder: {output_folder}")

'this code generates large numbers of graphs so it will some more time run,where all the perdicted graphs will genrated'

from zipfile import ZipFile
from pathlib import Path

zipname = 'Forecast_Results.zip'  # Name of the zip file
filenames = list(Path('.').glob('*.csv'))  # Get all CSV files in the current directory

with ZipFile(zipname, 'w') as zipf:
    for file in filenames:
        zipf.write(file, arcname=file.name)  # Write each CSV file to the ZIP

print(f"✅ '{zipname}' created successfully with {len(filenames)} CSV files!")

'''
THIS CODE TO CONVERT THE OBTAIN FILE INTO ZIP TO MAKE IT EASIER TO DOWNLOAD

This Above Code only generates the FOrecast, Next we are going to use GEO PANDAS and FOLIUM to map

* GeoPandas was used to process geospatial data and merge it with the forecasted dataset.
* Folium was used to create an interactive world map displaying forecasted population data by country.


2 setps u must follow & check

# 1 population 2026 to 2030 excelfile, the excel file will be created during Forecasting , with we are gonna generate geojson file for folium
# 2 ("ne_110m_admin_0_countries.zip") #include this file on your working directory i have uploaded in Data Set

'''

import os
import pandas as pd
import geopandas as gpd
import folium

# Ensure output folder exists
output_folder = "World pop"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load Population Forecast Data
pop_data = pd.read_excel("Forecast_Results/Population_Forecast_2026_2030.xlsx") # the excel file will be created during Forecasting

# Load World Map Data
world_map = gpd.read_file("ne_110m_admin_0_countries.zip") #include this file on your working directory i have uploaded in Data Sr

# Standardize Column Names
pop_data.columns = pop_data.columns.str.strip().str.upper()
world_map.columns = world_map.columns.str.strip().str.upper()

# Define Country Name Fixes
country_replacements = {
    "RUSSIA": "RUSSIAN FEDERATION",
    "IRAN": "IRAN (ISLAMIC REPUBLIC OF)",
    "VENEZUELA": "VENEZUELA (BOLIVARIAN REPUBLIC OF)",
    "BOLIVIA": "BOLIVIA (PLURINATIONAL STATE OF)",
    "VIETNAM": "VIET NAM",
    "SYRIA": "SYRIAN ARAB REPUBLIC",
    "NORTH KOREA": "DEM. PEOPLE'S REP. OF KOREA",
    "SOUTH KOREA": "REPUBLIC OF KOREA",
    "MOLDOVA": "REPUBLIC OF MOLDOVA",
    "PALESTINE": "STATE OF PALESTINE",
    "LAOS": "LAO PEOPLE'S DEM. REPUBLIC",
    "DEMOCRATIC REPUBLIC OF THE CONGO": "DEM. REP. OF THE CONGO",
    "REPUBLIC OF THE CONGO": "CONGO",
    "COTE D'IVOIRE": "CÔTE D'IVOIRE",
    "TAIWAN": "CHINA, TAIWAN PROVINCE OF CHINA",
    "TURKEY": "TÜRKIYE",
    "EAST TIMOR": "TIMOR-LESTE",
}

# Apply Fixes
world_map["ADMIN"] = world_map["ADMIN"].replace(country_replacements)

# Merge Data
merged = world_map.merge(pop_data, left_on="ADMIN", right_on="COUNTRY", how="left")

# Define GeoJSON Path
geojson_path = os.path.join(output_folder, "merged_population_map.geojson")

# Save Merged Data as GeoJSON
merged.to_file(geojson_path, driver="GeoJSON")

# Confirm File Creation
if os.path.exists(geojson_path):
    print(f"✅ GeoJSON file saved successfully at: {geojson_path}")
else:
    print("❌ ERROR: GeoJSON file was NOT created!")

# Generate Folium Map
m = folium.Map(location=[20, 0], zoom_start=2)
folium.GeoJson(geojson_path, name="Population Forecast").add_to(m)

# Save Map
map_path = os.path.join(output_folder, "population_forecast_map.html")
m.save(map_path)
print(f"✅ Interactive map saved: {map_path}")



'''

Next Folium to Generate the map

'''


import folium
import geopandas as gpd
import pandas as pd

# Load GeoJSON
geojson_path = "/Users/arula/World pop/merged_population_map.geojson"
world_map = gpd.read_file(geojson_path)



# Convert Timestamp values to string using map() instead of applymap()
world_map = world_map.map(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)

# Create folium map
m = folium.Map(location=[0, 0], zoom_start=2)

# Add GeoJSON Layer
geojson_layer = folium.GeoJson(world_map).add_to(m)

# Save map to HTML
html_path = "/Users/arula/World pop/population_forecast_map.html"
m.save(html_path)

print(f"✅ Map saved successfully at: {html_path}")








