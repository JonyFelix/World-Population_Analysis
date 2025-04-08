## ⚙️ How to Use This Project

### ✅ Use the code from the `Source` folder and run according to each step.

---

### 📥 Data Preparation  
- Use the files in the `Dataset` folder.  
- Load `Final_Pop_Data.xlsx` into your Jupyter Notebook.

---

### 📈 Forecasting (Using Prophet)  
- Open and run the notebook in the `Forecasting Code` folder.  
- This will forecast population metrics from **2026 to 2030**.  
- Forecasted Excel files will be saved in the `output` directory.

---

### 🌍 GeoData Mapping  
- Upload the `Admin_Countries_ShapeFile` to your working directory.  
- Run the notebook in `GeoPandas Merge` to merge shapefile with forecast data.  
- This will generate a `forecast_data.geojson` file.

---

### 🗺️ Generate Interactive Map  
- Use the code in the `Folium Map Code` folder.  
- It will read the `geojson` and generate the `population_forecast_map.html` file.  
- This is your final interactive map.

---
