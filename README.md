# **Population Forecasting & Visualization (1990-2030)**  

## **1. Data Collection & Cleaning**  
- Data was collected from an **unofficial web page**.  
- Initially, downloading all required columns at once caused **data corruption and inconsistencies**.  
- To resolve this, each required column was **downloaded separately and then merged** into a structured dataset.  
- **Data cleaning was performed using Python (Pandas & NumPy)** to handle missing values, remove duplicates, and ensure consistency.  

## **2. Data Visualization (1990-2025) using Power BI**  
Power BI was used to create **interactive visualizations** to understand historical trends.  

### **Key insights included:**  
- **Total Population Growth Trends**  
- **Birth Rate & Death Rate Variations**  
- **Country-wise Population Distribution**  
- **Fertility Rate & Immigration Rate Comparisons**  

These visualizations provided a clear picture of **how population trends have evolved over time**.  

## **3. Forecasting (2026-2030) using Python**  
Time-series forecasting was conducted using **Prophet (Facebook’s forecasting model)** in Python.  

### **Forecasted population-related factors include:**  
- **Total Population**  
- **Male & Female Population**  
- **Birth Rate & Death Rate**  
- **Fertility Rate & Immigration Rate**  

**Matplotlib & Seaborn** were used to plot and analyze the forecasted trends.  

The results helped in understanding the **future trajectory of global population growth**.  

## **4. Interactive Population Map (GeoPandas & Folium)**  
- **GeoPandas** was used to process **geospatial data** and merge it with the forecasted dataset.  
- **Folium** was used to create an **interactive world map** displaying forecasted population data by country.  

### **Features of the map:**  
- **Color-coded population growth** to represent high and low-growth regions.  
- **Hover functionality** to display country-wise forecasted data.  
- **Interactive zoom** for a better user experience.  

## **5. Final Organization & Insights**  
- All the **visualizations, forecasting models, and the interactive map** were structured into a complete project.  
- The project provides a **clear historical analysis (1990-2025) and predictive insights (2026-2030)** for population trends.  
- The combination of **Power BI, Python, GeoPandas, and Folium** offers a **data-driven approach to understanding population growth globally**.  

This project serves as a **comprehensive tool** for researchers, analysts, and policymakers to **study and interpret population dynamics effectively**.  
