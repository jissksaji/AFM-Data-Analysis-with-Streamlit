# **AFM Data Analysis with Streamlit**  

## **Overview**  
This project involves analyzing data from an **Atomic Force Microscope (AFM)**, where a thin needle measures forces applied on a surface with unknown material blobs. The objective is to process and visualize the data using Python, estimate slopes in force-distance curves, and create an interactive **Streamlit** application.

---

## **Project Tasks**  

### **1️⃣ Part I: Extract & Visualize AFM Data**  
- Extract force-distance measurement data from a text file.  
- Plot the data for each surface coordinate `(i, j)`.  
- Save plots for visualization.  
- **Implemented in `plotafm.py`**.  

### **2️⃣ Part II: Estimate Slopes**  
- Identify the linear-looking region in the force-distance curve.  
- Estimate the slope by splitting the space into blocks and detecting where the data changes drastically to find the linear region.  
- Output slope values for each measurement.  
- **Implemented in `estimate_afm.py`**.  

### **3️⃣ Part III: Streamlit Web App**  
- Load and process preprocessed AFM data (`afm.heights.npy`, `afm.data.pickled`).  
- Display:  
  - Heatmap of **height data**.  
  - Heatmap of **estimated slopes**.  
  - Interactive selection of `(s, i, j)` to show the force-distance curve.  
- **Implemented in `afm.py`** (not to be renamed).  

---

## **How to Download and Execute the Program**  
### **1️⃣ Download and Extract Data**  
The required dataset is available as a ZIP file. Download it from the following link:  
🔗 [AFM Data Download](https://kingsx.cs.uni-saarland.de/index.php/s/KFrpMwCfJtaLpX3)  

After downloading, extract the contents and ensure `afm.heights.npy` and `afm.data.pickled` are in the same directory as `afm.py`.

### **2️⃣ Running Part I & II (CLI Mode)**  
To extract and visualize force-distance data:  
```sh  
python plotafm.py --textfile sample.txt --plotprefix curve --show  
```  
To estimate slopes and save results:  
```sh  
python estimate_afm.py -t sample.txt  
```  

### **3️⃣ Running Part III (Streamlit App)**  
Run the Streamlit application:  
```sh  
streamlit run afm.py -- afm  
```  
Interact with heatmaps and force-distance plots via the web interface.  

---

## **Summary of the Process**  
1. **Data Extraction & Visualization:** AFM measurements are extracted and plotted as force-distance curves for analysis.  
2. **Slope Estimation:** The method splits the space into blocks and detects drastic changes in the data to determine the linear region for slope estimation.  
3. **Interactive Analysis:** A Streamlit-based web app visualizes height data and estimated slopes and allows users to explore AFM measurements interactively.  

---

 
Thanks to  **Rahmann Lab** at [Rahmann Lab](https://www.rahmannlab.de/)

