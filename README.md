# Multi-task learning of solute segregation energy across multiple alloy systems

## Dataset  
This is a complete dataset containing approximately 2 million rows generated based on molecular dynamics simulations. It includes nine alloy systems: Ag-Cu, Ag-Ni, Al-Co, Al-Fe, Al-Mg, Al-Ni, Al-Ti, Cu-Zr, and Ni-Zr. The dataset has 23 columns: atomic number of matrix elements, atomic number of solute elements, 20 SNAP features, and segregation energy.

## Source Code  
1. **Figure2_RF.py** and **Figure2_XGBoost.py** are example codes for machine learning using only 20 SNAP features or 3 PI features, respectively. Complete prediction results are shown in **Figure2**.  
2. **Figure3.py** performs feature importance ranking analysis using the XGBoost algorithm with two atomic numbers (matrix and solute elements) and 20 SNAP features as input features.  
3. **Figure4_XGBoost.py** extends **Figure2_XGBoost.py** by incorporating two critical atomic number features. Complete prediction results are shown in **Figure4**. The XGBoost algorithm outperforms other algorithms.  
4. **Figure4_XGBoost.py** is used to produce Figure 4, with the training process saved using `pickle` method for the subsequent prediction using **Figure6and7_XGBoost.py** to produce Figures 6 and 7. 