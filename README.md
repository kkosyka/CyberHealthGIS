# CyberHealthGIS
## May 2017 - August 2017
(Python)</br>
Code used for REU research. <br />
Research: How to detect road surface quality (smooth, brick road, grass, and irregular bumps) through cycling using wearable devices and machine learning.

WranglePhonePebbleData.py<br />
Folders (manually created) used are examples how code manipulates data, create graphs, and extract features.<br />
(note: example data are brick road surface data)<br /><br />
Process:
1. Convert .json to .csv
2. Splitting Pebble (smart watch) & Android (smart phone) data - saved in ~/byDevice
3. Splitting Pebble & Android data by user - saved in ~/byDeviceUser
4. (Manually) trim data in order to extract the correct acceleration data for surface quality, save in ~/trim
5. Rename data in ~/trim in order to distinguish original and trimmed data
6. Create plots from files in ~/trim - saved in ~/graphs
7. Extract features from files in ~/trim, used for machine learning process - saved in ~/features

