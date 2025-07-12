# FMI-SnowDepth-SWE

CSV Inputs
Folder 'CSV Inputs' contains the input data (in-situ, aerial, and spaceborne) from all data sources that were used for the machine learning models.
Files containing 'Inputs_SWE_DATA' contain input data for SWE estimation for all image objects across the study, except those that contained water or roads which were excluded.
Files containing 'Inputs_SnowDepth_DATA' contain input data for snow depth estimation for all image objects across the study, except those that contained water or roads which were excluded.
File Inputs_SnowDepth_SAMPLES.csv contains the sample input data with known snow depth values on December 14, 2022.
File Inputs_SWE_SAMPLES.csv contains the sample input data with known SWE values on December 14, 2022.
Files Inputs_SWE_DATA - Part_1.csv, Inputs_SWE_DATA - Part_2.csv, and Inputs_SWE_DATA - Part_3.csv need to be joined together and match the column header format as in Inputs_SWE_SAMPLES.csv.
Files Inputs_SnowDepth_DATA - Part_1.csv and Inputs_SnowDepth_DATA - Part_2.csv need to be joined together and match the column header format as in Inputs_SnowDepth_SAMPLES.csv.
Info about the column headers can be found in the file 'CSV model inputs.txt'.

Code
Contains two seperate R scripts that depend on the files found in the 'CSV Inputs' folder. Snow_Depth_Code.R should be run first to estimate the snow depth. Then SWE_Code.R should be run after. Predicted outputs from the first code should be used as an input for the SWE prediction for the DATA for all image objects. SAMPLE data does not need to be changed for the field values.

Shapefiles
Contains two shapefiles: 'Data_SnowDepth_SWE' and 'Field_SnowDepth_SWE'. The first contains the ID for all of the rows found in the 'Inputs_SWE_DATA' and 'Inputs_SnowDepth_DATA' files to easily join and georeference the .csv data to all the image objects in the study area. It also contains a 'LULC' column for all image objects except water and roads.
'Field_SnowDepth_SWE' contains the ID for all of the rows in Inputs_SnowDepth_SAMPLES.csv and Inputs_SnowDepth_SAMPLES.csv where field data was gathered. Values with '0' indicate no data was collected (specifically for SWE) and should be ignored. All rows pertain to snow depth data.
