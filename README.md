# Classification of Human Motion Time Series Data via Shapelet Transformations
This repo holds all the files for my Machine Learning Semester project: Classifying Human Motion Positional Time Series Data via Shapelet Transformations.

#### *Final Report Location:* `report/DavidsonLuke_FinalReport.pdf`

## Project Outline

|Folder|Description|
|------|-----------|
|`scripts/`|Location of the main scripts of the project, including `preprocess.py`, `analyze.py`, and `visualize.py`
|`report/` |Location of the final report and `.zip` project folder

## Script Descriptions
`preprocess.py`
- Defines the class `PreProcess()` holding all of the methods used to preprocess the raw data. These methods include reading the data into a useable format, replacing false data readings, combination of sensor axes and normalization.

`visualize.py`
- Defines the class `Visualizer()` holding all of the methods used to visualize the data. These methods include plotting the x, y and z data on a 3D grid, plotting normalized and unnormalized data next to each other, plotting all subjects for a specific action, and plotting translations between subject data.

`analyze.py`
- Defines the class `Analyzer()` holding the bulk of the analysis methods done for the project. Some of these methods include locating the offsets that minimize SSE, creating shapelets via created methods, comparing shapelets for classification, running the *k*-NN algorithm, and storing and printing results.
