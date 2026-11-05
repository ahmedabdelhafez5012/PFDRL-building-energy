\# Data



Raw data are not redistributed. Obtain them from the original sources:



\- Building Data Genome Project 2 (Miller et al., 2020, Scientific Data 7:368):

&#x20; https://github.com/buds-lab/building-data-genome-project-2 (files are in Git LFS; use the media URLs below).

&#x20; curl -L -o data\\bdg2\\metadata.csv https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/metadata/metadata.csv

&#x20; (same pattern for weather/weather.csv and meters/cleaned/{electricity,chilledwater,hotwater,steam}\_cleaned.csv)

&#x20; Then: python build\_multi\_data.py --k 3

\- HomeC (Smart Home Dataset with Weather Information): Kaggle, uploader "taranvee". Check the licence on the Kaggle page.

\- Ausgrid solar home electricity data 2010-2013 (Creative Commons Attribution):

&#x20; https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data

\- Weather: Open-Meteo (Zippenfenig, 2023), https://doi.org/10.5281/zenodo.7970649 (CC BY 4.0).

\- Pecan Street and the ICPE smart-office data were planned but not used.



Operator-survey responses are not included.

