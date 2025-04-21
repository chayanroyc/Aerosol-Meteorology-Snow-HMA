# HMA-Snow-Aerosol-Interactions - Ancilliary Information

This repository contains the auxiliary data associated with the preprint **"Diagnosing Aerosol-Meteorological Interactions on Snow within the Earth System: A Proof-of-Concept Study over High Mountain Asia"** (EGU 2024, DOI: [10.5194/egusphere-2024-2298](https://egusphere.copernicus.org/preprints/2024/egusphere-2024-2298/)).

## Overview

This study aims to understand the complexity of Earth's climate by proposing a novel, cost-effective approach to understand the web of interactions driving climate change. We focus on how pollution and weather processes interact and drive snowmelt in Asian glaciers. Our findings reveal significant yet overlooked processes across different climate models. Our approach can help in refining the development of these models for more reliable predictions in climate-vulnerable regions.

The repository contains all necessary links to the methods and datasets used in the study.

The schematic below depicts the overall methodology implemented in the paper.
<img width="709" alt="image" src="https://github.com/user-attachments/assets/9ea6cc48-a65c-4b60-9450-6774aa883181" />

## Links for the Reanalysis datasets:

- **MATCHA** from NSF NCAR: The entire 17 year simulation is hosted here at [NSIDC](https://nsidc.org/data/hma2_matcha/versions/1). NSIDC has tools to subset,extract and visualize the data as well. A brief dicussion of the article can be found here [HiMAT2](https://himat.org/topic/matcha/).
- **ERA5** from ECMWF: The ERA5 reanalysis was obtained from the [Climate Data Store](https://cds.climate.copernicus.eu/datasets?q=ERA5&limit=30).
- **CAMS-EAC4** from ECMWF: The CAMS reanalysis was obtained from the [Atmosphere Data Store](https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4?tab=overview). 
- **MERRA2** from NASA GMAO: The MERRA2 reanalysis was obtained from [NASA GES DISC](https://disc.gsfc.nasa.gov/datasets?keywords=MERRA2&page=1). The data collection and the variables can be looked up through the [MERRA-2 File Specification](https://gmao.gsfc.nasa.gov/reanalysis/merra-2/docs/).

## Links for Algorithms/Methods

- **Relative Importance Analysis** : The algorithms for using this in a linear regression setting is provided in Text S1 of the Supplementary Information in a previous work here [RIA Algorithm](https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1029%2F2022GL099317&file=2022GL099317-sup-0001-Supporting+Information+SI-S01.pdf). A script is also provided in *./algo/RelativeImp.py*. 

- **XGBoost-SHAPc** : The model setup using XGBoost and the SHAP ocntributions is provided in the supplementary of this current work. A set of instructions is provided in *./algo/XGBoost-SHAPc.md*.

## Links for Network Analysis




