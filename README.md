# Dataset for Fluctuation-Aware Attention Network (Flation) in Real-Time Runoff Forecasting
This dataset supports the research on real-time runoff forecasting (RRF) using the Flation model, focusing on a coastal watersheds study area with rainfall-driven runoff regimes. It is designed for training, validating, and testing data-driven RRF models, especially those emphasizing frequency-domain analysis and cross-channel interaction.

## Dataset Overview
The dataset consists of high-resolution hourly hydrometeorological and hydrological data from small coastal watersheds, capturing distinct topographic, climatic, and runoff characteristics.

### Coastal Watersheds Dataset
- **Study Area**: 5 small coastal watersheds (Station IDs: 1015, 703, 708, 693, 626) on Calvert and Hecate Islands (British Columbia, Canada), part of the Northeast Pacific Coastal Temperate Rainforest (NPCTR).
- **Time Range**: Continuous hourly observations from October 2013 to September 2019 (specific time ranges per station below).
- **Key Variables**: Hourly precipitation, air temperature, relative humidity, solar radiation, and runoff discharge.
- **Hydrological Features**: Rainfall-dominated (pluvial) runoff, with varying lake regulation effects, terrain slopes (21.7%–40.3%), forest coverage (29.9%–79.8%), and elevation (59–1012 meters).

#### Station-Specific Time Details
| Station ID | Time Range | Total Time Steps |
|------------|------------|------------------|
| 1015       | 2014/7/31 15:00 – 2019/10/1 0:00 | 45296 |
| 708        | 2014/9/9 12:00 – 2019/10/1 0:00 | 44127 |
| 703        | 2014/9/9 12:00 – 2019/10/1 0:00 | 44319 |
| 693        | 2014/8/2 13:00 – 2019/10/1 0:00 | 45248 |
| 626        | 2014/8/2 13:00 – 2019/10/1 0:00 | 45252 |

## Data Processing
All raw data have undergone standardized preprocessing to ensure reliability and suitability for deep-learning modeling:
1. **Quality Control**: Two-tier flagging system (Accepted Value/AV, Suspicious Value-Caution/SVC, Suspicious Value-Discard/SVD) to filter sensor malfunctions, noise, and invalid records.
2. **Uncertainty Exclusion**: Excluded watersheds with high data uncertainty (e.g., Watersheds 819 and 844 with ±32%–77% uncertainty) due to unstable monitoring conditions.
3. **Data Integration**: Retained valid data from 5 watersheds, unified data formats, and aligned time steps to form a structured dataset for model training.

## Data Access
### Raw Data Source
Coastal Watersheds Raw Data (Oct 2013–Sep 2019): [Hakai Institute Repository](https://catalogue.hakai.org/dataset/ca-cioos_395aa495-de81-4947-b1c5-2c98172a6def)

### Processed Data & Code
Processed datasets (quality-screened, uncertainty-filtered) and model code are available at: [GitHub Repository](https://github.com/jahaur1/Flation)
