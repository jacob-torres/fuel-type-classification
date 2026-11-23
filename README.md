# Predicting a Vehicle's Fuel Technology

A multi-class classification project: given a vehicle's engine size, cylinders, drive type, transmission, make, class, and model year, can a model tell whether it is a gasoline car, hybrid, plug-in hybrid, EV, diesel, or flex-fuel vehicle?

The notebook, [`fuel_type_classification.ipynb`](fuel_type_classification.ipynb), covers the whole workflow: building the label from a raw column, cleaning, encoding, avoiding leakage, feature selection, model comparison against a baseline, hyperparameter tuning, and per-class error analysis.

## Results

| Measure | Result |
|---|---|
| Majority-class baseline (always guess Gasoline) | 0.8652 accuracy |
| Tuned k-nearest neighbors (recommended) | 0.9243 accuracy, 0.6143 macro F1 |
| Logistic regression / random forest (class-weighted) | 0.5055 / 0.7657 accuracy |

Accuracy looks good mainly because 86.5% of vehicles are gasoline. Recall by class shows the real picture: EV 1.00, Gasoline 0.98, Diesel 0.52, Hybrid 0.47, Flex fuel 0.31, Plug-in 0.07. The notebook discusses why and what to try next (resampling, cross-validated feature selection, splitting by model year).

## Data

[EPA/DOE Fuel Economy dataset](https://www.fueleconomy.gov/feg/download.shtml) (`vehicles.csv`), about 50,000 U.S. vehicle configurations from model years 1984 to 2027. It is a U.S. government work in the public domain under the [EPA Data License](https://edg.epa.gov/EPA_Data_License.htm). The notebook downloads it directly, so no setup is needed. The published file is updated periodically, so a fresh run may differ slightly from the stored outputs.

## Run it

```
pip install -r requirements.txt
jupyter notebook fuel_type_classification.ipynb
```

Built with Python, pandas, NumPy, scikit-learn, and matplotlib.
