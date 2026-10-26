# Machine Learning-Based Accident Hotspot Prediction and Risk Classification Using GIS
Tamil Nadu | LightGBM + Leaflet GIS + Open-Meteo weather + OSRM routing

## Run (in this order)
```bash
python -m venv venv && venv\Scripts\activate      # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python train_model.py          # trains LightGBM, prints cross-validated results, saves model/
python fetch_areas.py          # (needs internet, 1-3 min) downloads ALL towns/suburbs of Tamil Nadu
python app.py                  # open http://127.0.0.1:5000
```
Optional: copy `.env.example` to `.env` and add a TomTom key for `/api/traffic`.

## How it maps to the Review-2 PPT
| PPT step | File / endpoint |
|---|---|
| Data collection / preprocessing | `data/preprocessed_tamil_nadu_accidents.csv` |
| LightGBM model | `train_model.py`, `features.py` |
| Risk classification Low/Medium/High | `/api/dashboard-data` |
| GIS map | `templates/map.html`, `tamil_nadu_accident_risk_map.html` |
| GPS + real-time weather -> live risk | `/api/predict` (weather rules in `app.py: weather_adjust`) |
| All 38 districts + all areas map | `/all-areas`, `fetch_areas.py` |
| Route risk analysis | `/api/route-risk` (called from `map.html`) |
| Safety recommendation | `recommendations()` in `app.py` |

## Known limitations (write these in your report - examiners like honesty)
* The 38-district map includes Chennai and Mayiladuthurai, but their risk class is **estimated from nearby district data** because local accident figures are missing; do not interpret it as measured local risk.
* Area risk = smooth blend of the 3 nearest districts, NOT area-level accident data.
* Historical accident figures are **district level, only 2 years (2020-21), for 36 of 38 districts**. Chennai and Mayiladuthurai are included as map locations but have no local accident figures in the supplied dataset.
* Real hotspot detection needs accident-point data (lat/lon of each accident).
* Weather is a rule-based adjustment, not learned: no Tamil Nadu accident-vs-weather training data exists in the project. Live weather availability depends on internet access. TomTom live traffic requires a valid API key.
* Model cross-validation on only 36 district rows reported 91.1% accuracy, while the simple previous-class baseline scored 94.4% in this run. This does **not** establish that the model is more accurate than the baseline; more years and real accident-point data are needed.
