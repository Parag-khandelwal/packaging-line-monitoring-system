
# Packaging Line Monitoring System

This project aims to predict machine failures based on various operational parameters, helping industries anticipate maintenance needs and avoid costly downtime. The model uses sensor data such as temperature, rotational speed, torque, and tool wear to classify if a machine failure will occur.

## Dataset

The dataset contains sensor readings from an industrial machine. Each row represents operational conditions and the corresponding failure status.

- **Columns:**
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Rotational speed [rpm]`
  - `Torque [Nm]`
  - `Tool wear [min]`
  - **Failure indicators:** `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
  - **Target variable:** `Machine failure` (0 = No Failure, 1 = Failure)

## Project Structure

```bash

Predictive-Maintenance-Using-ML/
│
├── data/
│   └── ai4i2020.csv            
│
├── eda/
│   └── EDA.ipynb              
│
├── src/
│   ├── preprocess.py  
│   ├── train.py               
│   ├── test.py               
│   ├── predict.py            
│   ├── config.py              
│
├── models/
│   └── random_forest.pkl      
│
├── notebooks/
    └── evaluate.ipynb  
                 
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Parag-khandelwal/packaging-line-monitoring-system.git
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv myenv
   source myenv/Scripts/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the project:**
   - Train the model: `python src/train.py`
   - Test the model: `python src/test.py`
   - Predict using new data: `python src/predict.py`

## Model

The model is a **Random Forest Classifier** trained to predict whether a machine will fail based on the operational conditions.

## Test Cases

The project includes test cases for both failure (`1`) and non-failure (`0`) conditions. You can check `predict.py` to test various scenarios.

## License

This project is open-source and available under the [MIT License](LICENSE).
```
