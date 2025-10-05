import joblib
import numpy as np
from joblib import load

# ----------------------------------------------------------------------
# INSTRUCTIONS FOR RUNNING THIS SCRIPT:
#
# 1. Place the 'exoplanet_app.joblib' file in the same folder as this script.
# 2. Install the necessary libraries:
#    $ pip install joblib numpy scikit-learn xgboost
# 3. Execute the script from your terminal (Command Prompt/PowerShell/Bash):
#    $ python test_prediction.py
# ----------------------------------------------------------------------

MODEL_FILE = 'exoplanet_app.joblib'

def run_prediction_example():
    """
    Loads the saved hybrid model and runs a test prediction on two samples
    representing a False Positive and a potential Exoplanet.
    """
    try:
        # Load the model from the .joblib file.
        # The loaded object is a tuple: (RandomForest, XGBoost, Meta-Classifier)
        rf_model, xgb_model, meta_model = load(MODEL_FILE)
        
        print(f"✅ Model successfully loaded: '{MODEL_FILE}'")
        print("-" * 50)
        
    except FileNotFoundError:
        print(f"❌ ERROR: Model file '{MODEL_FILE}' not found.")
        print("Please ensure 'exoplanet_app.joblib' is in the same directory.")
        return
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return
        
    # Test Data: 8 features (Order must match the training data)
    # Features: mean_flux, std_dev_flux, min_flux, max_flux, variance_flux, 
    #           best_frequency, lomb_scargle_power, false_alarm_probability
    
    # Sample 1: A possible False Positive (Low periodicity power, high FAP)
    false_positive_data = np.array([
        1.000001, 0.00045, 0.99902, 1.00109, 2.025e-07, 0.04631, 0.5987, 0.00000721
    ]).reshape(1, -1)

    # Sample 2: A strong Exoplanet Candidate (High periodicity power, low FAP)
    exoplanet_candidate_data = np.array([
        0.999912, 0.000305, 0.998934, 1.000785, 9.3094e-08, 0.08862, 0.61218, 0.000001455
    ]).reshape(1, -1)

    print("Running prediction on two synthetic test samples...")
    print("-" * 50)


    def predict_sample(data, name):
        """Helper function to run the hybrid prediction pipeline."""
        
        # 1. Get predictions/probabilities from base models
        rf_test_proba = rf_model.predict_proba(data)[:, 1]
        xgb_test_proba = xgb_model.predict_proba(data)[:, 1]
        
        # 2. Stack the probabilities for the Meta-Classifier input
        blend_input = np.vstack([rf_test_proba, xgb_test_proba]).T
        
        # 3. Get the final classification result
        final_prediction = meta_model.predict(blend_input)[0]
        final_proba = meta_model.predict_proba(blend_input)[0, 1]
        
        result_label = "EXOPLANET (1)" if final_prediction == 1 else "FALSE POSITIVE (0)"
        
        print(f"[{name}]")
        print(f"  > Final Classification: {result_label}")
        print(f"  > Confidence Score (Class 1): {final_proba:.4f}")
        print("-" * 50)
        
    
    predict_sample(false_positive_data, "Test Sample A (Likely Stellar Noise)")
    predict_sample(exoplanet_candidate_data, "Test Sample B (Strong Exoplanet Signal)")


if __name__ == "__main__":
    run_prediction_example()
