import sys
import json
import joblib
import numpy as np
import warnings

warnings.filterwarnings('ignore')

try:
    #input JSON from command line argument
    input_json = sys.argv[1]
    data = json.loads(input_json)

    model = joblib.load('random_forest_model.pkl')

    input_array = np.array([[
        data['lap'], 
        data['position'], 
        data['pit_stop'], 
        data['tyre_age'], 
        data['grid'],
        data['alt'],
        data['driver_skill'],
        data['circuit_difficulty'],
        data['race_year'],
        data['round'],
        data['prev_lap_time']
    ]])
    
    prediction = model.predict(input_array)

    result = {"predicted_lap_time_ms": prediction[0]}
    print(json.dumps(result))

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)