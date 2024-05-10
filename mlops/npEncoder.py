import numpy as np
import requests
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

feats = np.random.randn(16000)

json_data = json.dumps({"features": feats}, cls=NpEncoder)

print(json_data)
requests.post("http://127.0.0.1:8265/predict", data=json_data)
