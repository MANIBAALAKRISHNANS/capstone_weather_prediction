import pickle, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb, os

X = np.random.randn(100, 6)
y = np.random.randn(100)

rf = RandomForestRegressor(n_estimators=5, random_state=42)
rf.fit(X, y)

xgb_model = xgb.XGBRegressor(n_estimators=5, random_state=42)
xgb_model.fit(X, y)

scaler = StandardScaler()
scaler.fit(X)

os.makedirs('models', exist_ok=True)
with open('models/random_forest.pkl', 'wb') as f: pickle.dump(rf, f)
with open('models/xgboost.pkl', 'wb') as f: pickle.dump(xgb_model, f)
with open('models/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
print("✅ All models created!")
