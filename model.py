import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime, timedelta

def train_model():
    # دریافت داده‌های بیت‌کوین از سال 2020 تا امروز
    btc = yf.Ticker("BTC-USD")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 سال داده
    
    df = btc.history(start=start_date, end=end_date)
    
    # ایجاد ویژگی‌ها
    df['Price_Change'] = df['Close'].pct_change()
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Close'].rolling(window=7).std()
    
    # حذف ردیف‌های خالی
    df = df.dropna()
    
    # ویژگی‌ها و هدف
    features = ['Price_Change', 'MA_7', 'MA_30', 'Volatility', 'Volume']
    X = df[features].values
    y = df['Close'].shift(-1).values  # قیمت روز بعد
    
    # حذف آخرین ردیف (که مقدار هدف ندارد)
    X = X[:-1]
    y = y[:-1]
    
    # نرمال‌سازی
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # آموزش مدل
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # ذخیره مدل و scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(df.tail(30), 'last_data.pkl')  # ذخیره 30 روز آخر
    
    return model, scaler

def predict_price():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        last_data = joblib.load('last_data.pkl')
    except:
        model, scaler = train_model()
        last_data = joblib.load('last_data.pkl')
    
    # گرفتن آخرین داده
    latest = last_data.iloc[-1:]
    
    # محاسبه ویژگی‌ها
    features = ['Price_Change', 'MA_7', 'MA_30', 'Volatility', 'Volume']
    X_pred = latest[features].values
    
    # نرمال‌سازی
    X_scaled = scaler.transform(X_pred)
    
    # پیش‌بینی
    prediction = model.predict(X_scaled)[0]
    
    return {
        'predicted_price': float(prediction),
        'current_price': float(latest['Close'].values[0]),
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    result = predict_price()
    print(f"قیمت فعلی: ${result['current_price']:,.2f}")
    print(f"قیمت پیش‌بینی شده برای فردا: ${result['predicted_price']:,.2f}")