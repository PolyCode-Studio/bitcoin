from flask import Flask, jsonify
from flask_cors import CORS
from model import predict_price, train_model
import threading
import time

app = Flask(__name__)
CORS(app)

# کش کردن نتیجه برای کاهش بار
cache = {
    'data': None,
    'last_update': None
}

def update_cache():
    while True:
        try:
            cache['data'] = predict_price()
            cache['last_update'] = time.time()
            print(f"✅ پیش‌بینی به‌روز شد: {cache['data']['predicted_price']}")
        except Exception as e:
            print(f"❌ خطا در به‌روزرسانی: {e}")
        time.sleep(300)  # هر 5 دقیقه به‌روزرسانی

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    if cache['data'] is None:
        try:
            cache['data'] = predict_price()
        except:
            # اگر مدل وجود نداشت، آموزش بده
            train_model()
            cache['data'] = predict_price()
    
    return jsonify(cache['data'])

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # شروع ترد به‌روزرسانی
    thread = threading.Thread(target=update_cache, daemon=True)
    thread.start()
    
    # اجرای سرور
    app.run(debug=True, port=5000)
