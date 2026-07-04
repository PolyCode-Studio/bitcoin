from flask import Flask, jsonify
from flask_cors import CORS
from model import predict_price, train_model
import threading
import time
import os
import sys

app = Flask(__name__)
CORS(app)

cache = {'data': None, 'last_update': None}

def update_cache():
    while True:
        try:
            if not os.path.exists('model.pkl'):
                print("📚 در حال آموزش مدل...")
                train_model()
            cache['data'] = predict_price()
            cache['last_update'] = time.time()
            print(f"✅ پیش‌بینی: ${cache['data']['predicted_price']:,.2f}")
        except Exception as e:
            print(f"❌ خطا: {e}")
        time.sleep(300)

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    if cache['data'] is None:
        try:
            if not os.path.exists('model.pkl'):
                train_model()
            cache['data'] = predict_price()
        except Exception as e:
            print(f"❌ خطا در پیش‌بینی: {e}")
            train_model()
            cache['data'] = predict_price()
    return jsonify(cache['data'])

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'message': '🚀 سرور بیت‌کوین در حال کار است!'
    })

if __name__ == '__main__':
    # شروع ترد به‌روزرسانی
    thread = threading.Thread(target=update_cache, daemon=True)
    thread.start()
    
    # دریافت پورت از محیط (برای رندر)
    port = int(os.environ.get('PORT', 5000))
    
    # ⚠️ اینجا رو عوض کن: به جای 127.0.0.1 بزار 0.0.0.0
    app.run(debug=False, host='0.0.0.0', port=port)
