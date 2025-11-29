import logging
from logging.handlers import RotatingFileHandler
import os
from flask import Flask, request, jsonify, send_file
from preprocess import run_preprocessing
from analytics import word_frequency, generate_wordcloud

app = Flask(__name__)

# ============================
# SETUP LOGGING
# ============================
LOG_DIR = '/home/febryntara/master-analitics-flask/logs'  # ganti sesuai path VPS/hostinger
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'flask_app.log'),
    maxBytes=5*1024*1024,
    backupCount=5
)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.propagate = False

# ============================
# LOGGING REQUESTS
# ============================
@app.before_request
def log_request_info():
    try:
        payload = request.get_json(force=False, silent=True) or {}
        # ambil hanya raw_id
        payload_preview = {"raw_id": payload.get("raw_id")}
    except Exception:
        payload_preview = None
    app.logger.info(f'{request.method} {request.path} - from {request.remote_addr} - data: {payload_preview}')


# ============================
# ROUTE PREPROCESS
# ============================
@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.get_json()
    raw_id = data.get('raw_id')
    raw_text = data.get('raw_text')

    if not raw_id or not raw_text:
        return jsonify({"error": "raw_id dan raw_text wajib diisi"}), 422

    try:
        result = run_preprocessing(raw_id, raw_text)
        app.logger.info(f'Processed raw_id={raw_id}')
        return jsonify(result), 200
    except Exception as e:
        app.logger.exception(f'Error di /preprocess: {e}')
        return jsonify({"error": str(e)}), 500

# ============================
# ROUTE WORD FREQUENCY
# ============================
@app.route('/analytics/word-frequency', methods=['POST'])
def freq():
    data = request.get_json()
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "texts tidak boleh kosong"}), 422

    try:
        freq = word_frequency(texts)
        app.logger.info(f'Word frequency computed for {len(texts)} texts')
        return jsonify({"frequencies": freq}), 200
    except Exception as e:
        app.logger.exception(f'Error di /analytics/word-frequency: {e}')
        return jsonify({"error": str(e)}), 500

# ============================
# ROUTE WORDCLOUD IMAGE
# ============================
@app.route('/analytics/wordcloud', methods=['POST'])
def wc():
    data = request.get_json()
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "texts tidak boleh kosong"}), 422

    try:
        img_path = generate_wordcloud(texts)
        app.logger.info(f'Wordcloud generated for {len(texts)} texts')
        return send_file(img_path, mimetype='image/png'), 200
    except Exception as e:
        app.logger.exception(f'Error di /analytics/wordcloud: {e}')
        return jsonify({"error": str(e)}), 500

# ============================
# PRODUCTION SERVER ENTRY
# ============================
if __name__ == '__main__':
    # debug=False, host 0.0.0.0 supaya bisa diakses public
    app.run(host='0.0.0.0', port=5000, debug=False)
