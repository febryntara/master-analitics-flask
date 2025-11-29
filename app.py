from flask import Flask, request, jsonify, send_file
from preprocess import run_preprocessing
from analytics import word_frequency, generate_wordcloud

app = Flask(__name__)

# ============================
# ROUTE PREPROCESS (lama)
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
        return jsonify(result), 200

    except Exception as e:
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
        return jsonify({"frequencies": freq}), 200
    except Exception as e:
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
        return send_file(img_path, mimetype='image/png'), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
