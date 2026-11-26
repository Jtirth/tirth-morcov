"""Flask API and server-rendered shell for NextWord Lab."""
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from markov_model import MarkovModel, compare_models

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRAINING_FILE = DATA / "training.txt"
TEST_FILE = DATA / "testing.txt"
app = Flask(__name__, template_folder=str(ROOT / "frontend" / "templates"), static_folder=str(ROOT / "frontend" / "static"))


def read_text(path):
    return path.read_text(encoding="utf-8")


def active_model():
    return MarkovModel(app.config.get("TRAINING_TEXT", read_text(TRAINING_FILE)))


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=63072000"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy-Report-Only"] = ("default-src 'self'; img-src 'self' data: https:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; "
        "script-src 'self' https://cdn.jsdelivr.net; connect-src 'self' https://cdn.jsdelivr.net")
    return response


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.get("/about")
def about():
    return render_template("about.html")


@app.get("/mathematics")
def mathematics():
    return render_template("mathematics.html")


@app.get("/api/predict")
def predict():
    context = request.args.get("text", "")
    try:
        order = int(request.args.get("order", "2"))
        limit = int(request.args.get("limit", "5"))
        if len(context) > 2000:
            return jsonify(error="Input is too long (maximum 2,000 characters)."), 400
        return jsonify(active_model().predict(context, order, limit))
    except (ValueError, TypeError) as error:
        return jsonify(error=str(error)), 400


@app.post("/api/train")
def train():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    if not isinstance(text, str) or len(text) > 100000:
        return jsonify(error="Provide text up to 100,000 characters."), 400
    try:
        model = MarkovModel(text)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    app.config["TRAINING_TEXT"] = text
    return jsonify(message="Custom corpus trained in memory for this server session.", stats=model.stats())


@app.post("/api/reset")
def reset():
    app.config.pop("TRAINING_TEXT", None)
    return jsonify(message="Restored the included sample training corpus.")


@app.get("/api/dashboard")
def dashboard():
    return jsonify(active_model().stats())


@app.get("/api/compare")
def comparison():
    try:
        training_text = app.config.get("TRAINING_TEXT", read_text(TRAINING_FILE))
        return jsonify(compare_models(training_text, read_text(TEST_FILE)))
    except ValueError as error:
        return jsonify(error=str(error)), 400


@app.get("/api/example")
def classroom_example():
    example = MarkovModel(read_text(DATA / "classroom_example.txt"))
    return jsonify(example.predict("I love", order=2, limit=5))


@app.get("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
