"""
OpenAlgo Chart Blueprint
Serves the openalgo-chart React SPA at /chart.

Build and deploy:
    cd openalgo-chart
    npm run build
    # Windows
    xcopy /E /I /Y dist ..\\openalgo\\chart\\dist
    # Linux/Mac
    rsync -a --delete dist/ ../openalgo/chart/dist/
"""

from pathlib import Path

from flask import Blueprint, jsonify, send_file, send_from_directory, session

from database.auth_db import get_api_key_for_tradingview

chart_bp = Blueprint("chart", __name__)

CHART_DIST = Path(__file__).parent.parent / "chart" / "dist"


def is_chart_available() -> bool:
    return (CHART_DIST / "index.html").exists()


def _serve_index():
    if not is_chart_available():
        return (
            """
<html>
<head><title>OpenAlgo Chart - Not Built</title></head>
<body style="font-family:system-ui;padding:40px;max-width:600px;margin:0 auto">
<h2>Chart frontend not built</h2>
<pre style="background:#f4f4f4;padding:16px;border-radius:8px">cd openalgo-chart
npm install
npm run build
# then copy dist/ to openalgo/chart/dist/</pre>
</body></html>""",
            503,
        )
    return send_file(CHART_DIST / "index.html", mimetype="text/html")


@chart_bp.route("/chart", strict_slashes=False)
def chart_index():
    return _serve_index()


@chart_bp.route("/chart/assets/<path:filename>")
def chart_assets(filename):
    """Serve hashed JS/CSS assets with a 1-year immutable cache."""
    response = send_from_directory(CHART_DIST / "assets", filename)
    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


@chart_bp.route("/chart/config")
def chart_config():
    """Return the logged-in user's API key so the chart SPA skips the key dialog."""
    username = session.get("user")
    if not username:
        return jsonify({"error": "Not authenticated"}), 401
    api_key = get_api_key_for_tradingview(username)
    if not api_key:
        return jsonify({"error": "No API key configured"}), 404
    return jsonify({"api_key": api_key})


@chart_bp.route("/chart/<path:subpath>")
def chart_subpath(subpath):
    """SPA catch-all: serve real files when present, index.html otherwise."""
    file_path = CHART_DIST / subpath
    if file_path.is_file():
        return send_from_directory(CHART_DIST, subpath)
    return _serve_index()
