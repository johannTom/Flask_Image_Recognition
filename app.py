"""Serve the hand-sign image upload and prediction pages."""

from flask import Flask, render_template, request
from werkzeug.exceptions import BadRequestKeyError

from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def main():
    """Display the image upload form."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Display a prediction or an error for an invalid upload."""
    try:
        img = preprocess_img(request.files["file"].stream)
        pred = predict_result(img)
    except (BadRequestKeyError, OSError, ValueError):
        return render_template(
            "result.html", err="File cannot be processed."
        )

    return render_template("result.html", predictions=str(pred))


if __name__ == "__main__":
    app.run(port=9000, debug=True)
