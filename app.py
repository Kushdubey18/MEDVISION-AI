from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import google.generativeai as genai
from utils.predict import predict_image

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

# =====================================
# Global Variables
# =====================================

last_disease = ""
last_confidence = 0
last_explanation = ""

# =====================================
# Flask App
# =====================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("reports", exist_ok=True)

genai.configure(
    api_key="AQ.Ab8RN6Ir7SUyyYCaD33Rfqo0XFre2XgWrXBgAg7VX6toporUEw"
)

gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash"
)


# =====================================
# Home Page
# =====================================

@app.route("/")
def home():
    return render_template("index.html")

# =====================================
# Prediction Route
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    global last_disease
    global last_confidence
    global last_explanation

    # Check image uploaded
    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    # Check file selected
    if file.filename == "":
        return "No file selected"

    # Save image
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Predict
    disease, confidence, heatmap_path = predict_image(filepath)

    confidence = round(float(confidence), 2)

    # AI Explanation
    if disease == "PNEUMONIA":

        explanation = (
            "The uploaded chest X-ray shows patterns that may be "
            "associated with pneumonia. Please consult a medical "
            "professional for confirmation."
        )

    else:

        explanation = (
            "The uploaded chest X-ray appears normal with no "
            "significant signs of pneumonia detected."
        )

    # Save values for PDF report

    last_disease = disease
    last_confidence = confidence
    last_explanation = explanation

    # Save history

    history_data = pd.DataFrame([{
        "Image": file.filename,
        "Prediction": disease,
        "Confidence": confidence
    }])

    history_file = "reports/history.csv"

    if not os.path.exists(history_file):

        history_data.to_csv(
            history_file,
            index=False
        )

    else:

        history_data.to_csv(
            history_file,
            mode="a",
            header=False,
            index=False
        )

    # Show result page

    return render_template(
    "result.html",
    disease=disease,
    confidence=confidence,
    explanation=explanation,
    image_path="/" + filepath.replace("\\", "/"),
    heatmap_path=heatmap_path
)

# =====================================
# Download PDF Report
# =====================================

@app.route("/download_report")
def download_report():

    pdf_path = "reports/medical_report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    content = [

        Paragraph(
            "MedVision AI Medical Report",
            styles["Title"]
        ),

        Spacer(1, 20),

        Paragraph(
            f"Disease: {last_disease}",
            styles["Normal"]
        ),

        Spacer(1, 10),

        Paragraph(
            f"Confidence: {last_confidence}%",
            styles["Normal"]
        ),

        Spacer(1, 10),

        Paragraph(
            f"Explanation: {last_explanation}",
            styles["Normal"]
        )

    ]

    doc.build(content)

    return send_file(
        pdf_path,
        as_attachment=True
    )

# =====================================
# Contact Page
# =====================================

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():

    response = ""

    if request.method == "POST":

        try:

            user_message = request.form.get("message")

            if user_message:

                prompt = f"""
                You are MedVision AI Assistant.

                Answer in simple language.
                Do not provide final medical diagnosis.
                Recommend consulting a doctor when necessary.

                User Question:
                {user_message}
                """

                result = gemini_model.generate_content(prompt)

                response = result.text

        except Exception as e:

            response = f"Error: {str(e)}"

    return render_template(
        "chatbot.html",
        response=response
    )
# =====================================
# History Page
# =====================================

@app.route("/history")
def history():

    history_file = "reports/history.csv"

    if os.path.exists(history_file):
        data = pd.read_csv(history_file)
        table = data.to_html(
            classes="table table-bordered",
            index=False
        )
    else:
        table = "No prediction history available"

    return render_template(
        "history.html",
        tables=table
    )

# =====================================
# Run Application
# =====================================

if __name__ == "__main__":
    app.run(debug=True)