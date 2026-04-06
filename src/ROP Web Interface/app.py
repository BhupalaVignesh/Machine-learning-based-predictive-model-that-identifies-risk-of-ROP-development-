import csv
from datetime import datetime
from flask import Flask, render_template, request, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
import os

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

app = Flask(__name__)

# Load Model
model = load_model("rop_detection_model.h5")

UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER


# ===============================
# Image Enhancement (From Notebook)
# ===============================
def enhance_fundus_image(image_path, img_size=300):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))

    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    enhanced_img = enhanced_img.astype("float32")

    return enhanced_img


# ===============================
# Preprocess Image
# ===============================
def is_retinal_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (300, 300))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Count dark pixels (retina images have black borders)
    dark_pixels = np.sum(gray < 30)

    if dark_pixels > 5000:
        return True
    else:
        return False
def preprocess_image(path):

    img = enhance_fundus_image(path)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img


# ===============================
# Grad-CAM
# ===============================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[0]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


# ===============================
# Save GradCAM Image
# ===============================
def save_gradcam(image_path, heatmap):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (300, 300))

    heatmap = cv2.resize(heatmap, (300, 300))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * 0.4 + img

    gradcam_path = os.path.join(GRADCAM_FOLDER, "gradcam.jpg")

    cv2.imwrite(gradcam_path, superimposed)

    return gradcam_path


# ===============================
# Generate PDF Report
# ===============================
def generate_report(pred, conf, img, grad):

    report_path = "static/reports/report.pdf"

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("ROP Detection Report", styles['Heading1']))
    story.append(Spacer(1, 12))

    # Prediction
    if pred == "ROP":
        color = "red"
    else:
        color = "blue"

    story.append(Paragraph(
        f'<b>Prediction:</b> <font color="{color}"><b>{pred}</b></font>',
        styles['Heading3']
    ))

    story.append(Paragraph(
    f'<b>Confidence:</b> <font color="green"><b>{conf}%</b></font>',
    styles['Heading3']
))

    story.append(Spacer(1, 20))

    # ✅ Images FIRST
    story.append(Paragraph("<b>Uploaded Retinal Image:</b>", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Image(img, width=2.2*inch, height=2.2*inch))

    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Grad-CAM Visualization:</b>", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Image(grad, width=2.2*inch, height=2.2*inch))

    story.append(Spacer(1, 15))

    # ✅ Explanation AFTER images
    # Dynamic explanation for PDF

    if pred == "ROP":
        retinal_text = (
            "The uploaded retinal fundus image is analyzed using a deep learning model. "
            "The model examines blood vessels and detects abnormalities, indicating the presence of ROP."
        )
        gradcam_text = (
            "The heatmap highlights important regions used for prediction. "
            "The highlighted areas indicate abnormal regions that influenced the model's decision.\n\n"
            "- Red/Yellow → High importance\n"
            "- Blue → Less importance"
        )
    else:
        retinal_text = (
            "The uploaded retinal fundus image is analyzed using a deep learning model. "
            "The model examines blood vessels and confirms that no significant abnormalities are present, "
            "indicating a non-ROP condition."
        )
        gradcam_text = (
            "The heatmap highlights important regions used for prediction. "
            "The highlighted areas show normal retinal regions with no significant abnormalities.\n\n"
            "- Red/Yellow → High importance\n"
            "- Blue → Less importance"
        )

    story.append(Paragraph("<b>Retinal Image Analysis:</b>", styles['Heading3']))
    story.append(Paragraph(retinal_text, styles['Normal']))

    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Grad-CAM Explanation:</b>", styles['Heading3']))
    story.append(Paragraph(gradcam_text, styles['Normal']))
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    doc.build(story)

    return report_path

# ===============================
# Home
# ===============================
@app.route("/")
def home():
    return render_template("index.html")

def save_history(image, prediction, confidence):

    file = "history.csv"

    file_exists = os.path.isfile(file)

    with open(file, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Image", "Prediction", "Confidence", "Date"])

        writer.writerow([
            image,
            prediction,
            confidence,
            datetime.now()
        ])
        
# ===============================
# Prediction
# ===============================

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # ✅ Step 1: Check if retinal image
    if not is_retinal_image(filepath):
        return render_template("result.html",
                            prediction="Invalid Image",
                            confidence=0,
                            image=filepath,
                            gradcam=None,   # ✅ IMPORTANT
                            explanation="The uploaded image is not a retinal fundus image. Please upload a valid retinal image.")

    # ✅ Step 2: Preprocess + Predict
    img = preprocess_image(filepath)

    prediction = model.predict(img)

    # ✅ Step 3: DEFINE prob (THIS WAS MISSING)
    prob = float(prediction[0][0])

    # ✅ Step 4: Classification
    if prob > 0.7:
        result = "ROP"
    else:
        result = "NON-ROP"

    confidence = prob if prob >= 0.5 else 1 - prob
    confidence = round(confidence * 100, 2)

    # GradCAM
    heatmap = make_gradcam_heatmap(img, model)
    gradcam = save_gradcam(filepath, heatmap)

    # Generate report
    generate_report(result, confidence, filepath, gradcam)

    save_history(file.filename, result, confidence)

    # Explanation
    if result == "ROP":
        explanation = (
            "The retinal image shows abnormal blood vessel patterns, indicating possible "
            "Retinopathy of Prematurity (ROP). The highlighted regions in Grad-CAM "
            "represent areas of concern."
        )
    else:
        explanation = (
            "The retinal image appears normal with well-defined blood vessels. "
            "No significant abnormalities are detected, indicating a non-ROP condition."
        )

    return render_template("result.html",
                           prediction=result,
                           confidence=confidence,
                           image=filepath,
                           gradcam=gradcam,
                           explanation=explanation)
# ===============================
# Download Report
# ===============================
@app.route("/download")
def download():
    return send_file("static/reports/report.pdf", as_attachment=True)

def save_history(image, prediction, confidence):

    file = "history.csv"

    with open(file, mode='a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([
            image,
            prediction,
            confidence,
            datetime.now()
        ])
def read_history():

    history = []

    file = "history.csv"

    if os.path.exists(file):
        with open(file, "r") as f:
            reader = csv.reader(f)

            next(reader, None)

            for row in reader:
                history.append(row)

    return history
@app.route("/history")
def history():

    history = read_history()

    return render_template("history.html", history=history)
if __name__ == "__main__":
    app.run(debug=True)