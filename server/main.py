from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'secret_key'  # باش نخدمو flash messages
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpeg', 'jpg'}

# Config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the YOLO model
model = YOLO('skin_cancer_model.pt')  # Make sure to place the model file correctly

# Skin conditions dictionary

skin_conditions = {
    "actinic keratosis": {
        "short_description": "Rough, scaly patches from sun damage.",
        "detailed_description": "Actinic keratosis causes rough, scaly patches from sun damage, which can turn into skin cancer.",
        "can_be_solved_with_product": "no",
        "url": "",
        "asset": "img.png",
        "height": 200,
        "recommendation": "This condition requires medical attention. Sunscreen may help prevent further damage, but treatment like cryotherapy or topical chemotherapy should be done by a doctor."
    },
    "seborrheic keratosis": {
        "short_description": "Wart-like, harmless skin growths.",
        "detailed_description": "Seborrheic keratosis forms harmless but unsightly wart-like growths on the skin.",
        "can_be_solved_with_product": "yes",
        "url": "https://www.amazon.com.au/SkinPro-EXTREME-Skin-Remover-Corrector/dp/B075QZ2NCV",
        "asset": "1.png",
        "height": 300,
        "recommendation": "SkinPro EXTREME Skin Tag Remover offers an at-home solution for safely removing benign skin lesions. Consult a dermatologist if unsure about treatment."
    },
    "basal cell carcinoma": {
        "short_description": "The most common form of skin cancer.",
        "detailed_description": "Basal cell carcinoma is the most common type of skin cancer, caused by sun exposure.",
        "can_be_solved_with_product": "no",
        "url": "",
        "asset": "img.png",
        "height": 200,
        "recommendation": "This condition requires medical treatment such as surgical excision or topical therapies prescribed by a doctor. Sunscreen can help prevent further damage."
    },
    "nevus": {
        "short_description": "Common mole that may change over time.",
        "detailed_description": "A nevus (mole) is usually benign but may develop into melanoma if it changes over time.",
        "can_be_solved_with_product": "no",
        "url": "",
        "asset": "img.png",
        "height": 200,
        "recommendation": "Monitor moles with regular skin checks. If you notice any changes in a mole, consult a dermatologist for removal and biopsy."
    },
    "pigmented benign keratosis": {
        "short_description": "Dark spots that can mimic melanoma.",
        "detailed_description": "Pigmented benign keratosis causes dark spots that may resemble melanoma.",
        "can_be_solved_with_product": "yes",
        "url": "https://www.amazon.com/Murad-Environmental-Shield-Rapid-Correcting/dp/B08KR79NBR",
        "asset": "4.png",
        "height": 300,
        "recommendation": "Murad Rapid Age Spot and Pigment Lightening Serum helps reduce pigmentation, improving skin tone. It's ideal for cosmetic treatment of dark spots."
    },
    "vascular lesion": {
        "short_description": "Red or purple marks from blood vessels.",
        "detailed_description": "Vascular lesions are red or purple marks caused by abnormal blood vessels.",
        "can_be_solved_with_product": "yes",
        "url": "https://www.amazon.com/Verseo-Vein-Eraser-Spider-Concealer/dp/B0CCJXMB47",
        "asset": "5.png",
        "height": 300,
        "recommendation": "Vein Eraser – Varicose & Spider Vein Treatment Cream can help fade the appearance of vascular spots over time. Consult a doctor for more severe cases."
    }
}

# Route principale
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    message = ''
    if request.method == 'POST':
        if 'image' not in request.files:
            message = 'No file part'
            return render_template('index.html', message=message)

        file = request.files['image']

        if file.filename == '':
            message = 'No selected file'
            return render_template('index.html', message=message)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Predict the class of the image
            results = model.predict(source=file_path, imgsz=420)
            
            # Extract predicted class IDs
            names = model.names
            pred_classes = results[0].boxes.cls
            
            if pred_classes is not None and len(pred_classes) > 0:
                pred_classes = pred_classes.cpu().numpy().astype(int)
                pred_class_names = [names[c] for c in pred_classes]
                predicted_class = ", ".join(pred_class_names)  # Join multiple predicted classes, if any
            else:
                predicted_class = "No detection"
            
            # Retrieve additional data for the predicted class
            if predicted_class in skin_conditions:
                condition_info = skin_conditions[predicted_class]
            else:
                condition_info = {
                    "short_description":"check the medical",
                    "detailed_description": "No problem.",
                    "recommendation": "Please consult a medical professional if you don't feel well.",
                    "url": "",
                    "height": 200,
                    "asset": "img.png"
                }
            
            return render_template('analyse.html', filename=filename, condition_info=condition_info)
        else:
            message = 'Error: Only JPEG images are allowed!'
            return render_template('index.html', message=message)

    return render_template('index.html', message=message)

# New route to render analyse page with prediction
@app.route('/analyse/<filename>/<condition_info>')
def analyse_image(filename, condition_info):
    return render_template('analyse.html', filename=filename, condition_info=condition_info)

if __name__ == '__main__':
    app.run()
