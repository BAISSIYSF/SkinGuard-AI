# SkinGuard AI - HealthCare

**SkinGuard AI** is a skin health monitoring application powered by machine learning using the **YOLOv11n** model. The app is designed to classify various skin conditions based on uploaded images, providing users with recommendations and information about their skin health.

The application leverages computer vision to detect different types of skin conditions, including but not limited to:

- **Actinic Keratosis (76.08% mAP50)**
- **Seborrheic Keratosis (99.50% mAP50)**
- **Basal Cell Carcinoma (94.25% mAP50)**
- **Nevus (Mole) (99.50% mAP50)**
- **Pigmented Benign Keratosis (85.41% mAP50)**
- **Vascular Lesion (99.50% mAP50)**

## Features

- **Predictive Model**: Classifies various skin conditions using **YOLOv11n** (Ultralytics YOLO).
- **User-Friendly Interface**: Simple and intuitive image upload for users to interact with.
- **Condition Information**: Provides detailed descriptions and recommendations based on predicted skin conditions.
- **Product Recommendations**: Offers links to products or professional care options based on the classification results.
- **No Medical Expertise Needed**: Allows users to take a more informed approach to monitor and act on their skin health.

## Technologies Used

- **YOLOv11n (Ultralytics)**: The model used for detecting skin conditions from images. YOLOv11n is the smallest variant of the YOLOv11 model, optimized for speed and efficiency.
- **Flask**: The web framework used to handle the backend logic and serve the web interface.
- **Python**: The primary language used for implementing the application and running the model.
- **Werkzeug**: A library for handling file uploads in Python.
- **HTML/CSS/JS**: For building the frontend of the application and styling the user interface.

## Installation Instructions

Follow these steps to install and run **SkinGuard AI** locally.

### Prerequisites

Make sure you have the following installed:

- **Python 3.7+**
- **pip** (Python’s package installer)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SkinGuard-AI.git
cd SkinGuard-AI/server
```

### 2. Set up a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate  # For Windows
```

### 3. Install the Required Dependencies

Run the following command to install all necessary Python libraries:

```bash
pip install flask ultralytics Werkzeug
```

### 4. Run the Application

Start the Flask development server:

```bash
python app.py
```

This will start the server at `http://127.0.0.1:5000/` by default.

## Running the Application

1. Open a web browser and navigate to `http://127.0.0.1:5000/`.
2. You’ll see a simple form to upload an image of your skin.
3. After uploading an image, the model will classify it, and the page will display information about the condition, along with any recommendations or product links.

## Usage

- **Uploading an Image**: To use the AI-powered skin condition analysis, simply upload a clear image of your skin.
- **Receiving Recommendations**: Once the model processes the image, it will display the predicted skin condition, description, and recommendation.
- **Product Links**: If the condition is treatable with an over-the-counter product, the system will provide a link to the recommended product.

### 5. Download the Skin Cancer ISIC Dataset

This project uses a custom-trained **YOLOv11n** model specifically designed for skin condition classification. The model has been trained on the **Skin Cancer ISIC dataset**. You can download the dataset from the following link:

- **[Download YOLOv11n Model - Skin Cancer ISIC Dataset](https://drive.google.com/file/d/1bsCYjbz0YrkfJm9R6dYkTIMx-tyHbr5M/view?usp=sharing)**

## Training the Custom Model

The custom YOLOv11n model used in this project was trained on the **Skin Cancer ISIC dataset**. If you would like to explore the training process, you can find the complete training code in the `train.ipynb` notebook.

1. **No Need to Download the Dataset Manually**: The `train.ipynb` notebook contains a script that automatically downloads the **Skin Cancer ISIC dataset** for training the model. You do not need to download the dataset manually.

2. **Key Steps in the `train.ipynb` File**:
   - **Dataset Download**: The notebook automatically downloads the dataset from the ISIC archive.
   - **Converting to YOLO Format**: One of the key steps in the notebook is converting the dataset into the **YOLO format**, which is required for training with YOLOv11n. The notebook includes the necessary code to preprocess and format the dataset.
   - **Model Training**: After the data is prepared, the notebook proceeds to train the YOLOv11n model on the dataset.
   - **Model Evaluation**: The notebook also includes steps to evaluate the performance of the trained model.

Simply open and run the `train.ipynb` notebook to start the process. The notebook will handle the dataset download, preparation (including conversion to YOLO format), training, and evaluation automatically.

## License

This project is licensed under the **GNU Affero General Public License** (AGPL). The **AGPL** license ensures that the source code is freely available for use, modification, and redistribution, but it also includes a requirement that if the code is used to provide a service over a network, the source code must be made available to users of that service.

The **YOLOv11n model** used in this project is based on the Ultralytics YOLOv11n implementation, which is also licensed under the **GNU Affero General Public License**.
