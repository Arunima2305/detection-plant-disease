
# Plant Disease Detection System

This repository contains a Streamlit-based web application for detecting plant diseases from uploaded images. The system utilizes a deep learning model trained on a comprehensive dataset to identify various plant diseases, assisting farmers, researchers, and agricultural experts in promoting sustainable agriculture.

---

## Features
- Disease detection for a wide range of plants and diseases.
- Jupyter Notebook included for model training and evaluation.
- User-friendly interface for seamless interaction.
- Real-time predictions powered by TensorFlow.

---

## Live Demo
The app is deployed and accessible at: [Plant Disease Detection System](https://arunima2305-detection-plant-disease-app-2c9to6.streamlit.app/)

---

## Technologies Used
- **Frontend**: Streamlit
- **Machine Learning Framework**: TensorFlow
- **Preprocessing**: OpenCV, NumPy
- **Model Training and Evaluation**: Jupyter Notebook
- **Hosting**: Streamlit Cloud

---

## Supported Plant Diseases
The application can identify the following plant diseases:
- Apple: Scab, Black Rot, Rust, Healthy
- Blueberry: Healthy
- Cherry: Powdery Mildew, Healthy
- Corn: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- Grape: Black Rot, Esca, Leaf Blight, Healthy
- Orange: Haunglongbing (Citrus Greening)
- Peach: Bacterial Spot, Healthy
- Potato: Early Blight, Late Blight, Healthy
- Tomato: Various diseases, including Bacterial Spot, Early Blight, and Tomato Mosaic Virus

---

## How to Run Locally
Follow these steps to set up and run the app locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Arunima2305/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   Launch the app using Streamlit:
   ```bash
   streamlit run app.py
   ```

4. **Access the App**:
   Open your browser and go to:
   ```
   http://localhost:8501
   ```

---

## Model Training and Evaluation
The repository includes a Jupyter Notebook for training and evaluating the deep learning model used in the application. To explore or retrain the model:

1. Open the `model_training.ipynb` file in Jupyter Notebook.
2. Ensure you have the necessary dependencies installed (see `requirements.txt`).
3. Follow the instructions in the notebook to train or test the model on your dataset.

---

## Environment Variables
To securely use the app, ensure you configure your `secrets.toml` file for sensitive information:

1. Create a `.streamlit/secrets.toml` file:
   ```toml
   [default]
   API_TOKEN = "your-huggingface-api-token"
   ```

2. Add the `.streamlit/secrets.toml` file to `.gitignore` to avoid exposing it.

---

## Future Enhancements
- Add support for mobile-friendly layouts.
- Allow users to upload custom datasets for training.
- Integrate detailed reports on disease severity and treatment suggestions.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contributing
Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add feature-name'
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request.


