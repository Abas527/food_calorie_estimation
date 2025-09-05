

# 🍲 Food Calorie Estimation

An end-to-end **Deep Learning application** that predicts the **type of food** from an image and estimates its **calories**.
This project integrates **PyTorch**, **Streamlit**, **DVC**, and **MLflow** with remote experiment/data tracking via **Dagshub**.

---

## ✨ Features

* 📷 **Food classification** from images
* 🔥 **Calorie estimation** based on predicted class
* 📊 **Experiment tracking** with MLflow + Dagshub
* 📦 **Data versioning** with DVC
* 🎨 **Interactive web app** using Streamlit

---

## 📂 Project Structure

```bash
food_calorie_estimation/
│── app/                  # Streamlit frontend
│   └── app.py
|   └──api.py
│
│── src/                  # Source code
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── train.py
│   └── predict.py
│
│── data/                 # Data directory (tracked via DVC, not pushed to Git)
│   └── raw
|   └──processed
│
│── dvc.yaml              # DVC pipeline
│── params.yaml           # Model/training hyperparameters
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/Abas527/food_calorie_estimation.git
cd food_calorie_estimation
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Data Management (DVC + Dagshub)

This project uses **DVC** for dataset versioning and storage.

### Pull Data

```bash
dvc pull
```

### Push Data (after modifications)

```bash
dvc push
```

> 🔑 Ensure your `DAGSHUB_TOKEN` and `DAGSHUB_USER` are set in your environment variables for authentication.

---

## 🚀 Run the App

Launch the Streamlit app:

```bash
streamlit run app/app.py
```

By default, it runs on:

```
http://localhost:8501
```

---

## 🧠 Training the Model

To retrain the model from scratch:

```bash
python src/train.py
```

DVC will manage outputs, and MLflow will log experiments automatically.

---

## 📊 Experiment Tracking

* MLflow is integrated with **Dagshub**.
* You can view experiment runs, metrics, and artifacts directly in the Dagshub UI.

---

## 📝 Params

Hyperparameters are defined in `params.yaml`. Example:

```yaml
train:
  batch_size: 32
  learning_rate: 0.001
  epochs: 20
```

---

## 🤝 Contributing

Contributions are welcome!

* Fork the repo
* Create a new branch
* Submit a PR

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

* [PyTorch](https://pytorch.org/)
* [Streamlit](https://streamlit.io/)
* [DVC](https://dvc.org/)
* [Dagshub](https://dagshub.com/)
* [MLflow](https://mlflow.org/)

---
