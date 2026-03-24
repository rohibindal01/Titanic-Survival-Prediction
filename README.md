# 🚢 Titanic Survival Prediction — Machine Learning Project

A complete end-to-end ML pipeline that predicts Titanic passenger survival using the
[Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic).

---

## 📁 Project Structure

```
titanic-ml/
├── data/               ← Raw CSVs (git-ignored)
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── outputs/            ← Plots & submission CSV (auto-created)
│   ├── survival_by_class_sex.png
│   ├── age_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── submission.csv
├── model/              ← Saved model (git-ignored)
│   └── best_model.pkl
├── download_data.py    ← Step 1: Download Kaggle data
├── train.py            ← Step 2: Full ML pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Run

### 1 — Clone & create virtual environment

```bash
git clone https://github.com/<your-username>/titanic-ml.git
cd titanic-ml

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2 — Configure Kaggle API credentials

1. Go to [kaggle.com](https://www.kaggle.com) → **Account** → **Create New API Token**
2. This downloads `kaggle.json`
3. Move it to the right location:

```bash
# Linux / macOS
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell)
mkdir $env:USERPROFILE\.kaggle
Move-Item ~/Downloads/kaggle.json $env:USERPROFILE\.kaggle\
```

### 3 — Download the dataset

```bash
python download_data.py
```

This downloads and extracts `train.csv`, `test.csv`, and `gender_submission.csv` into `data/`.

### 4 — Run the ML pipeline

```bash
python train.py
```

What happens:
- **EDA** plots saved to `outputs/`
- **Feature engineering** (titles, family size, age bands, etc.)
- **Three models** trained & compared (Logistic Regression, Random Forest, Gradient Boosting)
- **Best model** selected by ROC-AUC, saved to `model/best_model.pkl`
- **Kaggle submission** CSV saved to `outputs/submission.csv`

---

## 📊 Models Compared

| Model                 | Metric         |
|-----------------------|----------------|
| Logistic Regression   | Accuracy + AUC |
| Random Forest         | Accuracy + AUC |
| Gradient Boosting     | Accuracy + AUC |

5-fold cross-validation is used for reliable estimates.

---

## 🔑 Key Features Engineered

| Feature      | Description                          |
|--------------|--------------------------------------|
| `Title`      | Extracted from passenger name        |
| `FamilySize` | SibSp + Parch + 1                    |
| `IsAlone`    | 1 if traveling alone                 |
| `AgeBand`    | Age grouped into 5 bins              |
| `FareBand`   | Fare quartile band                   |

---

## 🚀 Push to GitHub — Step-by-Step

See the [GitHub Push Guide](#-github-push-guide) section below.

---

## 📈 Sample Results

Generated plots include:
- Survival breakdown by class and sex
- Age distribution histogram
- Feature correlation heatmap
- ROC curves for all models
- Confusion matrix for the best model
- Feature importance bar chart

---

## 📜 License

MIT © 2024
