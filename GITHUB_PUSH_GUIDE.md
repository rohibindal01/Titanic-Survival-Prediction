# 🚀 GitHub Push Guide — Step-by-Step

Follow these steps to push your Titanic ML project to GitHub.

---

## ✅ Prerequisites

- Git installed → https://git-scm.com/downloads
- GitHub account → https://github.com
- (Optional) GitHub CLI → https://cli.github.com

---

## STEP 1 — Install Git (if not already)

```bash
# Check if Git is installed
git --version

# Install on Ubuntu/Debian
sudo apt install git

# Install on macOS (via Homebrew)
brew install git

# Windows: Download installer from https://git-scm.com
```

---

## STEP 2 — Configure Git (first time only)

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

---

## STEP 3 — Create a New Repo on GitHub

1. Go to https://github.com/new
2. Fill in:
   - **Repository name**: `titanic-ml`
   - **Description**: `Titanic Survival Prediction – End-to-End ML Pipeline`
   - **Visibility**: Public or Private
   - ❌ Do NOT check "Add a README" (you already have one)
3. Click **Create repository**
4. Copy the repo URL shown on the next page, e.g.:
   `https://github.com/your-username/titanic-ml.git`

---

## STEP 4 — Initialize Git in Your Project

```bash
# Navigate to your project folder
cd titanic-ml

# Initialize a git repo
git init

# Verify .gitignore is present (keeps data/ and model/ out of git)
cat .gitignore
```

---

## STEP 5 — Stage All Files

```bash
# See what files will be tracked
git status

# Stage everything
git add .

# Verify what's staged
git status
```

> 💡 `data/` and `model/` are excluded via `.gitignore` — good practice for large files.

---

## STEP 6 — Commit

```bash
git commit -m "🚀 Initial commit: Titanic ML pipeline with EDA, feature engineering, and model training"
```

---

## STEP 7 — Link Remote & Push

```bash
# Add your GitHub repo as the remote origin
git remote add origin https://github.com/your-username/titanic-ml.git

# Rename default branch to main (modern convention)
git branch -M main

# Push to GitHub!
git push -u origin main
```

You'll be prompted for your GitHub credentials (or use a Personal Access Token).

---

## STEP 8 — Authenticate (Personal Access Token)

GitHub no longer accepts passwords over HTTPS. Use a **Personal Access Token (PAT)**:

1. GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. **Generate new token** → tick `repo` scope
3. Copy the token (shown only once!)
4. Use it as your **password** when Git asks

```bash
# Or store credentials so you don't re-enter every time
git config --global credential.helper store
```

---

## STEP 9 — Verify on GitHub

Visit:
```
https://github.com/your-username/titanic-ml
```

You should see your files, README rendered, and commit history. ✅

---

## 🔄 Future Updates (Day-to-Day Workflow)

```bash
# After making changes to train.py, README, etc.
git add .
git commit -m "feat: add GridSearchCV hyperparameter tuning"
git push
```

---

## 🌿 Branching (Recommended for Experiments)

```bash
# Create a new branch for an experiment
git checkout -b feature/add-xgboost

# ... make changes ...
git add .
git commit -m "feat: add XGBoost model"
git push origin feature/add-xgboost

# Merge back into main when done
git checkout main
git merge feature/add-xgboost
git push
```

---

## 🏷️ Tagging a Release

```bash
git tag -a v1.0 -m "First working ML pipeline"
git push origin v1.0
```

---

## 🧹 Quick Reference

| Command                          | What it does                            |
|----------------------------------|-----------------------------------------|
| `git init`                       | Initialize repo                         |
| `git status`                     | Show changed/untracked files            |
| `git add .`                      | Stage all changes                       |
| `git commit -m "msg"`            | Commit staged changes                   |
| `git remote add origin <url>`    | Link to GitHub repo                     |
| `git push -u origin main`        | Push to GitHub (first time)             |
| `git push`                       | Push subsequent commits                 |
| `git pull`                       | Pull latest changes from GitHub         |
| `git log --oneline`              | View commit history                     |
| `git checkout -b <branch>`       | Create & switch to new branch           |
