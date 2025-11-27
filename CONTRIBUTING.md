## Contributing to the Project

Thank you for your interest in contributing to this project! There are two main tracks to contribute to:

1. **Beginner Track** – located in the `beginner/` folder
2. **Advanced Track** – located in the `advanced/` folder

Each track supports contributions from:

* **Project Team Members**: If you're officially on the team
* **Community Contributors**: Anyone from the broader SDS community

---

## Contribution Guidelines

### 1. Verify Git Installation

Make sure that you have Git installed:

```bash
git --version
```

### 2. Fork the Repository

Fork the repository to your GitHub account using the `Fork` button in the top-right of this page.

### 3. Clone the Repository

Once forked, clone your copy to your local machine:

```bash
git clone https://github.com/YOUR_USERNAME/PROJECT-NAME
cd PROJECT-NAME
```

### 4. Set Up a Virtual Environment

Use Python `venv` or Anaconda to create a clean environment.

#### Python venv

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
```

```bash
pip install -r requirements.txt
```

#### Anaconda

```bash
conda create --name myenv python=3.12
conda activate myenv
pip install -r requirements.txt
```

---

## 📁 Submission Instructions

### ▶️ Beginner Track

Submit to the appropriate folder inside `beginner/submissions/`:

* `team-members/` for official team members
* `community-contributions/` for external contributors

**Structure Example:**

```
beginner/submissions/team-members/
│── your-name/
│   ├── data-analysis.ipynb
│   ├── app.py
│   └── requirements.txt
```

### 🔴 Advanced Track

Submit to the appropriate folder inside `advanced/submissions/`:

* `team-members/` for official team members
* `community-contributions/` for external contributors

**Structure Example:**

```
advanced/submissions/community-contributions/
│── your-name/
│   ├── model-training.ipynb
│   ├── api.py
│   └── requirements.txt
```

---

### 6. Commit and Push Your Changes

After adding your files:

```bash
git add .
git commit -m "Added my contribution to beginner track"
git push origin your-branch-name
```

### 7. Create a Pull Request (PR)

1. Go to the original SDS repo on GitHub
2. Click the `Pull Requests` tab > `New Pull Request`
3. Choose your fork and branch
4. Write a clear title + description
5. Click `Create Pull Request`

We’ll review and merge once approved!

---

### 8. Keep Your Fork Updated

Regularly pull the latest changes from the main SDS repo:

```bash
git remote add upstream https://github.com/SuperDataScience/PROJECT-NAME
git pull upstream main
```

---

### 💡 Resources

If you're new to Git & GitHub, check out our beginner-friendly course:

🔗 [Intro to Git & GitHub – SuperDataScience](https://community.superdatascience.com/c/intro-to-git-github/?preview=true)

If you have questions, open an issue or contact us at [shaheer@superdatascience.com](mailto:shaheer@superdatascience.com).

Happy contributing! 🚀
