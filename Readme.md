
# 🎓 Student Assessment Preferences — Qualitative Analysis

This project analyzes students' preferences for various assessment formats (e.g., oral exams, essays, projects) using **qualitative data** collected through a Google Form.

It explores the research question:  
**“How do students want to be assessed in higher education?”**

This study is part of the *Human-Machine Interaction* module at Frankfurt University of Applied Sciences, with the broader research focus:  
**“Rethinking Essays: Alternatives in Education Using AI.”**

---

## 🧠 What the Script Does

The `main.py` Python script automates the analysis of all open-ended questions from the collected responses:

- ✅ Loads the responses from an Excel file (`Sheet.xlsx`)
- ✅ Cleans textual data using NLP techniques
- ✅ Generates:
  - 📊 A **3D-style pie chart** for Question 1 (which has fixed options)
  - ☁️ **High-resolution word clouds** for open-ended responses (Q2–Q7)
- ✅ Uses **unsupervised clustering** (KMeans) to identify **themes** in student responses
- ✅ Creates a **summary report** of extracted themes per question
- ✅ Outputs everything in a clean, visual format

---

## 🗂️ File Structure

```
project-folder/
├── Sheet.xlsx                          # Input file (Google Form responses)
├── main.py                             # Main analysis script
├── qualitative_analysis_all_questions.csv  # Final dataset with labels
├── final_summary_report.txt            # Thematic summary per question
├── output/
│   ├── Question-1.png                  # 3D pie chart (Q1)
│   ├── Question-2.png to Question-7.png# Word clouds (Q2–Q7)
```

---

## 🚀 How to Run the Script

### 1. Make a virtual environment

Run this in your terminal:

```bash
python -m venv .venv
```

### 1. Activate the virtual environment

Run this in your terminal:

```bash
source .venv/Scripts/activate
```

### 3. Install the Required Libraries

Run this in your terminal:

```bash
pip install -r requirements.txt
```

### 4. Prepare Your Excel File

Ensure your Google Form responses are downloaded and saved as:  
```Sheet.xlsx```

Place it in the same folder as `main.py`.

### 5. Run the Script

```bash
python main.py
```

---

## 📤 Outputs You’ll Get

| Output File | Description |
|-------------|-------------|
| `qualitative_analysis_all_questions.csv` | Cleaned responses + theme labels |
| `final_summary_report.txt` | Theme keywords (e.g., “Theme 0: project, portfolio…”) |
| `output/Question-1.png` | Donut-style 3D pie chart of Q1 choices |
| `output/Question-2.png to 7.png` | Word clouds per open-ended question |

---

## 🔍 How to Analyze the Output

### Pie Chart (Q1)
Shows how students responded to:
> “How do you personally prefer to be assessed in your courses?”

Useful to **quantify preference trends** (e.g., projects vs. essays).

---

### Word Clouds (Q2–Q7)
Give a quick visual overview of:
- Most frequent terms in responses
- What students emphasize (e.g., fairness, AI, stress)

---

### Summary Report
Located in: `final_summary_report.txt`

Each question will have 3 identified **themes** using KMeans clustering:
```text
Q2: Can you describe a positive experience...
Theme 0: presentation, confidence, explained
Theme 1: feedback, personal, engagement
Theme 2: understanding, fun, project
```

You can manually interpret and rename these themes for your final paper/report.

---

## 👥 Authors

- Kamal Sharma [kamal.sharma@stud.fra-uas.de]
- Hafiz Muhammad Ali [hafiz.ali2@stud.fra-uas.de]
- Ranjit Khude [ranjit.khude@stud.fra-uas.de]
- Tanishka Agale [tanishka.agale@stud.fra-uas.de]
- Rency Padasala [rency.padasala@stud.fra-uas.de]

---

## 🙏 Acknowledgements

This research project was conducted as part of the **Human Machine Interaction** module at **Frankfurt University of Applied Sciences**, during the **Summer Semester 2024**.

We would like to express our sincere gratitude to **Prof. Valentin Schwind [valentin.schwind@fb2.fra-uas.de]** for his guidance, support, and valuable feedback throughout this study.

---

## 📘 License

This project is for academic purposes only and is part of a university module submission. Use with credit.
