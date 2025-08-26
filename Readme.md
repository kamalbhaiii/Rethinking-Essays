# 🎓 Student Assessment Preferences — Qualitative & Descriptive Analysis

This project analyzes students' preferences for different assessment formats (oral, process-based, and performance-based) using **qualitative survey data** collected through Google Forms.

It explores the research question:  
**“How do students prefer to be assessed in higher education in the age of AI?”**

This study is part of the *Human-Machine Interaction* module at **Frankfurt University of Applied Sciences**, under the broader theme:  
**“Rethinking Essays: Alternatives in Education Using AI.”**

---

## 🧠 What the Script Does

The `main.py` script automates text cleaning, clustering, and visualization for survey responses:

- ✅ Loads responses from `Sheet.xlsx`  
- ✅ Cleans and normalizes open-ended responses using NLP (tokenization + stopwords removal)  
- ✅ Generates:
  - 📊 A **pie chart** for Question 1 (fixed-choice responses)  
  - ☁️ **Word clouds** for Questions 2–8  
  - 📈 **Bar charts & heatmaps** for descriptive insights  
- ✅ Performs **theme clustering (KMeans)** on textual responses  
- ✅ Saves a **summary report** and **descriptive conclusions**

---

## 🗂️ File Structure

```
project-folder/
├── Sheet.xlsx                          # Input data (Google Form responses)
├── main.py                             # Analysis script
├── qualitative_analysis_all_questions.csv  # Processed dataset with labels
├── final_summary_report.txt            # Thematic clustering per question
├── Descriptive_Conclusion.txt          # Extended descriptive analysis & findings
├── output/                             # Charts & word clouds
└── analysis/                           # Additional analysis plots (bar charts, heatmaps)
```

---

## 🚀 How to Run the Script

1. **Set up virtual environment**  
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate    # Windows
```

2. **Install requirements**  
```bash
pip install -r requirements.txt
```

3. **Place your survey file**  
Download your Google Form responses as Excel and rename it:  
`Sheet.xlsx`

4. **Run the script**  
```bash
python main.py
```

---

## 📤 Outputs You’ll Get

| Output File | Description |
|-------------|-------------|
| `qualitative_analysis_all_questions.csv` | Cleaned responses with theme labels |
| `final_summary_report.txt` | Clustered keywords per question |
| `Descriptive_Conclusion.txt` | Human-readable conclusions for each RQ |
| `output/Question-*.png` | Word clouds & charts per question |
| `analysis/*.png` | Extra descriptive analysis (bar charts, heatmaps) |

---

## 🔍 Findings (Summary)

- **Performance-based assessments** (projects, portfolios) were the most preferred (≈49%), valued for *skills, creativity, and authenticity*.  
- **Oral assessments** (presentations, vivas) followed (≈36%), appreciated for *confidence-building and communication*, though sometimes stressful.  
- **Process-based assessments** (essays, journals) were least preferred (≈15%), criticized as *time-consuming, stressful, and vulnerable to AI/plagiarism*.  
- Students want **hybrid formats** that blend projects with oral validation for fairness and authenticity.  
- Academic dishonesty was seen as *most likely in essays*, less so in projects and oral exams.  

---

## 📝 Conclusion

Students strongly favor **performance-based and oral methods** over traditional essays.  
They call for **hybrid assessments** that:  
- Allow *practical application* (projects, teamwork, portfolios)  
- Include *oral components* (presentations/vivas) for validation  
- Reduce reliance on *essays*, which are vulnerable to AI misuse  

This reflects a shift toward **authentic, engaging, and AI-resilient assessment models** aligned with real-world competencies.

---

## 👥 Authors

- Kamal Sharma  
- Hafiz Muhammad Ali  
- Ranjit Khude  
- Tanishka Agale  
- Rency Padasala  

---

## 🙏 Acknowledgements

This project was conducted as part of the **Human-Machine Interaction** module at **Frankfurt University of Applied Sciences** during **Summer Semester 2024**.  

We thank **Prof. Dr. Valentin Schwind** for his guidance, supervision, and support.

---

## 📘 License

For academic and research use only under university guidelines.
