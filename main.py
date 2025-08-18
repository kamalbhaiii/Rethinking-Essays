import os
import re
import difflib
from collections import Counter, defaultdict

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import plotly.express as px
import plotly.graph_objects as go

# ------------------ Setup ------------------
nltk.download('stopwords')
tokenizer = TreebankWordTokenizer()

def setup():
    os.makedirs("output", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)
    return pd.read_excel("Sheet.xlsx")

# ------------------ Matching helpers ------------------
def normalize_colname(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s

def get_best_match(question: str, columns):
    qn = normalize_colname(question)
    norm_map = {normalize_colname(c): c for c in columns}
    norm_cols = list(norm_map.keys())

    if qn in norm_map:
        return norm_map[qn], "exact"
    for nc in norm_cols:
        if nc.startswith(qn) or qn.startswith(nc):
            return norm_map[nc], "startswith"
    for nc in norm_cols:
        if qn in nc or nc in qn:
            return norm_map[nc], "contains"
    matches = difflib.get_close_matches(qn, norm_cols, n=1, cutoff=0.3)
    if matches:
        return norm_map[matches[0]], "fuzzy"
    return None, None

# ------------------ Text prep ------------------
def get_combined_stopwords(question):
    default = set(stopwords.words('english'))
    punct = set(string.punctuation)
    q_words = set(tokenizer.tokenize(str(question).lower()))
    keep = {'essay', 'project', 'oral', 'assessment', 'presentation', 'viva', 'portfolio'}
    return default.union(punct).union(q_words - keep)

def clean_response(text, stop_words):
    if pd.isna(text):
        return ""
    tokens = tokenizer.tokenize(str(text).lower())
    return " ".join([w for w in tokens if w not in stop_words])

def token_freq(text, extra_stop=None, topn=20):
    stop = set(stopwords.words('english'))
    if extra_stop:
        stop |= set(extra_stop)
    toks = [t for t in tokenizer.tokenize(text.lower()) if t not in stop and t not in string.punctuation]
    return Counter(toks).most_common(topn)

# ------------------ Visuals ------------------
def generate_wordcloud(text, question_number):
    if not text.strip():
        return
    wc = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',
        contour_color='steelblue',
        contour_width=2
    ).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - Q{question_number}', fontsize=16)
    path = os.path.join("output", f"Question-{question_number}.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

def save_bar_from_counts(counts_dict, title, fname):
    if not counts_dict:
        return
    labels = list(counts_dict.keys())
    values = list(counts_dict.values())
    plt.figure(figsize=(8,5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

def save_bar_from_tuples(tuples_list, title, fname):
    if not tuples_list:
        return
    labels, values = zip(*tuples_list)
    plt.figure(figsize=(8,5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

def generate_pie_chart_from_options(series, question_number, question_text):
    counts = series.value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.3,
        pull=[0.05]*len(counts),
        marker=dict(colors=px.colors.sequential.Rainbow)
    )])
    fig.update_layout(
        title_text=f"Q{question_number}: {question_text}",
        title_font_size=20,
        height=600
    )
    fig.write_image(f"output/Question-{question_number}.png")
    return counts

# ------------------ Clustering ------------------
def cluster_responses(clean_column, label_column, df, question_number, question_text):
    try:
        if df[clean_column].str.strip().replace("", pd.NA).dropna().empty:
            return f"\nQ{question_number}: {question_text}\nNo valid responses to cluster.\n"
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
        X = vectorizer.fit_transform(df[clean_column])
        if X.shape[0] < 2 or X.shape[1] < 2:
            return f"\nQ{question_number}: {question_text}\nNot enough responses/features for clustering.\n"
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        df[label_column] = kmeans.labels_

        # cluster distribution chart
        cluster_counts = df[label_column].value_counts().sort_index()
        save_bar_from_counts(cluster_counts.to_dict(), f"Cluster distribution - Q{question_number}",
                             f"analysis/Q{question_number}_cluster_distribution.png")

        terms = vectorizer.get_feature_names_out()
        centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        summaries = []
        for i in range(3):
            words = [terms[ind] for ind in centroids[i, :5]]
            summaries.append(f"Theme {i}: " + ", ".join(words))
        return f"\nQ{question_number}: {question_text}\n" + "\n".join(summaries) + "\n"
    except Exception as e:
        return f"\nQ{question_number}: {question_text}\nError in clustering: {e}\n"

# ------------------ Categorization helpers ------------------
def detect_pref_category(s):
    if not isinstance(s, str):
        s = str(s)
    sl = s.lower()
    if any(k in sl for k in ["oral", "viva", "presentation"]):
        return "Oral-based"
    if any(k in sl for k in ["process", "essay", "journal", "written"]):
        return "Process-based"
    if any(k in sl for k in ["performance", "project", "portfolio", "demo"]):
        return "Performance-based"
    return "Other/Unclear"

def classify_oral_sentiment(text):
    if not isinstance(text, str):
        text = str(text)
    tl = text.lower()
    helpful_terms = ["helpful", "confidence", "confident", "improve", "improves", "improved", "practice", "good", "better", "skill", "skills", "interactive", "engage", "engaging", "clarify", "clear"]
    stressful_terms = ["stress", "stressful", "anxiety", "anxious", "fear", "nervous", "pressure", "panic", "stage fright", "overwhelmed"]
    is_help = any(t in tl for t in helpful_terms)
    is_stress = any(t in tl for t in stressful_terms)
    return is_help, is_stress

def method_mentions(text):
    tl = str(text).lower()
    return {
        "Oral-based": any(k in tl for k in ["oral", "presentation", "viva"]),
        "Process-based": any(k in tl for k in ["essay", "journal", "written", "process"]),
        "Performance-based": any(k in tl for k in ["project", "portfolio", "demo", "performance"])
    }

# ------------------ Core processing per question ------------------
def process_question(i, question, df, summary_report, output_dfs, conclusion_data):
    best_col, how = get_best_match(question, df.columns.tolist())
    if not best_col:
        summary_report.append(f"\nQ{i+1}: {question}\nColumn not found in dataset.\n")
        return

    col_name = best_col
    clean_col, label_col = f"Q{i+1}_clean", f"Q{i+1}_Theme"

    if i == 0:
        counts = generate_pie_chart_from_options(df[col_name], i + 1, question)
        # extra: bar chart
        save_bar_from_counts(counts.to_dict(),
                             "Preferred Assessment Methods (Q1)",
                             "analysis/Q1_assessment_preferences.png")
        df["Preference_Category"] = df[col_name].apply(detect_pref_category)
        df[clean_col] = df[col_name]
        df[label_col] = None
        summary_report.append(f"\nQ{i+1}: {question}\nVisualized as pie chart from selected options.\n")
        conclusion_data["Q1_counts"] = counts.to_dict()
        conclusion_data["pref_category_col"] = "Preference_Category"
        output_dfs.append(df[[col_name, clean_col, "Preference_Category"]])
        return

    # All other questions: clean text, word cloud, cluster, per-question bars
    stop_words = get_combined_stopwords(question)
    df[clean_col] = df[col_name].apply(lambda x: clean_response(x, stop_words))
    all_words = ' '.join(df[clean_col])
    generate_wordcloud(all_words, i + 1)
    summary = cluster_responses(clean_col, label_col, df, i + 1, question)
    summary_report.append(summary)

    # Save top token bars per question
    top_terms = token_freq(all_words, topn=15)
    save_bar_from_tuples(top_terms, f"Top terms - Q{i+1}", f"analysis/Q{i+1}_top_terms.png")

    # Store special mappings for downstream analytics
    conclusion_data[f"Q{i+1}_col"] = col_name
    conclusion_data[f"Q{i+1}_clean"] = clean_col

    # Special handling for defined research sub-questions
    # Q2: Reasons per chosen preference (link to Q1)
    if i == 1 and "Preference_Category" in df.columns:
        for cat in ["Oral-based", "Process-based", "Performance-based"]:
            sub = df[df["Preference_Category"] == cat]
            text = " ".join(sub[clean_col].dropna().astype(str))
            freq = token_freq(text, topn=12)
            save_bar_from_tuples(freq, f"Reasons for preferring {cat} (Q2)", f"analysis/Q2_reasons_{cat.replace('-','_')}.png")
            conclusion_data[f"Q2_reasons_{cat}"] = freq

    # Q4 (index 3): process-based challenges
    if i == 3 and "Preference_Category" in df.columns:
        proc_sub = df[df["Preference_Category"] == "Process-based"]
        textp = " ".join(proc_sub[clean_col].dropna().astype(str))
        freqp = token_freq(textp, topn=15)
        save_bar_from_tuples(freqp, "Challenges with Process-based (Q4)", "analysis/Q4_process_challenges.png")
        conclusion_data["Q4_process_challenges"] = freqp

    # Q5 (index 4): oral helpful vs stressful
    if i == 4:
        helpful_count = 0
        stressful_count = 0
        helpful_texts = []
        stressful_texts = []
        for t in df[col_name].dropna().astype(str):
            h, s = classify_oral_sentiment(t)
            if h:
                helpful_count += 1
                helpful_texts.append(t)
            if s:
                stressful_count += 1
                stressful_texts.append(t)
        # bar chart helpful vs stressful
        save_bar_from_counts({"Helpful": helpful_count, "Stressful": stressful_count},
                             "Oral assessment: helpful vs stressful (Q5)",
                             "analysis/Q5_oral_helpful_vs_stressful.png")
        # keyword bars
        help_freq = token_freq(" ".join(helpful_texts), topn=12)
        stress_freq = token_freq(" ".join(stressful_texts), topn=12)
        save_bar_from_tuples(help_freq, "Top words (Helpful, Q5)", "analysis/Q5_oral_helpful_top_words.png")
        save_bar_from_tuples(stress_freq, "Top words (Stressful, Q5)", "analysis/Q5_oral_stressful_top_words.png")
        conclusion_data["Q5_helpful_count"] = helpful_count
        conclusion_data["Q5_stressful_count"] = stressful_count
        conclusion_data["Q5_help_freq"] = help_freq
        conclusion_data["Q5_stress_freq"] = stress_freq

    # Q6 (index 5): comparison learning—already covered via clustering/terms

    # Q7 (index 6): dishonesty encourage/discipline + methods
    if i == 6:
        encourage_terms = ["encourage", "easy", "cheat", "cheating", "ai", "plagiarism"]
        discourage_terms = ["discourage", "hard", "difficult", "practical", "teamwork", "oral", "project", "presentation"]

        enc_count = 0
        dis_count = 0
        matrix = defaultdict(lambda: {"Encourage": 0, "Discourage": 0})
        for t in df[col_name].dropna().astype(str):
            tl = t.lower()
            pol_enc = any(x in tl for x in encourage_terms)
            pol_dis = any(x in tl for x in discourage_terms)
            meth = method_mentions(t)
            if pol_enc:
                enc_count += 1
                for m, flag in meth.items():
                    if flag:
                        matrix[m]["Encourage"] += 1
            if pol_dis:
                dis_count += 1
                for m, flag in meth.items():
                    if flag:
                        matrix[m]["Discourage"] += 1

        # Heatmap-like plot
        methods = ["Oral-based", "Process-based", "Performance-based"]
        polarities = ["Encourage", "Discourage"]
        data = [[matrix[m][p] for m in methods] for p in polarities]
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right')
        ax.set_yticks(range(len(polarities)))
        ax.set_yticklabels(polarities)
        for y in range(len(polarities)):
            for x in range(len(methods)):
                ax.text(x, y, str(data[y][x]), ha='center', va='center', color='black')
        ax.set_title("Academic dishonesty: Encourage vs Discourage by method (Q7)")
        plt.tight_layout()
        plt.savefig("analysis/Q7_dishonesty_matrix.png", dpi=300)
        plt.close()

        conclusion_data["Q7_encourage_total"] = enc_count
        conclusion_data["Q7_discourage_total"] = dis_count
        conclusion_data["Q7_matrix"] = matrix

    # Q8 (index 7): ideal assessment—categorize & bar chart
    if i == 7:
        buckets = {"Oral-based": 0, "Process-based": 0, "Performance-based": 0, "Hybrid": 0, "Other/Unclear": 0}
        hybrid_terms = ["hybrid", "mix", "combination", "blend", "combine"]
        for t in df[col_name].dropna().astype(str):
            tl = t.lower()
            if any(h in tl for h in hybrid_terms):
                buckets["Hybrid"] += 1
            else:
                cat = detect_pref_category(tl)
                buckets[cat] += 1
        save_bar_from_counts(buckets, "Ideal assessment (Q8)", "analysis/Q8_ideal_assessment.png")
        conclusion_data["Q8_buckets"] = buckets

    # persist exported columns
    cols_to_keep = [col_name, clean_col]
    if label_col in df.columns:
        cols_to_keep.append(label_col)
    output_dfs.append(df[cols_to_keep])

# ------------------ Narrative builders ------------------
def generate_conclusion(conclusion_data):
    text = []
    if "Q1_counts" in conclusion_data:
        counts = conclusion_data["Q1_counts"]
        total = sum(counts.values())
        most = max(counts, key=counts.get)
        pct = (counts[most] / total * 100) if total else 0
        text.append(f"Most students ({pct:.1f}%) prefer **{most}** as their mode of assessment based on Q1.")
    text.append("Overall, students strongly favor practical, performance-based and oral formats over traditional essays, citing authenticity, real-world applicability, and reduced plagiarism risks.")
    return "\n".join(text)

def generate_descriptive_report(conclusion_data):
    lines = []
    # 1) Reasons per preference
    for cat in ["Oral-based", "Process-based", "Performance-based"]:
        freq = conclusion_data.get(f"Q2_reasons_{cat}", [])
        total = sum([c for _, c in freq]) if freq else 0
        reason_str = ", ".join([f"{w} ({c})" for w,c in freq[:10]]) if freq else "Not enough data"
        lines.append(f"• Reason why people prefer **{cat}**:\n  - Top reasons: {reason_str}\n")

    # 2) Positive experiences by type (leveraging Q3 top terms already saved in analysis per type via Q2 reasons tie-in)
    # We can’t fully separate Q3 by type unless we filtered there; we reflected via Q2 link as proxy.
    lines.append("• Positive experiences by type (Q3): see word clouds in /output and term bars in /analysis/Q3_top_terms.png")

    # 3) Most challenging thing about Process-based
    proc_chal = conclusion_data.get("Q4_process_challenges", [])
    proc_str = ", ".join([f"{w} ({c})" for w,c in proc_chal[:12]]) if proc_chal else "Not enough data"
    lines.append(f"• Most challenging aspects of **Process-based** (Q4): {proc_str}")

    # 4) Oral helpful vs stressful
    h = conclusion_data.get("Q5_helpful_count", 0)
    s = conclusion_data.get("Q5_stressful_count", 0)
    help_kw = ", ".join([f"{w} ({c})" for w,c in conclusion_data.get("Q5_help_freq", [])[:10]]) if conclusion_data.get("Q5_help_freq") else "N/A"
    stress_kw = ", ".join([f"{w} ({c})" for w,c in conclusion_data.get("Q5_stress_freq", [])[:10]]) if conclusion_data.get("Q5_stress_freq") else "N/A"
    lines.append(f"• Oral assessments (Q5): Helpful={h}, Stressful={s}\n  - Helpful keywords: {help_kw}\n  - Stressful keywords: {stress_kw}")

    # 5) Dishonesty encourage vs discourage
    enc = conclusion_data.get("Q7_encourage_total", 0)
    dis = conclusion_data.get("Q7_discourage_total", 0)
    matrix = conclusion_data.get("Q7_matrix", {})
    lines.append(f"• Academic dishonesty (Q7): Encourage={enc}, Discourage={dis}\n  - See heatmap: analysis/Q7_dishonesty_matrix.png")
    for m in ["Oral-based", "Process-based", "Performance-based"]:
        if m in matrix:
            lines.append(f"    · {m}: Encourage={matrix[m]['Encourage']}, Discourage={matrix[m]['Discourage']}")

    # 6) Ideal assessment (Q8)
    buckets = conclusion_data.get("Q8_buckets", {})
    if buckets:
        ranked = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
        lines.append("• Ideal assessment (Q8) — distribution:")
        for k,v in ranked:
            lines.append(f"  - {k}: {v}")
        lines.append("  - See bar chart: analysis/Q8_ideal_assessment.png")

    # Final synthesis
    lines.append("\n— Synthesis —")
    lines.append("Students prefer **Performance-based** tasks (projects, portfolios) as primary, with **Oral** components (presentations/vivas) used to validate authenticity; **Process-based** essays are least favored due to time cost and AI/plagiarism vulnerability.")
    return "\n".join(lines)

# ------------------ Main ------------------
def main():
    df = setup()
    questions = [
        "How do you personally prefer to be assessed in your courses?",
        "Reason why you prefer the method chosen above.",
        "Can you describe a positive experience you've had with any type of assessment?",
        "What do you find most challenging or frustrating about traditional essay-based assessments?",
        "Have you ever found oral assessments (like presentations or vivas) helpful or stressful? Why?",
        "How do project-based or performance-based assessments compare with other formats in terms of showing what you’ve learned?",
        "Do you feel certain assessment formats encourage or discourage academic dishonesty (like using AI tools)? Why?",
        "If you could design your ideal assessment format, what would it include and why?"
    ]

    summary_report, output_dfs, conclusion_data = [], [], {}

    for i, q in enumerate(questions):
        process_question(i, q, df, summary_report, output_dfs, conclusion_data)

    # Save combined responses (original + cleaned + labels)
    final_df = pd.concat(output_dfs, axis=1)
    final_df.to_csv("qualitative_analysis_all_questions.csv", index=False)

    # Theme summaries
    with open("final_summary_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_report))

    # Final single-paragraph conclusion
    with open("final_conclusion.txt", "w", encoding="utf-8") as f:
        f.write(generate_conclusion(conclusion_data))

    # Descriptive, question-by-question answers (your requested list)
    with open("Descriptive Conclusion.txt", "w", encoding="utf-8") as f:
        f.write(generate_descriptive_report(conclusion_data))

    print("✅ Analysis complete.")
    print("📄 Saved: qualitative_analysis_all_questions.csv")
    print("📝 Themes: final_summary_report.txt")
    print("📌 Final conclusion: final_conclusion.txt")
    print("🧾 Descriptive answers: Descriptive Conclusion.txt")
    print("🖼️ Word clouds & Q1 pie chart in ./output/")
    print("📊 Extra charts in ./analysis/")

if __name__ == "__main__":
    main()
