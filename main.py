import os
import re
import difflib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

nltk.download('stopwords')
tokenizer = TreebankWordTokenizer()

def setup():
    os.makedirs("output", exist_ok=True)
    return pd.read_excel("Sheet.xlsx")

def normalize_colname(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)           # collapse all whitespace/newlines
    s = s.strip()
    return s

def get_best_match(question: str, columns):
    qn = normalize_colname(question)

    # Build normalized mapping
    norm_map = {normalize_colname(c): c for c in columns}
    norm_cols = list(norm_map.keys())

    # 1) exact normalized
    if qn in norm_map:
        return norm_map[qn], "exact"

    # 2) startswith
    for nc in norm_cols:
        if nc.startswith(qn) or qn.startswith(nc):
            return norm_map[nc], "startswith"

    # 3) contains
    for nc in norm_cols:
        if qn in nc or nc in qn:
            return norm_map[nc], "contains"

    # 4) fuzzy (looser cutoff)
    matches = difflib.get_close_matches(qn, norm_cols, n=1, cutoff=0.3)
    if matches:
        return norm_map[matches[0]], "fuzzy"

    return None, None

def get_combined_stopwords(question):
    default = set(stopwords.words('english'))
    punct = set(string.punctuation)
    q_words = set(tokenizer.tokenize(question.lower()))
    keep = {'essay', 'project', 'oral', 'assessment'}
    return default.union(punct).union(q_words - keep)

def clean_response(text, stop_words):
    if pd.isna(text):
        return ""
    tokens = tokenizer.tokenize(str(text).lower())
    return " ".join([word for word in tokens if word not in stop_words])

def generate_wordcloud(text, question_number):
    if not text.strip():
        return
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',
        contour_color='steelblue',
        contour_width=2
    ).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - Q{question_number}', fontsize=16)
    path = os.path.join("output", f"Question-{question_number}.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
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
    # NOTE: Plotly deprecation warns about engine param; Kaleido is already the engine used.
    fig.write_image(f"output/Question-{question_number}.png")

def cluster_responses(clean_column, label_column, df, question_number, question_text):
    try:
        # guard for empty text
        if df[clean_column].str.strip().replace("", pd.NA).dropna().empty:
            return f"\nQ{question_number}: {question_text}\nNo valid responses to cluster.\n"

        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
        X = vectorizer.fit_transform(df[clean_column])
        if X.shape[0] < 2 or X.shape[1] < 2:
            return f"\nQ{question_number}: {question_text}\nNot enough responses/features for clustering.\n"

        kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        df[label_column] = kmeans.labels_

        terms = vectorizer.get_feature_names_out()
        centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        summaries = []
        for i in range(3):
            words = [terms[ind] for ind in centroids[i, :5]]
            summaries.append(f"Theme {i}: " + ", ".join(words))
        return f"\nQ{question_number}: {question_text}\n" + "\n".join(summaries) + "\n"
    except Exception as e:
        return f"\nQ{question_number}: {question_text}\nError in clustering: {e}\n"

def process_question(i, question, df, summary_report, output_dfs):
    best_col, how = get_best_match(question, df.columns.tolist())

    if not best_col:
        summary_report.append(f"\nQ{i+1}: {question}\nColumn not found in dataset.\n")
        return

    col_name = best_col
    clean_col, label_col = f"Q{i+1}_clean", f"Q{i+1}_Theme"

    if i == 0:
        generate_pie_chart_from_options(df[col_name], i + 1, question)
        df[clean_col] = df[col_name]
        df[label_col] = None
        output_dfs.append(df[[col_name, clean_col]])
        summary_report.append(f"\nQ{i+1}: {question}\nVisualized as pie chart from selected options.\n")
        return

    stop_words = get_combined_stopwords(question)
    df[clean_col] = df[col_name].apply(lambda x: clean_response(x, stop_words))
    all_words = ' '.join(df[clean_col])
    generate_wordcloud(all_words, i + 1)
    summary = cluster_responses(clean_col, label_col, df, i + 1, question)
    summary_report.append(summary)
    output_dfs.append(df[[col_name, clean_col] + ([label_col] if label_col in df.columns else [])])

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

    summary_report, output_dfs = [], []

    for i, q in enumerate(questions):
        process_question(i, q, df, summary_report, output_dfs)

    final_df = pd.concat(output_dfs, axis=1)
    final_df.to_csv("qualitative_analysis_all_questions.csv", index=False)

    with open("final_summary_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_report))

    print("✅ Analysis complete.")
    print("📄 Saved: qualitative_analysis_all_questions.csv")
    print("📝 Report: final_summary_report.txt")
    print("🖼️ High-quality visuals saved in ./output/")

if __name__ == "__main__":
    main()