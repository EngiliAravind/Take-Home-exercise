# customer_feedback_analyzer.py

import pandas as pd
import openai
import time
from collections import Counter
import matplotlib.pyplot as plt
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure to set this in your environment

### Step 1: Load and Preprocess the Data ###
def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['review_body'])  # Ensure no null reviews
    df = df[['review_body']].rename(columns={'review_body': 'text'})
    return df

### Step 2: Chunk Reviews for LLM Processing ###
def chunk_reviews(df, chunk_size=5):
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = "\n".join(df['text'].iloc[i:i+chunk_size].tolist())
        chunks.append(chunk)
    return chunks

### Step 3: Use LLM to Summarize Each Chunk and Classify Sentiment ###
def summarize_chunk(chunk):
    prompt = f"""
    You are a customer experience analyst. Analyze the following customer feedback messages:

    {chunk}

    1. Summarize the top 3 issues or themes customers are talking about.
    2. Provide a sentiment breakdown: how many are Positive, Neutral, or Negative.
    3. List any common keywords or recurring terms.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print("Error in summarization:", e)
        return None

### Step 4: Run Summarization Over Chunks ###
def process_feedback(chunks):
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        summary = summarize_chunk(chunk)
        if summary:
            summaries.append(summary)
        time.sleep(1.5)  # Avoid hitting rate limits
    return summaries

### Step 5: Visualize Sentiment Breakdown ###
def extract_sentiment(summary_texts):
    sentiments = Counter()
    for summary in summary_texts:
        lines = summary.split("\n")
        for line in lines:
            if any(sent in line.lower() for sent in ["positive", "neutral", "negative"]):
                parts = line.split(":")
                if len(parts) == 2:
                    label = parts[0].strip().lower()
                    try:
                        count = int(parts[1].strip())
                        sentiments[label] += count
                    except:
                        continue
    return sentiments

def plot_sentiment(sentiments):
    labels = list(sentiments.keys())
    counts = [sentiments[label] for label in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color=['green', 'blue', 'red'])
    plt.title("Sentiment Breakdown from Customer Feedback")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("sentiment_breakdown.png")
    plt.show()

### Step 6: Save Prompt Outputs for LLM Usage Evidence ###
def save_prompt_outputs(summaries):
    with open("feedback_summary.txt", "w") as f:
        for idx, s in enumerate(summaries):
            f.write(f"=== Summary {idx+1} ===\n")
            f.write(s + "\n\n")

### Step 7: Main Pipeline ###
def main():
    filepath = "sample_amazon_reviews.csv"  # Replace with your dataset
    df = load_data(filepath)
    chunks = chunk_reviews(df, chunk_size=5)
    summaries = process_feedback(chunks)
    save_prompt_outputs(summaries)

    sentiments = extract_sentiment(summaries)
    plot_sentiment(sentiments)

if __name__ == "__main__":
    main()
