import pandas as pd
from groq import Groq
import os
import time

# -----------------------------
# File paths
# -----------------------------
input_file = "Data/test.csv"
output_file = "math_dataset_with_rules.csv"

# -----------------------------
# Step 1: Load dataset
# -----------------------------
if os.path.exists(output_file):
    print(f"Found existing file '{output_file}', loading it...")
    df = pd.read_csv(output_file)
else:
    print(f"Loading original dataset from '{input_file}'...")
    df = pd.read_csv(input_file)
    df['rules'] = None  # initialize rules column

print("Dataset loaded! Sample:")
print(df.head())

# -----------------------------
# Step 2: Prepare AI model
# -----------------------------
client = Groq(api_key=os.environ.get("NEW_GROQ_API_KEY"))
print("API Key:", os.environ.get("NEW_GROQ_API_KEY"))

def generate_rules(problem_text):
    if pd.isna(problem_text) or not str(problem_text).strip():
        return "No problem text provided"

    prompt = f"""Explain the minimal math rules or concepts needed to solve this problem, without giving the direct solution:
    Problem: "{problem_text}" 
    Return your answer as a short bullet-point list."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful math teacher."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )

        # Extract the content from the response
        rules_text = response.choices[0].message.content.strip()
        return rules_text

    except Exception as e:
        print(f"Error generating rules for problem: {problem_text[:50]}...")
        print(f"Error: {e}")
        return "Rules generation failed"

# -----------------------------
# Step 3: Generate rules for new rows
# -----------------------------
for idx, row in df.iterrows():
    # Only process rows where rules are empty or failed
    if pd.isna(row['rules']) or row['rules'] == "Rules generation failed":
        problem = row['problem']
        print(f"Processing row {idx+1}/{len(df)}...")
        df.at[idx, 'rules'] = generate_rules(problem)
        time.sleep(0.5)  # avoid rate-limiting

# -----------------------------
# Step 4: Retry failed rows
# -----------------------------
failed_indices = df[df['rules'] == "Rules generation failed"].index
if len(failed_indices) > 0:
    print(f"Retrying {len(failed_indices)} failed rows...")
    for idx in failed_indices:
        problem = df.at[idx, 'problem']
        print(f"Retrying row {idx}...")
        rules = generate_rules(problem)
        if rules != "Rules generation failed":
            df.at[idx, 'rules'] = rules
        time.sleep(0.3)

# -----------------------------
# Step 5: Save updated dataframe
# -----------------------------
df.to_csv(output_file, index=False)
print(f"Dataset saved as '{output_file}'")
print("\nSample of the dataset:")
print(df[['problem', 'rules']].head())
