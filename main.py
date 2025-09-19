import pandas as pd
from groq import Groq
import os
import re

from dotenv import load_dotenv
load_dotenv()
# -----------------------------
# Step 1: Load dataset locally
# -----------------------------

df = pd.read_csv("Data/test.csv")

print("Dataset loaded! Sample:")
print(df.head())

# -----------------------------
# Step 2: Prepare AI model
# -----------------------------

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("API Key:", os.environ.get("GROQ_API_KEY"))

def generate_rules(problem_text):
    """
    Send a prompt to Groq LLaMA 3 to get minimal math rules for a problem.
    """
    prompt = f"""Explain the minimal math rules or concepts needed to solve this problem, without giving the direct solution:
    Problem: "{problem_text}" 
    Return your answer as a short bullet-point list."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            #model="llama-3.3-70b-versatile",
            #model="groq/compound-mini",
            

            messages=[
                {"role": "system", "content": "You are a helpful math teacher."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3  # Lower temperature for more focused responses
        )
        
        # Extract the content from the response
        rules_text = response.choices[0].message.content.strip()
        return rules_text
    
    except Exception as e:
        print(f"Error generating rules for problem: {problem_text[:50]}...")
        print(f"Error: {e}")
        return "Rules generation failed"

# Remove Latex code 
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    # 1. Remove LaTeX math ($...$)
    text = re.sub(r'\$.*?\$', '', text)

    # 2. Remove LaTeX commands (\sqrt{}, \frac{}, etc.)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    # 3. Replace newlines and bullet markers
    text = text.replace('\n*', ' ')   # remove newline + bullet
    text = text.replace('\n', ' ')    # remove remaining newlines
    text = text.replace('*', '-')     # convert bullets to dashes for readability

    # 4. Remove double spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# -----------------------------
# Step 3: Generate 'rules' column
# -----------------------------
rules_list = []
total_rows = len(df)

for idx, row in df.iterrows():
    problem = row['problem']  # adjust column name if different
    print(f"Processing row {idx+1}/{total_rows}...")
    
    rules = generate_rules(problem)
    rules = clean_text(rules)     
    rules_list.append(rules)
    # print(f"rules {rules}...")
    # print(f"rules_list {clean_text(rules_list)}...")

    
    # Optional: Add a small delay to avoid rate limiting
    import time
    time.sleep(0.5)

# Add rules column to dataframe
df['rules'] = rules_list

# -----------------------------
# Step 4: Save locally
# -----------------------------
df.to_csv("math_dataset_with_rules.csv", index=False)
print("Saved new dataset with 'rules' column as 'math_dataset_with_rules.csv'")
print("\nSample of the new dataset:")
print(df[['problem', 'rules']].head())
