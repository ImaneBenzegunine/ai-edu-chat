import pandas as pd
import openai
import os
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

# -----------------------------
# Configuration
# -----------------------------
openai.api_key = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"  # Fast and cost-effective
BATCH_SIZE = 20  # Process multiple requests concurrently
MAX_CONCURRENT_REQUESTS = 10  # Adjust based on rate limits

# -----------------------------
# Async Batch Processing
# -----------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_rules_async(session, problem_text):
    """
    Async function to generate rules for a single problem
    """
    prompt = f"""Explain the minimal math rules or concepts needed to solve this problem, without giving the direct solution. Handle LaTeX symbols properly:
    Problem: "{problem_text}" 
    Return your answer as a short bullet-point list."""
    
    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful math teacher that understands LaTeX notation."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.3
            }
        ) as response:
            
            result = await response.json()
            return result['choices'][0]['message']['content'].strip()
            
    except Exception as e:
        print(f"Error for problem: {problem_text[:50]}...")
        print(f"Error: {e}")
        return "Rules generation failed"

async def process_batch(problems_batch):
    """
    Process a batch of problems concurrently
    """
    async with aiohttp.ClientSession() as session:
        tasks = [generate_rules_async(session, problem) for problem in problems_batch]
        return await asyncio.gather(*tasks)

def generate_rules_batch(df, batch_size=BATCH_SIZE):
    """
    Process the entire dataset in batches
    """
    all_rules = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df))
        batch = df['problem'].iloc[start_idx:end_idx].tolist()
        
        print(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch)} problems)...")
        
        # Process batch asynchronously
        batch_rules = asyncio.run(process_batch(batch))
        all_rules.extend(batch_rules)
        
        # Optional: Save progress after each batch
        if batch_num % 10 == 0:
            temp_df = df.iloc[:len(all_rules)].copy()
            temp_df['rules'] = all_rules
            temp_df.to_csv("math_dataset_with_rules_partial.csv", index=False)
            print(f"Checkpoint saved after batch {batch_num + 1}")
    
    return all_rules

# -----------------------------
# Main Execution
# -----------------------------

def main():
    # Load dataset
    df = pd.read_csv("Data/test.csv")
    print(f"Dataset loaded with {len(df)} rows")
    
    # Generate rules
    rules_list = generate_rules_batch(df)
    
    # Add rules to dataframe
    df['rules'] = rules_list
    
    # Save results
    df.to_csv("math_dataset_with_rules.csv", index=False)
    print("Dataset saved with rules column")
    
    # Show sample
    print("\nSample results:")
    for i, (problem, rules) in enumerate(zip(df['problem'].head(3), df['rules'].head(3))):
        print(f"\nProblem {i+1}: {problem[:100]}...")
        print(f"Rules: {rules[:200]}...")

if __name__ == "__main__":
    main()