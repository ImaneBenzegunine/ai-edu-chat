import pandas as pd
import random
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline

# =============================
# 1. Load Dataset
# =============================
df = pd.read_csv("math_dataset_with_rules.csv")  # your dataset with columns: problem, level, type, solution, answer, rules



# =============================
# 2. Load Pretrained Model (DistilBERT for intent classification)
# =============================
# For now, we simulate intents with a small model (can be fine-tuned later)
intent_labels = ["solve", "ask_hint", "ask_rules", "other"]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(intent_labels))

# Wrap in pipeline for easy inference
intent_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# =============================
# 3. Chatbot Logic
# =============================
def ask_question():
    row = df.sample(1).iloc[0]
    return row["problem"], str(row["answer"]), row["solution"], row["rules"]

def classify_intent(user_input):
    result = intent_classifier(user_input)[0]
    intent = intent_labels[result["label"] if isinstance(result["label"], int) else int(result["label"].split("_")[-1])]
    return intent

def chatbot():
    print("🤖 Welcome to Math Tutor Bot!")
    print("I will give you a question. Try to solve it. Type 'hint' for help, or 'rules' to see the rules.\n")

    while True:
        problem, answer, solution, rules = ask_question()
        print(f"📝 Problem:  {problem}")

        user_input = input("👉 Your Answer (or type 'quit' to exit): ")

        if user_input.lower() == "quit":
            print("👋 Goodbye, keep practicing!")
            break

        # 1. Intent Classification
        intent = classify_intent(user_input)

        # 2. Handle Intent
        if intent == "ask_hint":
            print("💡 Hint: Think about the steps carefully. Here are the rules:")
            print(rules)

        elif intent == "ask_rules":
            print("📘 Rules used in this problem:")
            print(rules)

        else:  # Assume student is solving
            if user_input.strip() == answer.strip():
                print("✅ Correct! Well done 🎉")
                print("Here’s the solution:")
                print(solution)
            else:
                print(f"❌ Not quite. Try again")
                #print(f"❌ Not quite. The correct answer is {answer}.")
                print("Here’s how to solve it:")
                print(rules)
        print("\n----------------------\n")

# =============================
# 4. Run Chatbot
# =============================
if __name__ == "__main__":
    chatbot()
