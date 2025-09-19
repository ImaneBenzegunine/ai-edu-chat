import pandas as pd
import math
import re
from collections import defaultdict

# =============================
# 1) Load Dataset
# =============================
# Required columns: problem, level, solution, answer, rules
df = pd.read_csv("math_dataset_with_rules.csv")

# =============================
# 2) Level utils
# =============================
def level_num(level_str: str) -> int:
    """Extract integer from strings like 'Level 1', 'Level 2'."""
    m = re.search(r"(\d+)", str(level_str))
    return int(m.group(1)) if m else 0

def level_str(n: int) -> str:
    return f"Level {n}"

# Normalize available levels (sorted numerically)
available_levels_nums = sorted({level_num(lv) for lv in df["level"].dropna()})
available_levels = [level_str(n) for n in available_levels_nums]

def clamp_to_available(num: int) -> int:
    """Clamp desired level number to the nearest available within bounds."""
    if not available_levels_nums:
        return num
    return max(min(num, available_levels_nums[-1]), available_levels_nums[0])

# Keep a per-level set of used row indices to avoid repeats
used_by_level = defaultdict(set)

# =============================
# 3) Sampling without replacement per level
# =============================
def sample_row_for_level(level_label: str):
    level_mask = df["level"].astype(str) == level_label
    candidates = df[level_mask].copy()

    if candidates.empty:
        return None

    # remove used indices for this level
    remaining = candidates[~candidates.index.isin(used_by_level[level_label])]
    if remaining.empty:
        # reset if exhausted (or you can switch levels)
        used_by_level[level_label].clear()
        remaining = candidates

    row = remaining.sample(1).iloc[0]
    used_by_level[level_label].add(row.name)
    return row

# =============================
# 4) Answer matching
# =============================
def _to_float(s):
    try:
        return float(str(s).strip())
    except Exception:
        return None

def answers_match(user_input: str, answer: str, tol: float = 1e-6) -> bool:
    ui = user_input.strip()
    ans = str(answer).strip()
    u_num = _to_float(ui)
    a_num = _to_float(ans)
    if u_num is not None and a_num is not None:
        return math.isclose(u_num, a_num, rel_tol=0, abs_tol=tol)
    # fallback string compare (case/space insensitive)
    norm = lambda x: "".join(str(x).split()).lower()
    return norm(ui) == norm(ans)

# =============================
# 5) Adaptive move prompt
# =============================
def ask_move(correct: bool, current_lvl_num: int) -> int:
    """
    If correct: allow 'same' or 'up' (default: up).
    If wrong:   allow 'same' or 'down' (default: same).
    """
    low = available_levels_nums[0]
    high = available_levels_nums[-1]

    if correct:
        # options: same / up
        default = "up"
        opts_text = "[same/up]"
        while True:
            choice = input(f"🎯 Level change? {opts_text} (Enter = {default}): ").strip().lower()
            if choice == "":
                choice = default
            if choice in {"same", "up"}:
                break
            print("Please type 'same' or 'up' (or press Enter).")
        if choice == "same":
            return current_lvl_num
        else:
            return min(current_lvl_num + 1, high)
    else:
        # options: same / down
        default = "same"
        opts_text = "[same/down]"
        while True:
            choice = input(f"🔁 Level change? {opts_text} (Enter = {default}): ").strip().lower()
            if choice == "":
                choice = default
            if choice in {"same", "down"}:
                break
            print("Please type 'same' or 'down' (or press Enter).")
        if choice == "same":
            return current_lvl_num
        else:
            return max(current_lvl_num - 1, low)

# =============================
# 6) Chatbot
# =============================
def chatbot():
    print("🤖 Welcome to Math Tutor Bot (Adaptive Levels)!")
    print(f"Available levels: {', '.join(available_levels)}")
    print("Type 'hint' for help, 'rules' to see the rules, or 'quit' to exit.\n")

    # --- First question: truly random across dataset ---
    first_row = df.sample(1).iloc[0]
    current_level_num = level_num(first_row["level"])
    current_level_num = clamp_to_available(current_level_num)  # just in case
    current_level = level_str(current_level_num)

    # Mark first as used for its level
    used_by_level[current_level].add(first_row.name)

    try:
        row = first_row
        while True:
            # If row is None (edge cases), sample from current level
            if row is None:
                current_level = level_str(current_level_num)
                row = sample_row_for_level(current_level)
                if row is None:
                    print(f"⚠️ No questions found for {current_level}. Exiting.")
                    return

            problem = str(row.get("problem", ""))
            answer = str(row.get("answer", ""))
            solution = str(row.get("solution", "") or "")
            rules = str(row.get("rules", "") or "")
            current_level = level_str(current_level_num)

            print(f"📝 [{current_level}] Problem: {problem}")

            attempts = 0
            gave_hint = False

            while True:
                user_input = input("👉 Your answer (or 'hint' / 'rules' / 'quit'): ").strip()

                if user_input.lower() in {"quit", "exit"}:
                    print("👋 Goodbye, keep practicing!")
                    return

                if "hint" in user_input.lower():
                    if not gave_hint:
                        print("💡 Hint: Identify what is asked; isolate the unknown; apply the key rule.")
                        gave_hint = True
                    else:
                        print("💡 Extra hint (from rules):")
                        print(rules if rules.strip() else "(No rules provided.)")
                    continue

                if "rule" in user_input.lower():
                    print("📘 Rules used in this problem:")
                    print(rules if rules.strip() else "(No rules provided.)")
                    continue

                # Solve attempt
                attempts += 1
                correct = answers_match(user_input, answer)

                if correct:
                    print("✅ Correct! Well done 🎉")
                    if solution.strip():
                        print("Here’s a solution:")
                        print(solution)
                else:
                    if attempts < 2:
                        print("❌ Not quite. Try again!")
                        continue
                    elif attempts == 2:
                        print("❌ Still off. Here’s a nudge from the rules:")
                        print(rules if rules.strip() else "(No rules provided.)")
                        continue
                    else:
                        print(f"❌ The correct answer is: {answer}")
                        if solution.strip():
                            print("Here’s one way to solve it:")
                            print(solution)

                # After correctness decided, ask move and fetch next question at new level
                next_level_num = ask_move(correct=correct, current_lvl_num=current_level_num)
                current_level_num = next_level_num
                row = sample_row_for_level(level_str(current_level_num))
                print("\n----------------------\n")
                break

    except KeyboardInterrupt:
        print("\n👋 Goodbye, keep practicing!")

# =============================
# 7) Run
# =============================
if __name__ == "__main__":
    chatbot()
