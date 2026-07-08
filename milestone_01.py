"""
Milestone 1: Probability Foundations & Variable Types

Author: Renita Parumal

Description:
Computes sample spaces, basic probabilities, conditional probabilities,
verifies Bayes' theorem, calculates an odds ratio,
and classifies random variables using the FinFlow dataset.
"""

import pandas as pd

# ==========================================================
# LOAD DATA
# ==========================================================

try:
    df = pd.read_csv("data/finflow_users.csv")
except FileNotFoundError:
    print("Error: data/finflow_users.csv was not found.")
    exit()

# ==========================================================
# PART 1: SAMPLE SPACE & BASIC PROBABILITIES
# ==========================================================

sample_space_premium = set(df["premium_user"])

p_premium = (df["premium_user"] == 1).mean()

p_high_engagement = (df["score_views"] >= 5).mean()

p_aggressive = (df["risk_profile"] == "aggressive").mean()

p_joint = (
    ((df["score_views"] >= 5) &
     (df["premium_user"] == 1))
).mean()

# ==========================================================
# PART 2: CONDITIONAL PROBABILITY & BAYES
# ==========================================================

engaged = df[df["score_views"] >= 3]

premium = df[df["premium_user"] == 1]

p_premium_given_engaged = (
    engaged["premium_user"] == 1
).mean()

p_engaged_given_premium = (
    premium["score_views"] >= 3
).mean()

p_engaged = (
    df["score_views"] >= 3
).mean()

bayes_check = (
    p_engaged_given_premium *
    p_premium
) / p_engaged

overall_odds = p_premium / (1 - p_premium)

engaged_odds = (
    p_premium_given_engaged /
    (1 - p_premium_given_engaged)
)

odds_ratio = engaged_odds / overall_odds

# ==========================================================
# PART 3: RANDOM VARIABLE CLASSIFICATION
# ==========================================================

classifications = {

    "days_active": {

        "type": "discrete",

        "support": "Non-negative integers",

        "distribution": "Poisson",

        "justification":
        "Days active is a count measured in whole numbers."

    },

    "score_views": {

        "type": "discrete",

        "support": "Non-negative integers {0,1,2,...}",

        "distribution": "Poisson",

        "justification":
        "Counts how many times a user viewed scores."

    },

    "session_minutes": {

        "type": "continuous",

        "support": "Positive real numbers",

        "distribution": "Normal",

        "justification":
        "Session duration can take decimal values."

    },

    "risk_profile": {

        "type": "categorical",

        "support": "{conservative, moderate, aggressive}",

        "distribution": "Categorical",

        "justification":
        "Represents labels instead of numeric measurements."

    },

    "premium_user": {

        "type": "binary",

        "support": "{0,1}",

        "distribution": "Binomial",

        "justification":
        "Only two possible outcomes exist."

    }

}

# ==========================================================
# VALIDATION
# ==========================================================

assert 0 <= p_premium <= 1

assert 0 <= p_high_engagement <= 1

assert 0 <= p_aggressive <= 1

assert 0 <= p_joint <= 1

assert abs(
    p_premium_given_engaged - bayes_check
) < 0.01

assert odds_ratio > 0

assert all(
    classifications[var]["type"]
    for var in classifications
)
# ==========================================================
# OUTPUT
# ==========================================================

print("=" * 70)
print("PART 1: BASIC PROBABILITIES")
print("=" * 70)

print(f"Sample Space (premium_user): {sample_space_premium}\n")

print(f"P(premium_user = 1): {p_premium:.1%}")
print("  → Interpretation: This is the probability that a randomly selected user is a premium subscriber.\n")

print(f"P(score_views >= 5): {p_high_engagement:.1%}")
print("  → Interpretation: This represents the proportion of users with high engagement.\n")

print(f"P(risk_profile = 'aggressive'): {p_aggressive:.1%}")
print("  → Interpretation: This is the probability that a randomly selected user has an aggressive risk profile.\n")

print(f"P(score_views >= 5 AND premium_user = 1): {p_joint:.1%}")
print("  → Interpretation: This is the probability that a user is both highly engaged and a premium subscriber.")

print("\n" + "=" * 70)
print("PART 2: CONDITIONAL PROBABILITY & BAYES")
print("=" * 70)

print(f"P(premium_user = 1 | score_views >= 3): {p_premium_given_engaged:.1%}")
print("  → Interpretation: Among engaged users, this percentage are premium users.\n")

print(f"P(score_views >= 3 | premium_user = 1): {p_engaged_given_premium:.1%}")
print("  → Interpretation: Among premium users, this percentage are engaged users.\n")

print(f"P(score_views >= 3): {p_engaged:.1%}")

print(f"\nBayes' theorem verification:")
print(f"Left Side : {p_premium_given_engaged:.4f}")
print(f"Right Side: {bayes_check:.4f}")
print("✓ Bayes' theorem verified.\n")

print(f"Odds Ratio: {odds_ratio:.2f}x")

if odds_ratio > 1:
    print("Recommendation: Increasing score views is associated with higher premium conversion. Encourage users to engage with score-related features.")
elif odds_ratio == 1:
    print("Recommendation: Score views do not appear to influence premium conversion.")
else:
    print("Recommendation: Higher score views are associated with lower premium conversion. Investigate user behaviour further.")

print("\n" + "=" * 70)
print("PART 3: RANDOM VARIABLE CLASSIFICATION")
print("=" * 70)

print(f"{'Variable':<20}{'Type':<15}{'Distribution':<15}Support")
print("-" * 90)

for variable, info in classifications.items():
    print(f"{variable:<20}{info['type']:<15}{info['distribution']:<15}{info['support']}")

print("\n" + "=" * 70)
print("JUSTIFICATIONS")
print("=" * 70)

for variable, info in classifications.items():
    print(f"\n{variable}:")
    print(f"  {info['justification']}")

print("\n" + "=" * 70)
print("CRITICAL THINKING")
print("=" * 70)

print(
    "score_views is a discrete random variable because it counts whole-number "
    "events (0, 1, 2, 3, ... views). "
    "session_minutes is a continuous random variable because time can be "
    "measured to any level of precision, such as 10.2 or 15.75 minutes."
)

print("\n" + "=" * 70)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 70)
