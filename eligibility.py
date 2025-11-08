import json
import os

def check_eligibility(scheme_name, user_data):
    rules_path = "rules/eligibility.json"
    if not os.path.exists(rules_path):
        return "❌ No eligibility rules file found."

    with open(rules_path, "r") as f:
        rules = json.load(f)

    matched_scheme = None
    for key in rules.keys():
        if key.lower() == scheme_name.lower():
            matched_scheme = key
            break

    if not matched_scheme:
        return f"❌ No rules found for {scheme_name}"

    scheme_rules = rules[matched_scheme]
    income_limits = scheme_rules.get("income_limit", {})
    subsidy_amounts = scheme_rules.get("subsidy_amount", {})

    errors = []

    # ✅ Age validation
    if user_data["age"] < scheme_rules["age_limit"]:
        errors.append(f"Minimum age required is {scheme_rules['age_limit']}")

    # ✅ Income validation (check eligible category)
    eligible_category = None
    for category, limit in income_limits.items():
        if user_data["income"] <= limit:
            eligible_category = category
            break

    if not eligible_category:
        highest = max(income_limits.values())
        errors.append(f"Income exceeds maximum eligible limit of ₹{highest:,}")

    # ✅ State validation
    if "states" in scheme_rules and user_data.get("state"):
        if user_data["state"] not in scheme_rules["states"]:
            errors.append(f"Scheme not available in {user_data['state']}")

    # ✅ Final eligibility message
    if errors:
        return "❌ Not eligible:\n- " + "\n- ".join(errors)
    else:
        subsidy = subsidy_amounts.get(eligible_category, 0)
        return (
            f"✅ Eligible for {matched_scheme} ✅\n"
            f"🏷️ Category: {eligible_category}\n"
            f"💰 Approximate Subsidy: ₹{subsidy:,}"
        )

if __name__ == "__main__":
    scheme_name = input("Enter scheme name: ").strip()
    age = int(input("Enter your age: "))
    income = int(input("Enter your annual income (in ₹): "))
    state = input("Enter your state: ").strip()

    user_data = {"age": age, "income": income, "state": state}
    print(check_eligibility(scheme_name, user_data))