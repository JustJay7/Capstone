import json
p = "/Users/jay/Desktop/Projects/Capstone/configs/intent_taxonomy.json"
data = json.load(open(p, "r", encoding="utf-8"))

for intent in data["intents"]:
    if "routing_policy" not in intent:
        intent["routing_policy"] = {"boost_rules": []}

json.dump(data, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("✅ Patched taxonomy — every intent now has routing_policy.")
