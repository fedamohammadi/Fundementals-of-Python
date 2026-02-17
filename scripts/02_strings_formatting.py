"""
02_strings_formatting.py

Purpose:
--------
Learn how to work with text data in Python.
Strings are everywhere in research: names, categories, IDs, survey responses,
dates read as text, messy labels, and scraped web text.

Concepts covered (progressively):
- Creating strings
- Indexing and slicing
- Common string methods
- f-strings and formatting
- Cleaning text: strip, lower, replace
- Simple parsing patterns used in data analysis

Run:
----
python scripts/02_strings_formatting.py
"""


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("Strings and Formatting")
    print("This lesson will introduce string basics and clean formatting.")


def main() -> None:
    section("1) Creating strings")

    name = "Feda"
    country = "Afghanistan"
    sentence = "Python is useful for research."

    print(f"name = {name}")
    print(f"country = {country}")
    print(f"sentence = {sentence}")

    section("2) Indexing and slicing")

    text = "econometrics"
    print(f"text = {text}")
    print(f"text[0] = {text[0]}")
    print(f"text[1] = {text[1]}")
    print(f"text[-1] = {text[-1]}")

    print(f"text[0:4] = {text[0:4]}")
    print(f"text[:4] = {text[:4]}")
    print(f"text[4:] = {text[4:]}")
    print(f"text[-4:] = {text[-4:]}")


def main() -> None:
    section("1) Creating strings")

    name = "Feda"
    country = "Afghanistan"
    sentence = "Python is useful for research."

    print(f"name = {name}")
    print(f"country = {country}")
    print(f"sentence = {sentence}")

    section("2) Indexing and slicing")

    text = "econometrics"
    print(f"text = {text}")
    print(f"text[0] = {text[0]}")
    print(f"text[-1] = {text[-1]}")
    print(f"text[:4] = {text[:4]}")
    print(f"text[4:] = {text[4:]}")

    section("3) Common string methods")

    messy = "   GDP per Capita (USD)   "
    print(f"raw = '{messy}'")
    print(f"strip = '{messy.strip()}'")
    print(f"lower = '{messy.strip().lower()}'")
    print(f"upper = '{messy.strip().upper()}'")

    label = "Real_GDP_Growth_Rate"
    print(f"\nlabel = {label}")
    print(f"replace '_' -> ' ' : {label.replace('_', ' ')}")
    print(f"starts with 'Real'? {label.startswith('Real')}")
    print(f"contains 'GDP'? {'GDP' in label}")

    section("4) Cleaning a survey-style response")

    response = "  Yes, I strongly agree.  "
    cleaned = response.strip().lower()
    print(f"raw response: '{response}'")
    print(f"cleaned response: '{cleaned}'")







if __name__ == "__main__":
    main()
