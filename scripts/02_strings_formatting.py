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


"""


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("Strings and Formatting")
    print("This lesson will introduce string basics and clean formatting.")


if __name__ == "__main__":
    main()


# ============================================================
# Add string basics: create, index, slice
# ============================================================

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


# ============================================================
# Add common string methods + cleaning patterns
# ============================================================

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


# ============================================================
# Add f-strings, number formatting, and parsing
# ============================================================

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

    section("4) f-strings and formatting (research output)")

    gdp = 24567.89123
    inflation = 0.03784
    year = 2024

    print(f"GDP per capita in {year}: ${gdp:,.2f}")
    print(f"Inflation rate in {year}: {inflation:.2%}")

    section("5) Simple parsing pattern: split")

    record = "state=Kentucky, year=2023, income=42000"
    parts = [p.strip() for p in record.split(",")]

    print(f"raw record: {record}")
    print(f"split parts: {parts}")

    kv = {}
    for part in parts:
        key, value = [x.strip() for x in part.split("=")]
        kv[key] = value

    print(f"parsed dict: {kv}")

    section("6) Mini exercise: clean and standardize a column name")

    raw_col = "  Median Household Income ($)  "
    standardized = raw_col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("$", "")
    print(f"raw: '{raw_col}'")
    print(f"standardized: '{standardized}'")

    section("Done")
    print("Next lesson: numbers, math operations, and working with randomness.")


if __name__ == "__main__":
    main()
