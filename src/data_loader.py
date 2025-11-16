import pandas as pd

def load_data(path):
    """
    Loads the SMS Spam Collection dataset (tab-separated),
    cleans columns, encodes labels and returns a DataFrame
    with columns: ['label', 'message'] where label is 0 (ham) or 1 (spam).
    """
    try:
        df = pd.read_csv(
            path,
            sep='\t',             # <-- important: use tab as separator
            encoding='latin-1',   # handles special characters like £, ü, etc.
            names=['label', 'message'],
            on_bad_lines='skip'   # skips any malformed rows if present
        )
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

    # Encode labels: ham → 0, spam → 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Drop any missing rows
    df = df.dropna()

    return df
