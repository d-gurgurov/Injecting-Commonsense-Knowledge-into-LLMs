from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

language = "ms"

# loading mt (20231101.mt), bg (20231101.bg), and ms (20231101.ms) wikipedia
dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}")

# extracting text and title columns
df = pd.DataFrame({
    'text': dataset['train']['text']
})

# splitting into training and validation sets (90:10)
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)

# saving to CSV with only 'text' column
train_df[['text']].to_csv(f'train_wiki_{language}.csv', index=False)
valid_df[['text']].to_csv(f'val_wiki_{language}.csv', index=False)