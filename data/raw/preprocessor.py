import pandas as pd
from datasets import load_dataset

def get_tone(text):
    # Heuristic for Tone: Formal vs Casual
    if any(word in text.lower() for word in ["shall", "honored", "sincerely", "regarding"]):
        return "formal"
    return "casual"

def download_and_format_dataset():
    print("Fetching SemEval-2018 Task 1...")
    # Load the English subtask for emotion classification
    dataset = load_dataset("semeval_2018_task_1", "subtask5.english", split='train')
    df_raw = pd.DataFrame(dataset)

    # 1. Map intensity (SemEval uses 0/1 for classification, 
    # but the 'EI-reg' subset provides floats. Let's simulate for this structure.)
    # For a truer 0-1 intensity, we'll use a placeholder or pull from the EI-reg split.
    
    formatted_data = []

    for _, row in df_raw.head(50).iterrows():
        text = row['Tweet']
        
        # Determine primary emotion (the one with label 1)
        emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
        active_emotions = [e for e in emotions if row[e] == 1]
        
        primary_emotion = active_emotions[0] if active_emotions else "neutral"
        
        # Map Polarity
        positive_emotions = ['joy', 'love', 'optimism', 'trust', 'surprise']
        polarity = "positive" if primary_emotion in positive_emotions else "negative"
        if primary_emotion == "neutral": polarity = "neutral"

        formatted_data.append({
            "text": text,
            "polarity": polarity,
            "emotion": primary_emotion,
            "tone": get_tone(text),
            "intensity": round(0.5 + (len(text) % 50) / 100, 2) # Simulated intensity float
        })

    # Create Final DataFrame
    df_final = pd.DataFrame(formatted_data)
    
    # Save to CSV
    output_file = "resembled_dataset.csv"
    df_final.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(df_final.head())

if __name__ == "__main__":
    download_and_format_dataset()