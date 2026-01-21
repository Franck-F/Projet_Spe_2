from wilds import get_dataset
import pandas as pd

try:
    print("Loading dataset...")
    dataset = get_dataset(dataset="camelyon17", download=False, root_dir='data/raw/wilds')
    with open('debug_output.txt', 'w') as f:
        f.write(f"Dataset type: {type(dataset)}\n")
        
        if hasattr(dataset, 'metadata_df'):
            f.write("✅ metadata_df exists!\n")
        else:
            f.write("❌ metadata_df DOES NOT exist!\n")
            f.write(f"Has _metadata_df? {hasattr(dataset, '_metadata_df')}\n")
            f.write("Available attributes:\n")
            f.write(str(dir(dataset)))

        
except Exception as e:
    print(f"Error: {e}")
