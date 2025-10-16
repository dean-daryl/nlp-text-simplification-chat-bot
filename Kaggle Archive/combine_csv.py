import os
import glob
import numpy as np
import pandas as pd

# Use current directory instead of hardcoded Windows path
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Working directory: {current_dir}")

# Change to the current directory
os.chdir(current_dir)

# Find CSV files (exclude the output file to avoid circular processing)
extension = 'csv'
files = glob.glob('*.{}'.format(extension))
files = [f for f in files if f != 'all_data.csv']  # Exclude output file

print(f"Found {len(files)} CSV files to process")

Elementary = list()
Intermediate = list()
Advanced = list()

for i, file in enumerate(files):
    try:
        print(f"Processing file {i+1}/{len(files)}: {file}")
        # Try different encodings to handle various file formats
        try:
            df = pd.read_csv(file, encoding='cp1252')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding='latin-1')
        
        if df.shape[0] > 0 and df.shape[1] >= 3:  # Ensure we have at least 3 columns
            Elementary.append(df.iloc[:,0].to_list())
            Intermediate.append(df.iloc[:,1].to_list())
            Advanced.append(df.iloc[:,2].to_list())
            print(f"   ✅ Processed: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"   ⚠️ Skipped: insufficient data ({df.shape})")
    except Exception as e:
        print(f"   ❌ Error processing {file}: {e}")

# Create the combined dataframe
df_combined = pd.DataFrame({
    'Elementary': Elementary,
    'Intermediate': Intermediate, 
    'Advanced': Advanced
})

# Save the combined data
output_file = 'all_data.csv'
df_combined.to_csv(output_file, index=False)

print(f"\n🎉 Success!")
print(f"📊 Combined data shape: {df_combined.shape}")
print(f"💾 Saved to: {output_file}")
print(f"📈 Total files processed: {len(Elementary)}")
