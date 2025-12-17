import pandas as pd
import numpy as np
import re

def preprocess_clothing_data(file_path):
    # === Load the CSV file ===
    df = pd.read_csv(file_path)
    
    # 1. Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # 2. Fix known typos in Category
    df['Category'] = df['Category'].str.strip()
    df['Category'] = df['Category'].str.replace('MEN_Costs', 'MEN_Coats', regex=False)

    # ✅ Keep MEN_ and WOMEN_ categories only
    df = df[df['Category'].str.contains("MEN_|WOMEN_", na=False)]

    # 3. Extract measurements into separate columns
    def extract_measurements(text):
        if pd.isna(text):
            return np.nan, np.nan, np.nan, np.nan
        text = str(text).replace('Col 6: Measurements :', '').strip()
        
        measurements = {}
        for part in re.findall(r'(\w+):\s*([\d.]+)', text):
            key, value = part
            measurements[key.lower()] = float(value)
        
        return (
            measurements.get('chest', np.nan),
            measurements.get('waist', np.nan),
            measurements.get('hips', np.nan),
            measurements.get('shoulders', np.nan)
        )

    df[['Chest_cm', 'Waist_cm', 'Hips_cm', 'Shoulders_cm']] = pd.DataFrame(
        df['Measurements'].apply(extract_measurements).tolist(),
        index=df.index
    )

    # 4. Clean text columns (standardize casing & spacing)
    for col in ['Body_Type', 'Skin_Tone', 'Gender', 'Suitable_Styles', 'Preferred_Colors', 'Clothing_Type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # 5. Fix truncated Clothing_Type values
    clothing_type_mapping = {
        'Jea': 'Jeans',
        'Je': 'Jeans',
        'Hoo': 'Hoodie',
        'Dr': 'Dress',
        'Ja': 'Jacket'
    }
    df['Clothing_Type'] = df['Clothing_Type'].replace(clothing_type_mapping)

    # 6. Handle missing values
    df['Preferred_Colors'] = df['Preferred_Colors'].fillna('Not Specified')
    df['Clothing_Type'] = df['Clothing_Type'].fillna('Not Specified')

    # 7. Drop the original Measurements column
    df = df.drop(columns=['Measurements'])

    # 8. Reorder columns
    columns_order = [
        'Image', 'Category', 'Body_Type', 'Skin_Tone', 'Gender',
        'Chest_cm', 'Waist_cm', 'Hips_cm', 'Shoulders_cm',
        'Suitable_Styles', 'Preferred_Colors', 'Clothing_Type'
    ]
    df = df[columns_order]

    return df


# === Run Preprocessing ===
input_file = 'Data/unique_clothing_dataset.csv'
output_file = 'Preprocess_outputs/Preprocess_clothing_dataset.csv'

preprocessed_df = preprocess_clothing_data(input_file)
preprocessed_df.to_csv(output_file, index=False)

print(f" Data preprocessing complete. Cleaned data saved to {output_file}")
print("Categories available:", preprocessed_df['Category'].unique())
print(f" Total rows: {len(preprocessed_df)}")
