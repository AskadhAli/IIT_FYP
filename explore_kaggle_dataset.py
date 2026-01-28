import kagglehub
import pandas as pd
import os
import json

# Download latest version (will use cache if already downloaded)
print("="*60)
print("Loading dataset from Kaggle...")
print("(Will use cache if already downloaded)")
print("="*60)
path = kagglehub.dataset_download("Cornell-University/arxiv")

print(f"\n[OK] Dataset path: {path}")

# List all files we want to load in the dataset
print("\n" + "="*60)
print("Files in the dataset:")
print("="*60)
data_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        rel_path = os.path.relpath(file_path, path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"  {rel_path} ({file_size:.2f} MB)")
        
        # Track data files
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in ['.csv', '.json', '.parquet', '.tsv', '.jsonl']:
            data_files.append((file_path, file, file_ext, file_size))

# Try to find and load data files
print("\n" + "="*60)
print("Attempting to load data files...")
print("="*60)

if not data_files:
    print("[WARNING] No standard data files found. Listing all files for manual inspection.")
else:
    # Sort by size (largest first) - usually the main data file
    data_files.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\nFound {len(data_files)} data file(s). Loading the largest one first...")
    
    for file_path, file_name, file_ext, file_size in data_files:
        print(f"\n[FILE] Processing: {file_name} ({file_size:.2f} MB)")
        print(f"   Type: {file_ext}")
        
        try:
            # For large files, read a sample first
            sample_size = 1000
            
            if file_ext == '.csv':
                print(f"   Reading first {sample_size} rows...")
                df = pd.read_csv(file_path, nrows=sample_size)
                print(f"   [OK] Successfully loaded CSV")
            elif file_ext in ['.json', '.jsonl']:
                print(f"   Reading first {sample_size} rows...")
                try:
                    # Try JSONL (line-delimited JSON) first
                    df = pd.read_json(file_path, lines=True, nrows=sample_size)
                except:
                    # Try regular JSON
                    df = pd.read_json(file_path, nrows=sample_size)
                print(f"   [OK] Successfully loaded JSON")
            elif file_ext == '.parquet':
                print(f"   Reading parquet file...")
                df = pd.read_parquet(file_path)
                # If too large, sample it
                if len(df) > 10000:
                    print(f"   File is large ({len(df)} rows), sampling first 10000 rows...")
                    df = df.head(10000)
                print(f"   [OK] Successfully loaded Parquet")
            elif file_ext == '.tsv':
                print(f"   Reading first {sample_size} rows...")
                df = pd.read_csv(file_path, sep='\t', nrows=sample_size)
                print(f"   [OK] Successfully loaded TSV")
            else:
                continue
            
            print(f"\n   [INFO] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            print(f"\n   " + "="*56)
            print(f"   COLUMN NAMES ({len(df.columns)} columns):")
            print(f"   " + "="*56)
            for i, col in enumerate(df.columns, 1):
                dtype = df[col].dtype
                non_null = df[col].notna().sum()
                pct = (non_null / len(df) * 100) if len(df) > 0 else 0
                print(f"   {i:2d}. {col:40s} | {str(dtype):15s} | {non_null}/{len(df)} ({pct:.1f}%)")
            
            print(f"\n   " + "="*56)
            print(f"   DATA TYPES:")
            print(f"   " + "="*56)
            for col, dtype in df.dtypes.items():
                print(f"      {col}: {dtype}")
            
            print(f"\n   " + "="*56)
            print(f"   SAMPLE DATA (first 3 rows):")
            print(f"   " + "="*56)
            # Show sample with truncated long strings
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_colwidth', 50)
            pd.set_option('display.width', None)
            print(df.head(3).to_string())
            
            # Save column info to JSON file
            col_info = {
                'file_name': file_name,
                'file_path': file_path,
                'file_type': file_ext,
                'file_size_mb': file_size,
                'shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_data': df.head(3).to_dict('records')
            }
            
            output_file = 'arxiv_dataset_columns.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(col_info, f, indent=2, default=str)
            print(f"\n   [OK] Column information saved to '{output_file}'")
            
            # Also save a CSV with just column names for easy reference
            col_df = pd.DataFrame({
                'column_name': df.columns,
                'data_type': [str(dtype) for dtype in df.dtypes],
                'non_null_count': [df[col].notna().sum() for col in df.columns],
                'null_count': [df[col].isna().sum() for col in df.columns]
            })
            col_csv = 'arxiv_dataset_columns.csv'
            col_df.to_csv(col_csv, index=False)
            print(f"   [OK] Column list saved to '{col_csv}'")
            
            print(f"\n   [OK] Successfully analyzed: {file_name}")
            break  # Process the first/largest file
            
        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
            import traceback
            traceback.print_exc()
            continue

print("\n" + "="*60)
print("[OK] Exploration complete!")
print("="*60)
