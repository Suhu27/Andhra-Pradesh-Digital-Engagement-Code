import pandas as pd
import glob

# Get all xlsx files in the current directory
xlsx_files = glob.glob("*.xlsx")

print(f"Found {len(xlsx_files)} Excel files to process\n")
print("="*80)

# Summary statistics
total_files_processed = 0
total_non_political_dropped = 0
total_rows_kept_and_relabeled = 0

# Process each file
for file in xlsx_files:
    try:
        print(f"\nProcessing: {file}")
        
        # Read the Excel file
        df = pd.read_excel(file, engine='openpyxl')
        
        original_row_count = len(df)
        
        # Check if required columns exist
        if 'Political_Label' not in df.columns:
            print(f"  ⚠ 'Political_Label' column not found, skipping file")
            continue
            
        if 'Hashtag_Theme' not in df.columns:
            print(f"  ⚠ 'Hashtag_Theme' column not found, skipping file")
            continue
        
        # Identify Non-Political posts WITH Hashtag_Theme (these will be kept and relabeled)
        non_political_with_themes_mask = (
            (df['Political_Label'].str.strip() == 'Non-Political') &
            (df['Hashtag_Theme'].notna()) & 
            (df['Hashtag_Theme'].astype(str).str.strip() != '')
        )
        
        # Count rows to be relabeled
        rows_to_relabel = non_political_with_themes_mask.sum()
        
        # Change Political_Label to "Political" for Non-Political rows with Hashtag_Theme
        df.loc[non_political_with_themes_mask, 'Political_Label'] = 'Political'
        
        # Now drop ALL remaining Non-Political rows
        df_filtered = df[df['Political_Label'].str.strip() != 'Non-Political'].copy()
        
        rows_dropped = original_row_count - len(df_filtered)
        
        # Save the modified dataframe back to the same file
        df_filtered.to_excel(file, index=False, engine='openpyxl')
        
        # Update statistics
        total_files_processed += 1
        total_non_political_dropped += (rows_dropped - rows_to_relabel)
        total_rows_kept_and_relabeled += rows_to_relabel
        
        print(f"  ✓ Original rows: {original_row_count}")
        print(f"  ✓ Rows kept & relabeled to 'Political': {rows_to_relabel}")
        print(f"  ✓ Non-Political rows dropped: {rows_dropped - rows_to_relabel}")
        print(f"  ✓ Final rows: {len(df_filtered)}")
        print(f"  ✓ Saved: {file}")
        
    except Exception as e:
        print(f"  ✗ Error processing {file}: {str(e)}")

# Summary Report
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)
print(f"Total files processed: {total_files_processed}")
print(f"Total Non-Political rows with Hashtag_Theme (kept & relabeled): {total_rows_kept_and_relabeled}")
print(f"Total Non-Political rows dropped: {total_non_political_dropped}")
print("="*80)
print("All datasets have been updated successfully!")
print("="*80)
