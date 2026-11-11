import pandas as pd
import os
import glob
import sys
import re

# ==============================================================================
# FOLLOWER DATA (June 2023 - May 2024)
# ==============================================================================
FOLLOWERS_DATA = {
    "Instagram": {
        "tdp": { 
            "2023-06": 131999, "2023-07": 141483, "2023-08": 153679, "2023-09": 173588,
            "2023-10": 190901, "2023-11": 199093, "2023-12": 210172, "2024-01": 223267,
            "2024-02": 235550, "2024-03": 254639, "2024-04": 281234, "2024-05": 301422
        },
        "ysrcp": { 
            "2023-06": 64775, "2023-07": 68765.5, "2023-08": 72676, "2023-09": 76807,
            "2023-10": 79903.5, "2023-11": 83841.5, "2023-12": 101540, "2024-01": 129324,
            "2024-02": 155881.5, "2024-03": 182969, "2024-04": 210794, "2024-05": 228984
        },
        "ncbn": { 
            "2023-06": 460628.5, "2023-07": 469762.5, "2023-08": 486692.5, "2023-09": 513916,
            "2023-10": 533596, "2023-11": 540164.5, "2023-12": 545118.5, "2024-01": 548887,
            "2024-02": 554731, "2024-03": 562935, "2024-04": 570324.5, "2024-05": 577395
        },
        "ysjagan": { 
            "2023-06": 696814, "2023-07": 709540.5, "2023-08": 718886, "2023-09": 730507,
            "2023-10": 744848.5, "2023-11": 755655.5, "2023-12": 772872, "2024-01": 798788,
            "2024-02": 831102, "2024-03": 866092, "2024-04": 908744.5, "2024-05": 947539
        }
    },
    "Facebook": {
        "ysjagan": { 
            "2023-06": 2176552, "2023-07": 2178837, "2023-08": 2181334, "2023-09": 2195392,
            "2023-10": 2198517, "2023-11": 2199118, "2023-12": 2200989, "2024-01": 2202764,
            "2024-02": 2204743, "2024-03": 2206643, "2024-04": 2219042, "2024-05": 2222917
        },
        "ysrcp": { 
            "2023-06": 1245531, "2023-07": 1258772, "2023-08": 1269518, "2023-09": 1278300,
            "2023-10": 1283070, "2023-11": 1285061, "2023-12": 1287225, "2024-01": 1289489,
            "2024-02": 1292196, "2024-03": 1294620, "2024-04": 1296458, "2024-05": 1298249
        },
        "ncbn": { 
            "2023-06": 1863366, "2023-07": 1864288, "2023-08": 1867234, "2023-09": 1871397,
            "2023-10": 1872958, "2023-11": 1872958, "2023-12": 1873484, "2024-01": 1874312,
            "2024-02": 1875143, "2024-03": 1876185, "2024-04": 1876454, "2024-05": 1876906
        },
        "tdp": { 
            "2023-06": 4142228, "2023-07": 4149559, "2023-08": 4159198, "2023-09": 4171433,
            "2023-10": 4175999, "2023-11": 4176419, "2023-12": 4177560, "2024-01": 4179130,
            "2024-02": 4180026, "2024-03": 4181804, "2024-04": 4181371, "2024-05": 4180581
        }
    }
}

ACCOUNT_MAP = {
    "ysjagan": "ysjagan", "jagan": "ysjagan", "jmr": "ysjagan",
    "ncbn": "ncbn", "chandrababu": "ncbn", "cbn": "ncbn",
    "tdp": "tdp", "telugudesam": "tdp",
    "ysrcp": "ysrcp", "ysrcongress": "ysrcp"
}

def identify_file_context(filename):
    fname_lower = filename.lower()
    platform = None
    if "instagram" in fname_lower: platform = "Instagram"
    elif "facebook" in fname_lower: platform = "Facebook"
    
    account = None
    for alias, key in ACCOUNT_MAP.items():
        if alias in fname_lower:
            account = key
            break
    return account, platform

def main():
    # Detect current folder
    if getattr(sys, 'frozen', False):
        folder_path = os.path.dirname(sys.executable)
    elif __file__:
        folder_path = os.path.dirname(os.path.abspath(__file__))
    else:
        folder_path = os.getcwd()

    print(f"Running in folder: {folder_path}")

    files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    count = 0
    for file in files:
        filename = os.path.basename(file)
        
        # Skip system files or Thematic file
        if filename.startswith("~$") or "Hastags Thematic" in filename:
            continue
            
        print(f"\nProcessing: {filename}...")
        
        try:
            account, platform = identify_file_context(filename)
            
            if not account or not platform:
                print(f"  -> Skipped: Cannot identify Account/Platform from name.")
                continue

            # Load Data
            df = pd.read_excel(file)
            
            # --- TASK 1: RENAME OLD ENGAGEMENT COLUMN ---
            if 'engagement_rate' in df.columns:
                df.rename(columns={'engagement_rate': 'Engagement_Rate_Timeline_Avg'}, inplace=True)
            
            # --- TASK 2: CALCULATE NEW MONTHLY ENGAGEMENT ---
            if 'date' in df.columns:
                def get_monthly_rate(row):
                    try:
                        post_date = pd.to_datetime(row['date'])
                        month_key = post_date.strftime('%Y-%m')
                        
                        likes = row.get('likes', 0)
                        shares = row.get('shares', 0)
                        comments = row.get('comments_count', 0)
                        
                        interactions = likes + shares + comments
                        
                        followers = FOLLOWERS_DATA.get(platform, {}).get(account, {}).get(month_key)
                        
                        if followers and followers > 0:
                            return interactions / followers
                        else:
                            return None 
                    except:
                        return None

                df['Engagement_Rate_Monthly'] = df.apply(get_monthly_rate, axis=1)
            
            # --- TASK 3: CLEAN FILENAME ---
            timeline = "JanMay2024" # Default
            timeline_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z]*202\d', filename, re.IGNORECASE)
            if timeline_match:
                timeline = timeline_match.group(0)
            
            # Construct standard name
            new_name = f"{account}_{platform}_{timeline}_HashtagClassified.xlsx"
            new_path = os.path.join(folder_path, new_name)
            
            # --- TASK 4: OVERWRITE / REPLACE ---
            # Save to the new path (preserves unicode)
            df.to_excel(new_path, index=False)
            
            # If the name changed, delete the old file to simulate "overwriting" the entry in the folder
            # If name didn't change, we just overwrote it above.
            if new_path.lower() != file.lower():
                os.remove(file)
                print(f"  -> Renamed & Overwritten: {new_name}")
            else:
                print(f"  -> Overwritten (in place): {new_name}")
                
            count += 1

        except Exception as e:
            print(f"  -> Error: {e}")

    print(f"\nAll done! Processed {count} files.")
    input("Press Enter to close...")

if __name__ == "__main__":
    main()