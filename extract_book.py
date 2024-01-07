# Extract single book from phil_nlp dataset. Anti-Oedipus for example.
import pandas as pd
import json
from tqdm import tqdm

# Replace with the actual file paths and details
csv_file_path = './philosophy_data.csv'
jsonl_file_path = './all.jsonl'
start_row = '2'  
end_row = '396428'
column = 'E' # 5th column, aka sentece_str

# Convert column letter to zero-based index (if necessary)
column_index = ord(column.upper()) - 65 # Actual index is 4

def create_jsonl_from_csv(csv_path, jsonl_path, start, end, column_idx):
    data = pd.read_csv(csv_path, skiprows=start-1, nrows=end-start+1, usecols=[column_idx])
    
    concatenated_sentences = ' '.join(data.iloc[:,0].astype(str))
    
    jsonl_content = {"text": concatenated_sentences}
    
    # Write the dictionary to a JSONL file
    with open(jsonl_path, 'w') as outfile:
        for entry in tqdm([jsonl_content]):
            json_line = json.dumps(entry)
            outfile.write(json_line + '\n')

# Call the function with the specified parameters
create_jsonl_from_csv(csv_file_path, jsonl_file_path, int(start_row), int(end_row), column_index)