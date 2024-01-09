import pandas as pd
from tqdm import tqdm
import random
import json
import os

# File and folder paths
csv_file_path = './philosophy_data.csv'
output_folder = './anti_oedipus_finetune_240109'

# Rows details
start_row = 212552
end_row = 219230

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def create_random_jsonl_files(csv_path, start, end, output_dir):
    # Read the specified range of rows from the CSV file
    data = pd.read_csv(csv_path, skiprows=start-1, nrows=end-start+1, header=None, usecols=[4])

    # Process rows to create JSONL files
    for i in tqdm(range(start, end)):
        for _ in range(5):  # Repeat 5 times for each row
            j = random.randint(0, 20)  # Random value between 0 and 20
            end_index = min(i + j, end)  # Ensure not to exceed the end row
            text_block = ' '.join(data.iloc[i-start:end_index-start, 0].astype(str))

            # Randomly split the text block into 'input' and 'output'
            split_index = random.randint(int(len(text_block) * 0.2), len(text_block))
            input_text = text_block[:split_index]
            output_text = text_block[split_index:]

            # Create a dictionary for JSONL format
            jsonl_content = {"instruction": input_text, "output": output_text}

            # Create a unique filename
            file_name = f'{output_dir}/file_{i}_{_}.jsonl'

            # Write the dictionary to a JSONL file
            with open(file_name, 'w') as outfile:
                json_line = json.dumps(jsonl_content)
                outfile.write(json_line + '\n')

# Call the function
create_random_jsonl_files(csv_file_path, start_row, end_row, output_folder)