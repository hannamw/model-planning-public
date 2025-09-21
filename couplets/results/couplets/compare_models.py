#%%
#!/usr/bin/env python3

import pandas as pd
import os

def load_all_csvs():
    """Load all 6 CSV files for the different Qwen3 models."""
    models = ['0.6B', '1.7B', '4B', '8B', '14B', '32B']
    dataframes = {}
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for model in models:
        file_path = os.path.join(base_dir, f'Qwen3-{model}.csv')
        df = pd.read_csv(file_path)
        dataframes[f'Qwen3-{model}'] = df
    
    return dataframes

def print_row_completions(dataframes, row_index):
    """Print the first line and each model's clean completion for a given row index."""
    models = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B']
    
    if row_index >= len(dataframes[models[0]]):
        print(f"Row index {row_index} is out of range.")
        return
    
    # Get the first line from any model (they should all be the same)
    first_line = dataframes[models[0]].iloc[row_index]['first_line']
    print(row_index)
    print(f"First line: {first_line}")
    print()
    
    # Print each model's completion
    for model in models:
        clean_completion = dataframes[model].iloc[row_index]['clean_completion']
        print(f"{model}: {clean_completion}")
    
    print("-" * 80)

def print_row_completions_latex(dataframes, row_index):
    """Print LaTeX formatted output with minipage and outline for a given row index."""
    models = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B']
    
    if row_index >= len(dataframes[models[0]]):
        print(f"% Row index {row_index} is out of range.")
        return
    
    # Get the first line from any model (they should all be the same)
    first_line = dataframes[models[0]].iloc[row_index]['first_line']
    
    # Escape LaTeX special characters
    def escape_latex(text):
        text = str(text)
        replacements = {
            '\\': '\\textbackslash{}',
            '{': '\\{',
            '}': '\\}',
            '$': '\\$',
            '&': '\\&',
            '%': '\\%',
            '#': '\\#',
            '^': '\\textasciicircum{}',
            '_': '\\_',
            '~': '\\textasciitilde{}'
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text
    
    print("\\fbox{%")
    print("\\parbox{\\dimexpr\\textwidth-2\\fboxsep-2\\fboxrule\\relax}{%")
    print(f"\\textbf{{First line:}} {escape_latex(first_line)}\\")
    print()
    
    # Print each model's completion
    for model in models:
        clean_completion = dataframes[model].iloc[row_index]['clean_completion']
        print(f"\\textbf{{{model}:}} {escape_latex(clean_completion)}\\")
    
    print("}")
    print("}")
    print("\\vspace{0.5em}")
    print()

#%%
"""Main function to load CSVs and call functions for each row."""
print("Loading CSV files...")
dataframes = load_all_csvs()

# Get the number of rows (should be the same for all models)
num_rows = len(dataframes['Qwen3-32B'])
print(f"Found {num_rows} rows in each CSV file.")
print("=" * 80)
print()

print("PLAIN TEXT OUTPUT:")
print("=" * 80)

# Call print function for each row
for row_idx in range(num_rows):
    print_row_completions(dataframes, row_idx)



# Call LaTeX function for each row
#for row_idx in range(num_rows):
#    print_row_completions_latex(dataframes, row_idx)

# %%
print_row_completions_latex(dataframes, 237)
# %%
print_row_completions_latex(dataframes, 195)
#%%
print_row_completions_latex(dataframes, 181)
#%%
print_row_completions_latex(dataframes, 513)
# %%
