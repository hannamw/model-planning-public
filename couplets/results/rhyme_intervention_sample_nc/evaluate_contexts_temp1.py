#%%
import re
from functools import lru_cache

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from transformers import AutoTokenizer
from nltk.corpus import cmudict

phonemizer = cmudict.dict()

# Extract just vowels (phonemes ending in 0, 1, or 2)
def get_last_vowel(word):
    if word.lower() in phonemizer:
        phonemes = phonemizer[word.lower()][0]  # First pronunciation
        vowels = [p for p in phonemes if p[-1] in '012']
        return vowels[-1] if vowels else None
    return None

def get_final_consonant(word, depluralize=False):
    if word.lower() in phonemizer:
        phonemes = phonemizer[word.lower()][0]  # First pronunciation
        if not phonemes:
            return None
        if depluralize and phonemes[-1] == 'Z':
            phonemes = phonemes[:-1]
        if not phonemes:
            return None
        last_phoneme = phonemes[-1]
        return last_phoneme if last_phoneme[-1] not in '012' else None
    return None

models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

# new column headers
# ,first_line,first_last_word,second_last_word,rhyme_group,rhyme_success,clean_completion,raw_completion,raw_completion_without_last,feature_count,found_valid_row,chosen_id,chosen_index,chosen_rhyme_group,chosen_feature_count,original_generation,intervention_generation,temp_0.3_sample_1,temp_0.3_sample_2,temp_0.3_sample_3,temp_0.3_sample_4,temp_0.3_sample_5,temp_0.7_sample_1,temp_0.7_sample_2,temp_0.7_sample_3,temp_0.7_sample_4,temp_0.7_sample_5,temp_1.0_sample_1,temp_1.0_sample_2,temp_1.0_sample_3,temp_1.0_sample_4,temp_1.0_sample_5

# Keep a list of all model dataframes with metrics
all_model_dfs = []
for model in models:
    tokenizer = AutoTokenizer.from_pretrained('Qwen/' + model)
    print(model)
    df = pd.read_csv(model + '.csv')
    
    # Add columns for correctness metrics and vowel/consonant detection
    df['model'] = model
    df['tokens_same'] = None
    df['orig_len'] = None
    df['intervened_len'] = None
    
    # Add columns for phonetic analysis
    df['vowel_correct'] = False
    df['consonant_correct'] = False
    df['singular_consonant_correct'] = False
    df['detected_vowel'] = None
    df['detected_consonant'] = None
    df['target_vowel'] = None
    df['target_consonant'] = None
    
    # Add columns for temperature sampling prefix overlaps
    df['temp_0.3_tokens_same'] = None
    df['temp_0.7_tokens_same'] = None
    df['temp_1.0_tokens_same'] = None
    
    # Add columns for intervention temperature sampling prefix overlaps
    df['intervention_temp_1.0_tokens_same'] = None
    
    # Add columns for temperature sampling phonetic analysis
    df['temp_0.3_vowel_correct'] = None
    df['temp_0.7_vowel_correct'] = None
    df['temp_1.0_vowel_correct'] = None
    df['temp_0.3_consonant_correct'] = None
    df['temp_0.7_consonant_correct'] = None
    df['temp_1.0_consonant_correct'] = None
    df['temp_0.3_singular_consonant_correct'] = None
    df['temp_0.7_singular_consonant_correct'] = None
    df['temp_1.0_singular_consonant_correct'] = None
    
    # Add columns for intervention temperature sampling phonetic analysis
    df['intervention_temp_1.0_vowel_correct'] = None
    df['intervention_temp_1.0_consonant_correct'] = None
    df['intervention_temp_1.0_singular_consonant_correct'] = None
    
    for idx, row in df.iterrows():
        original_generation = re.sub('.*</think>\n\n', '', row['original_generation'], flags=re.DOTALL).strip('.')
        intervention_generation = re.sub('.*</think>\n\n', '', row['intervention_generation'], flags=re.DOTALL).strip('.')
        
        original_tokens = tokenizer(original_generation, return_tensors='pt').input_ids.squeeze(0)
        intervention_tokens = tokenizer(intervention_generation, return_tensors='pt').input_ids.squeeze(0)
        min_len = min(original_tokens.size(0), intervention_tokens.size(0))
        matches = (original_tokens[:min_len] == intervention_tokens[:min_len])
        match_len = (matches.cumprod(dim=0)).sum().item()
        
        df.loc[idx, 'tokens_same'] = match_len
        df.loc[idx, 'orig_len'] = original_tokens.size(0)
        df.loc[idx, 'intervened_len'] = intervention_tokens.size(0)
        
        # Phonetic analysis for intervention generation
        original_rhyme_group = row['rhyme_group']
        target_rhyme_group = row['chosen_rhyme_group']
        intervention_last_word = intervention_generation.split()[-1]
        intervention_last_word = re.sub(r'^[^\w]+|[^\w]+$', '', intervention_last_word.strip()).lower()
        
        # Get and compare vowels
        original_rhyme_vowel = get_last_vowel(original_rhyme_group)
        rhyme_vowel = get_last_vowel(target_rhyme_group)
        intervention_vowel = get_last_vowel(intervention_last_word)
        df.loc[idx, 'target_vowel'] = rhyme_vowel
        df.loc[idx, 'detected_vowel'] = intervention_vowel
        df.loc[idx, 'vowel_correct'] = rhyme_vowel == intervention_vowel

        # Get and compare consonants
        original_rhyme_consonant = get_final_consonant(original_rhyme_group)
        rhyme_consonant = get_final_consonant(target_rhyme_group)
        intervention_consonant = get_final_consonant(intervention_last_word)
        df.loc[idx, 'target_consonant'] = rhyme_consonant
        df.loc[idx, 'detected_consonant'] = intervention_consonant
        df.loc[idx, 'consonant_correct'] = rhyme_consonant == intervention_consonant

        rhyme_consonant_singular = get_final_consonant(target_rhyme_group, depluralize=True)
        intervention_consonant_singular = get_final_consonant(intervention_last_word, depluralize=True)
        df.loc[idx, 'singular_consonant_correct'] = rhyme_consonant_singular == intervention_consonant_singular
        
        # Calculate temperature sampling prefix overlaps and phonetic analysis
        for temp in ['0.3', '0.7', '1.0']:
            temp_samples = []
            temp_vowel_correct = []
            temp_consonant_correct = []
            temp_singular_consonant_correct = []
            
            for sample_num in range(1, 6):  # samples 1-5
                col_name = f'temp_{temp}_sample_{sample_num}'
                if col_name in row and pd.notna(row[col_name]):
                    temp_generation = re.sub('.*</think>\n\n', '', str(row[col_name]), flags=re.DOTALL).strip('.')
                    temp_tokens = tokenizer(temp_generation, return_tensors='pt').input_ids.squeeze(0)
                    temp_min_len = min(original_tokens.size(0), temp_tokens.size(0))
                    temp_matches = (original_tokens[:temp_min_len] == temp_tokens[:temp_min_len])
                    temp_match_len = (temp_matches.cumprod(dim=0)).sum().item()
                    temp_samples.append(temp_match_len)
                    
                    # Phonetic analysis for temperature sampling
                    temp_last_word = temp_generation.split()[-1]
                    temp_last_word = re.sub(r'^[^\w]+|[^\w]+$', '', temp_last_word.strip()).lower()
                    
                    temp_vowel = get_last_vowel(temp_last_word)
                    temp_consonant = get_final_consonant(temp_last_word)
                    temp_consonant_singular = get_final_consonant(temp_last_word, depluralize=True)
                    
                    temp_vowel_correct.append(original_rhyme_vowel == temp_vowel)
                    temp_consonant_correct.append(original_rhyme_consonant == temp_consonant)
                    temp_singular_consonant_correct.append(rhyme_consonant_singular == temp_consonant_singular)
            
            # Take mean of all temperature samples for this temperature
            if temp_samples:
                df.loc[idx, f'temp_{temp}_tokens_same'] = sum(temp_samples) / len(temp_samples)
                df.loc[idx, f'temp_{temp}_vowel_correct'] = sum(temp_vowel_correct) / len(temp_vowel_correct)
                df.loc[idx, f'temp_{temp}_consonant_correct'] = sum(temp_consonant_correct) / len(temp_consonant_correct)
                df.loc[idx, f'temp_{temp}_singular_consonant_correct'] = sum(temp_singular_consonant_correct) / len(temp_singular_consonant_correct)
            else:
                df.loc[idx, f'temp_{temp}_tokens_same'] = 0
                df.loc[idx, f'temp_{temp}_vowel_correct'] = 0
                df.loc[idx, f'temp_{temp}_consonant_correct'] = 0
                df.loc[idx, f'temp_{temp}_singular_consonant_correct'] = 0

        # Calculate intervention temperature sampling prefix overlaps and phonetic analysis
        intervention_temp_samples = []
        intervention_temp_vowel_correct = []
        intervention_temp_consonant_correct = []
        intervention_temp_singular_consonant_correct = []
        
        for sample_num in range(1, 6):  # samples 1-5
            col_name = f'intervention_temp_1.0_sample_{sample_num}'
            if col_name in row and pd.notna(row[col_name]):
                intervention_temp_generation = re.sub('.*</think>\n\n', '', str(row[col_name]), flags=re.DOTALL).strip('.')
                intervention_temp_tokens = tokenizer(intervention_temp_generation, return_tensors='pt').input_ids.squeeze(0)
                intervention_temp_min_len = min(original_tokens.size(0), intervention_temp_tokens.size(0))
                intervention_temp_matches = (original_tokens[:intervention_temp_min_len] == intervention_temp_tokens[:intervention_temp_min_len])
                intervention_temp_match_len = (intervention_temp_matches.cumprod(dim=0)).sum().item()
                intervention_temp_samples.append(intervention_temp_match_len)
                
                # Phonetic analysis for intervention temperature sampling
                intervention_temp_last_word = intervention_temp_generation.split()[-1]
                intervention_temp_last_word = re.sub(r'^[^\w]+|[^\w]+$', '', intervention_temp_last_word.strip()).lower()
                
                intervention_temp_vowel = get_last_vowel(intervention_temp_last_word)
                intervention_temp_consonant = get_final_consonant(intervention_temp_last_word)
                intervention_temp_consonant_singular = get_final_consonant(intervention_temp_last_word, depluralize=True)
                
                intervention_temp_vowel_correct.append(rhyme_vowel == intervention_temp_vowel)
                intervention_temp_consonant_correct.append(rhyme_consonant == intervention_temp_consonant)
                intervention_temp_singular_consonant_correct.append(rhyme_consonant_singular == intervention_temp_consonant_singular)
        
        # Take mean of all intervention temperature samples
        if intervention_temp_samples:
            df.loc[idx, 'intervention_temp_1.0_tokens_same'] = sum(intervention_temp_samples) / len(intervention_temp_samples)
            df.loc[idx, 'intervention_temp_1.0_vowel_correct'] = sum(intervention_temp_vowel_correct) / len(intervention_temp_vowel_correct)
            df.loc[idx, 'intervention_temp_1.0_consonant_correct'] = sum(intervention_temp_consonant_correct) / len(intervention_temp_consonant_correct)
            df.loc[idx, 'intervention_temp_1.0_singular_consonant_correct'] = sum(intervention_temp_singular_consonant_correct) / len(intervention_temp_singular_consonant_correct)
        else:
            df.loc[idx, 'intervention_temp_1.0_tokens_same'] = 0
            df.loc[idx, 'intervention_temp_1.0_vowel_correct'] = 0
            df.loc[idx, 'intervention_temp_1.0_consonant_correct'] = 0
            df.loc[idx, 'intervention_temp_1.0_singular_consonant_correct'] = 0

        print("Original:", original_generation)
        print("Intervention:", intervention_generation)
        print("Target rhyme group:", target_rhyme_group)
        print("Intervention last word:", intervention_last_word)
        print("Tokens same:", df.loc[idx, 'tokens_same'])
        print("Original length:", df.loc[idx, 'orig_len'])
        print("Intervention length:", df.loc[idx, 'intervened_len'])
        print("Vowel correct:", df.loc[idx, 'vowel_correct'])
        print("Consonant correct:", df.loc[idx, 'consonant_correct'])
        print("Singular consonant correct:", df.loc[idx, 'singular_consonant_correct'])
        print("Temp 1.0 tokens same:", df.loc[idx, 'temp_1.0_tokens_same'])
        print("Temp 1.0 vowel correct:", df.loc[idx, 'temp_1.0_vowel_correct'])
        print("Intervention temp 1.0 tokens same:", df.loc[idx, 'intervention_temp_1.0_tokens_same'])
        print("Intervention temp 1.0 vowel correct:", df.loc[idx, 'intervention_temp_1.0_vowel_correct'])
        print("--------------------------------")
    
    # Add this model's dataframe to the list
    all_model_dfs.append(df)

#%%
# Combine all model dataframes
combined_df = pd.concat(all_model_dfs, ignore_index=True)

# Calculate metrics for each model
model_metrics = []
for model in models:
    model_data = combined_df[combined_df['model'] == model]
    n = len(model_data)
    
    # Calculate vowel & consonant correct rate
    vowel_consonant_correct_data = (model_data['vowel_correct'] & model_data['consonant_correct']).astype(int)
    vowel_consonant_correct = vowel_consonant_correct_data.mean()
    vowel_consonant_correct_se = vowel_consonant_correct_data.sem()
    
    # Calculate vowel & singular consonant correct rate  
    vowel_singular_consonant_correct_data = (model_data['vowel_correct'] & model_data['singular_consonant_correct']).astype(int)
    vowel_singular_consonant_correct = vowel_singular_consonant_correct_data.mean()
    vowel_singular_consonant_correct_se = vowel_singular_consonant_correct_data.sem()
    
    # Calculate individual rates
    vowel_correct_data = model_data['vowel_correct'].astype(int)
    vowel_correct = vowel_correct_data.mean()
    vowel_correct_se = vowel_correct_data.sem()
    
    consonant_correct_data = model_data['consonant_correct'].astype(int)
    consonant_correct = consonant_correct_data.mean()
    consonant_correct_se = consonant_correct_data.sem()
    
    singular_consonant_correct_data = model_data['singular_consonant_correct'].astype(int)
    singular_consonant_correct = singular_consonant_correct_data.mean()
    singular_consonant_correct_se = singular_consonant_correct_data.sem()
    
    # Calculate tokens_same metrics
    tokens_same_data = model_data['tokens_same']
    mean_tokens_same = tokens_same_data.mean()
    mean_tokens_same_se = tokens_same_data.sem()
    
    tokens_same_pct_orig_data = model_data['tokens_same'] / model_data['orig_len']
    tokens_same_pct_orig = tokens_same_pct_orig_data.mean()
    tokens_same_pct_orig_se = tokens_same_pct_orig_data.sem()
    
    tokens_same_pct_intervened_data = model_data['tokens_same'] / model_data['intervened_len']
    tokens_same_pct_intervened = tokens_same_pct_intervened_data.mean()
    tokens_same_pct_intervened_se = tokens_same_pct_intervened_data.sem()
    
    min_len = model_data[['orig_len', 'intervened_len']].min(axis=1)
    tokens_same_pct_min_data = model_data['tokens_same'] / min_len
    tokens_same_pct_min = tokens_same_pct_min_data.mean()
    tokens_same_pct_min_se = tokens_same_pct_min_data.sem()
    
    # Calculate temperature sampling metrics
    temp_03_tokens_same_data = model_data['temp_0.3_tokens_same']
    temp_03_tokens_same = temp_03_tokens_same_data.mean()
    temp_03_tokens_same_se = temp_03_tokens_same_data.sem()
    
    temp_07_tokens_same_data = model_data['temp_0.7_tokens_same']
    temp_07_tokens_same = temp_07_tokens_same_data.mean()
    temp_07_tokens_same_se = temp_07_tokens_same_data.sem()
    
    temp_10_tokens_same_data = model_data['temp_1.0_tokens_same']
    temp_10_tokens_same = temp_10_tokens_same_data.mean()
    temp_10_tokens_same_se = temp_10_tokens_same_data.sem()
    
    temp_03_pct_orig_data = model_data['temp_0.3_tokens_same'] / model_data['orig_len']
    temp_03_pct_orig = temp_03_pct_orig_data.mean()
    temp_03_pct_orig_se = temp_03_pct_orig_data.sem()
    
    temp_07_pct_orig_data = model_data['temp_0.7_tokens_same'] / model_data['orig_len']
    temp_07_pct_orig = temp_07_pct_orig_data.mean()
    temp_07_pct_orig_se = temp_07_pct_orig_data.sem()
    
    temp_10_pct_orig_data = model_data['temp_1.0_tokens_same'] / model_data['orig_len']
    temp_10_pct_orig = temp_10_pct_orig_data.mean()
    temp_10_pct_orig_se = temp_10_pct_orig_data.sem()
    
    # Calculate intervention temperature sampling metrics
    intervention_temp_10_tokens_same_data = model_data['intervention_temp_1.0_tokens_same']
    intervention_temp_10_tokens_same = intervention_temp_10_tokens_same_data.mean()
    intervention_temp_10_tokens_same_se = intervention_temp_10_tokens_same_data.sem()
    
    intervention_temp_10_pct_orig_data = model_data['intervention_temp_1.0_tokens_same'] / model_data['orig_len']
    intervention_temp_10_pct_orig = intervention_temp_10_pct_orig_data.mean()
    intervention_temp_10_pct_orig_se = intervention_temp_10_pct_orig_data.sem()
    
    # Calculate temperature sampling phonetic metrics
    temp_03_vowel_correct_data = model_data['temp_0.3_vowel_correct']
    temp_03_vowel_correct = temp_03_vowel_correct_data.mean()
    temp_03_vowel_correct_se = temp_03_vowel_correct_data.sem()
    
    temp_07_vowel_correct_data = model_data['temp_0.7_vowel_correct']
    temp_07_vowel_correct = temp_07_vowel_correct_data.mean()
    temp_07_vowel_correct_se = temp_07_vowel_correct_data.sem()
    
    temp_10_vowel_correct_data = model_data['temp_1.0_vowel_correct']
    temp_10_vowel_correct = temp_10_vowel_correct_data.mean()
    temp_10_vowel_correct_se = temp_10_vowel_correct_data.sem()
    
    temp_03_consonant_correct_data = model_data['temp_0.3_consonant_correct']
    temp_03_consonant_correct = temp_03_consonant_correct_data.mean()
    temp_03_consonant_correct_se = temp_03_consonant_correct_data.sem()
    
    temp_07_consonant_correct_data = model_data['temp_0.7_consonant_correct']
    temp_07_consonant_correct = temp_07_consonant_correct_data.mean()
    temp_07_consonant_correct_se = temp_07_consonant_correct_data.sem()
    
    temp_10_consonant_correct_data = model_data['temp_1.0_consonant_correct']
    temp_10_consonant_correct = temp_10_consonant_correct_data.mean()
    temp_10_consonant_correct_se = temp_10_consonant_correct_data.sem()
    
    temp_03_singular_consonant_correct_data = model_data['temp_0.3_singular_consonant_correct']
    temp_03_singular_consonant_correct = temp_03_singular_consonant_correct_data.mean()
    temp_03_singular_consonant_correct_se = temp_03_singular_consonant_correct_data.sem()
    
    temp_07_singular_consonant_correct_data = model_data['temp_0.7_singular_consonant_correct']
    temp_07_singular_consonant_correct = temp_07_singular_consonant_correct_data.mean()
    temp_07_singular_consonant_correct_se = temp_07_singular_consonant_correct_data.sem()
    
    temp_10_singular_consonant_correct_data = model_data['temp_1.0_singular_consonant_correct']
    temp_10_singular_consonant_correct = temp_10_singular_consonant_correct_data.mean()
    temp_10_singular_consonant_correct_se = temp_10_singular_consonant_correct_data.sem()
    
    # Calculate intervention temperature sampling phonetic metrics
    intervention_temp_10_vowel_correct_data = model_data['intervention_temp_1.0_vowel_correct']
    intervention_temp_10_vowel_correct = intervention_temp_10_vowel_correct_data.mean()
    intervention_temp_10_vowel_correct_se = intervention_temp_10_vowel_correct_data.sem()
    
    intervention_temp_10_consonant_correct_data = model_data['intervention_temp_1.0_consonant_correct']
    intervention_temp_10_consonant_correct = intervention_temp_10_consonant_correct_data.mean()
    intervention_temp_10_consonant_correct_se = intervention_temp_10_consonant_correct_data.sem()
    
    intervention_temp_10_singular_consonant_correct_data = model_data['intervention_temp_1.0_singular_consonant_correct']
    intervention_temp_10_singular_consonant_correct = intervention_temp_10_singular_consonant_correct_data.mean()
    intervention_temp_10_singular_consonant_correct_se = intervention_temp_10_singular_consonant_correct_data.sem()
    
    model_metrics.append({
        'model': model,
        'vowel_consonant_correct': vowel_consonant_correct,
        'vowel_consonant_correct_se': vowel_consonant_correct_se,
        'vowel_singular_consonant_correct': vowel_singular_consonant_correct,
        'vowel_singular_consonant_correct_se': vowel_singular_consonant_correct_se,
        'vowel_correct': vowel_correct,
        'vowel_correct_se': vowel_correct_se,
        'consonant_correct': consonant_correct,
        'consonant_correct_se': consonant_correct_se,
        'singular_consonant_correct': singular_consonant_correct,
        'singular_consonant_correct_se': singular_consonant_correct_se,
        'mean_tokens_same': mean_tokens_same,
        'mean_tokens_same_se': mean_tokens_same_se,
        'tokens_same_pct_orig': tokens_same_pct_orig,
        'tokens_same_pct_orig_se': tokens_same_pct_orig_se,
        'tokens_same_pct_intervened': tokens_same_pct_intervened,
        'tokens_same_pct_intervened_se': tokens_same_pct_intervened_se,
        'tokens_same_pct_min': tokens_same_pct_min,
        'tokens_same_pct_min_se': tokens_same_pct_min_se,
        'temp_03_tokens_same': temp_03_tokens_same,
        'temp_03_tokens_same_se': temp_03_tokens_same_se,
        'temp_07_tokens_same': temp_07_tokens_same,
        'temp_07_tokens_same_se': temp_07_tokens_same_se,
        'temp_10_tokens_same': temp_10_tokens_same,
        'temp_10_tokens_same_se': temp_10_tokens_same_se,
        'temp_03_pct_orig': temp_03_pct_orig,
        'temp_03_pct_orig_se': temp_03_pct_orig_se,
        'temp_07_pct_orig': temp_07_pct_orig,
        'temp_07_pct_orig_se': temp_07_pct_orig_se,
        'temp_10_pct_orig': temp_10_pct_orig,
        'temp_10_pct_orig_se': temp_10_pct_orig_se,
        'temp_03_vowel_correct': temp_03_vowel_correct,
        'temp_03_vowel_correct_se': temp_03_vowel_correct_se,
        'temp_07_vowel_correct': temp_07_vowel_correct,
        'temp_07_vowel_correct_se': temp_07_vowel_correct_se,
        'temp_10_vowel_correct': temp_10_vowel_correct,
        'temp_10_vowel_correct_se': temp_10_vowel_correct_se,
        'temp_03_consonant_correct': temp_03_consonant_correct,
        'temp_03_consonant_correct_se': temp_03_consonant_correct_se,
        'temp_07_consonant_correct': temp_07_consonant_correct,
        'temp_07_consonant_correct_se': temp_07_consonant_correct_se,
        'temp_10_consonant_correct': temp_10_consonant_correct,
        'temp_10_consonant_correct_se': temp_10_consonant_correct_se,
        'temp_03_singular_consonant_correct': temp_03_singular_consonant_correct,
        'temp_03_singular_consonant_correct_se': temp_03_singular_consonant_correct_se,
        'temp_07_singular_consonant_correct': temp_07_singular_consonant_correct,
        'temp_07_singular_consonant_correct_se': temp_07_singular_consonant_correct_se,
        'temp_10_singular_consonant_correct': temp_10_singular_consonant_correct,
        'temp_10_singular_consonant_correct_se': temp_10_singular_consonant_correct_se,
        'intervention_temp_10_tokens_same': intervention_temp_10_tokens_same,
        'intervention_temp_10_tokens_same_se': intervention_temp_10_tokens_same_se,
        'intervention_temp_10_pct_orig': intervention_temp_10_pct_orig,
        'intervention_temp_10_pct_orig_se': intervention_temp_10_pct_orig_se,
        'intervention_temp_10_vowel_correct': intervention_temp_10_vowel_correct,
        'intervention_temp_10_vowel_correct_se': intervention_temp_10_vowel_correct_se,
        'intervention_temp_10_consonant_correct': intervention_temp_10_consonant_correct,
        'intervention_temp_10_consonant_correct_se': intervention_temp_10_consonant_correct_se,
        'intervention_temp_10_singular_consonant_correct': intervention_temp_10_singular_consonant_correct,
        'intervention_temp_10_singular_consonant_correct_se': intervention_temp_10_singular_consonant_correct_se
    })

metrics_df = pd.DataFrame(model_metrics)

# Plot 1: Mean tokens_same by model
plt.figure(figsize=(10, 6))
x_positions = range(len(models))

plt.errorbar(x_positions, metrics_df['mean_tokens_same'], 
             yerr=metrics_df['mean_tokens_same_se'],
             marker='o', linewidth=2, markersize=8, capsize=5,
             label='Mean Tokens Same', color='blue')

plt.xlabel('Model')
plt.ylabel('Mean Tokens Same')
plt.title('Mean Tokens Same by Model')
plt.xticks(x_positions, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tokens_same_mean.pdf")
plt.show()

# Plot 2: Tokens same as percentage of different length measures
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Percentage of original length
axes[0].errorbar(x_positions, metrics_df['tokens_same_pct_orig'], 
                 yerr=metrics_df['tokens_same_pct_orig_se'],
                 marker='o', linewidth=2, markersize=8, capsize=5, color='green')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Tokens Same / Original Length')
axes[0].set_title('Tokens Same as % of Original Length')
axes[0].set_xticks(x_positions)
axes[0].set_xticklabels(models, rotation=45)
axes[0].grid(True, alpha=0.3)

# Percentage of intervention length
axes[1].errorbar(x_positions, metrics_df['tokens_same_pct_intervened'], 
                 yerr=metrics_df['tokens_same_pct_intervened_se'],
                 marker='s', linewidth=2, markersize=8, capsize=5, color='red')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Tokens Same / Intervention Length')
axes[1].set_title('Tokens Same as % of Intervention Length')
axes[1].set_xticks(x_positions)
axes[1].set_xticklabels(models, rotation=45)
axes[1].grid(True, alpha=0.3)

# Percentage of minimum length
axes[2].errorbar(x_positions, metrics_df['tokens_same_pct_min'], 
                 yerr=metrics_df['tokens_same_pct_min_se'],
                 marker='^', linewidth=2, markersize=8, capsize=5, color='purple')
axes[2].set_xlabel('Model')
axes[2].set_ylabel('Tokens Same / Min Length')
axes[2].set_title('Tokens Same as % of Min Length')
axes[2].set_xticks(x_positions)
axes[2].set_xticklabels(models, rotation=45)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tokens_same_percentages.pdf")
plt.show()

# Plot 3: Temperature sampling mean tokens same
plt.figure(figsize=(12, 6))
x_positions = range(len(models))

plt.errorbar(x_positions, metrics_df['mean_tokens_same'], 
             yerr=metrics_df['mean_tokens_same_se'],
             marker='o', linewidth=2, markersize=8, capsize=5,
             label='Intervention Greedy', color='blue')

plt.errorbar(x_positions, metrics_df['temp_10_tokens_same'], 
             yerr=metrics_df['temp_10_tokens_same_se'],
             marker='d', linewidth=2, markersize=8, capsize=5,
             label='Normal Temperature 1.0', color='purple')

plt.errorbar(x_positions, metrics_df['intervention_temp_10_tokens_same'], 
             yerr=metrics_df['intervention_temp_10_tokens_same_se'],
             marker='*', linewidth=2, markersize=10, capsize=5,
             label='Intervention Temperature 1.0', color='orange')

plt.xlabel('Model')
plt.ylabel('Mean Tokens Same')
plt.title('Mean Tokens Same by Model: Intervention vs Temperature 1.0 Sampling')
plt.xticks(x_positions, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("temperature_sampling_tokens_same.pdf")
plt.show()

# Plot 4: Temperature sampling as percentage of original length
plt.figure(figsize=(12, 6))

plt.errorbar(x_positions, metrics_df['tokens_same_pct_orig'], 
             yerr=metrics_df['tokens_same_pct_orig_se'],
             marker='o', linewidth=2, markersize=8, capsize=5,
             label='Intervention Greedy', color='blue')

plt.errorbar(x_positions, metrics_df['temp_10_pct_orig'], 
             yerr=metrics_df['temp_10_pct_orig_se'],
             marker='d', linewidth=2, markersize=8, capsize=5,
             label='Normal Temperature 1.0', color='purple')

plt.errorbar(x_positions, metrics_df['intervention_temp_10_pct_orig'], 
             yerr=metrics_df['intervention_temp_10_pct_orig_se'],
             marker='*', linewidth=2, markersize=10, capsize=5,
             label='Intervention Temperature 1.0', color='orange')

plt.xlabel('Model')
plt.ylabel('Tokens Same / Original Length')
plt.title('Tokens Same as % of Original Length: Intervention vs Temperature 1.0 Sampling')
plt.xticks(x_positions, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("temperature_sampling_pct_orig.pdf")
plt.show()

# Plot 5: Temperature sampling vowel correctness
plt.figure(figsize=(12, 6))

plt.errorbar(x_positions, metrics_df['vowel_correct'], 
             yerr=metrics_df['vowel_correct_se'],
             marker='o', linewidth=2, markersize=8, capsize=5,
             label='Intervention Greedy', color='blue')

plt.errorbar(x_positions, metrics_df['temp_10_vowel_correct'], 
             yerr=metrics_df['temp_10_vowel_correct_se'],
             marker='d', linewidth=2, markersize=8, capsize=5,
             label='Normal Temperature 1.0', color='purple')

plt.errorbar(x_positions, metrics_df['intervention_temp_10_vowel_correct'], 
             yerr=metrics_df['intervention_temp_10_vowel_correct_se'],
             marker='*', linewidth=2, markersize=10, capsize=5,
             label='Intervention Temperature 1.0', color='orange')

plt.xlabel('Model')
plt.ylabel('Vowel Correctness Rate')
plt.title('Vowel Correctness by Model: Intervention vs Temperature 1.0 Sampling')
plt.xticks(x_positions, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("temperature_sampling_vowel_correctness.pdf")
plt.show()

# Plot 6: Temperature sampling consonant correctness
plt.figure(figsize=(12, 6))

plt.errorbar(x_positions, metrics_df['consonant_correct'], 
             yerr=metrics_df['consonant_correct_se'],
             marker='o', linewidth=2, markersize=8, capsize=5,
             label='Intervention Greedy', color='blue')

plt.errorbar(x_positions, metrics_df['temp_10_consonant_correct'], 
             yerr=metrics_df['temp_10_consonant_correct_se'],
             marker='d', linewidth=2, markersize=8, capsize=5,
             label='Normal Temperature 1.0', color='purple')

plt.errorbar(x_positions, metrics_df['intervention_temp_10_consonant_correct'], 
             yerr=metrics_df['intervention_temp_10_consonant_correct_se'],
             marker='*', linewidth=2, markersize=10, capsize=5,
             label='Intervention Temperature 1.0', color='orange')

plt.xlabel('Model')
plt.ylabel('Consonant Correctness Rate')
plt.title('Consonant Correctness by Model: Intervention vs Temperature 1.0 Sampling')
plt.xticks(x_positions, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("temperature_sampling_consonant_correctness.pdf")
plt.show()

# Plot 7: Rhyme correctness (vowel & consonant correct) across intervention and temperature sampling
plt.figure(figsize=(12, 6))

# Calculate vowel & consonant correct for intervention and temperature sampling
# For combined metrics, we need to calculate standard errors from the raw data
temp_10_vowel_consonant_correct = []
temp_10_vowel_consonant_correct_se = []
intervention_temp_10_vowel_consonant_correct = []
intervention_temp_10_vowel_consonant_correct_se = []

for model in models:
    model_data = combined_df[combined_df['model'] == model]
    
    # Temperature 1.0
    temp_10_combined = (model_data['temp_1.0_vowel_correct'] * model_data['temp_1.0_consonant_correct'])
    temp_10_vowel_consonant_correct.append(temp_10_combined.mean())
    temp_10_vowel_consonant_correct_se.append(temp_10_combined.sem())
    
    # Intervention Temperature 1.0
    intervention_temp_10_combined = (model_data['intervention_temp_1.0_vowel_correct'] * model_data['intervention_temp_1.0_consonant_correct'])
    intervention_temp_10_vowel_consonant_correct.append(intervention_temp_10_combined.mean())
    intervention_temp_10_vowel_consonant_correct_se.append(intervention_temp_10_combined.sem())

plt.errorbar(x_positions, metrics_df['vowel_consonant_correct'], 
             yerr=metrics_df['vowel_consonant_correct_se'],
             marker='o', linewidth=2, markersize=8, capsize=5,
             label='Intervention Greedy', color='blue')

plt.errorbar(x_positions, temp_10_vowel_consonant_correct, 
             yerr=temp_10_vowel_consonant_correct_se,
             marker='d', linewidth=2, markersize=8, capsize=5,
             label='Normal Temperature 1.0', color='purple')

plt.errorbar(x_positions, intervention_temp_10_vowel_consonant_correct, 
             yerr=intervention_temp_10_vowel_consonant_correct_se,
             marker='*', linewidth=2, markersize=10, capsize=5,
             label='Intervention Temperature 1.0', color='orange')

plt.xlabel('Model')
plt.ylabel('Rhyme Correctness Rate (Vowel & Consonant)')
plt.title('Rhyme Correctness by Model: Intervention vs Temperature 1.0 Sampling')
plt.xticks(x_positions, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rhyme_correctness_comparison.pdf")
plt.show()

# Plot 8: Conditioned on vowel_consonant_correct
# First add the vowel_consonant_correct column to the combined dataframe
combined_df['vowel_consonant_correct'] = combined_df['vowel_correct'] & combined_df['consonant_correct']

# Calculate metrics conditioned on vowel_consonant_correct
conditioned_metrics = []
for model in models:
    model_data = combined_df[combined_df['model'] == model]
    
    for condition in [True, False]:
        condition_data = model_data[model_data['vowel_consonant_correct'] == condition]
        
        if len(condition_data) > 0:
            tokens_same_data = condition_data['tokens_same']
            mean_tokens_same = tokens_same_data.mean()
            mean_tokens_same_se = tokens_same_data.sem()
            
            tokens_same_pct_orig_data = condition_data['tokens_same'] / condition_data['orig_len']
            tokens_same_pct_orig = tokens_same_pct_orig_data.mean()
            tokens_same_pct_orig_se = tokens_same_pct_orig_data.sem()
            
            tokens_same_pct_intervened_data = condition_data['tokens_same'] / condition_data['intervened_len']
            tokens_same_pct_intervened = tokens_same_pct_intervened_data.mean()
            tokens_same_pct_intervened_se = tokens_same_pct_intervened_data.sem()
            
            min_len = condition_data[['orig_len', 'intervened_len']].min(axis=1)
            tokens_same_pct_min_data = condition_data['tokens_same'] / min_len
            tokens_same_pct_min = tokens_same_pct_min_data.mean()
            tokens_same_pct_min_se = tokens_same_pct_min_data.sem()
            
            count = len(condition_data)
        else:
            mean_tokens_same = 0
            mean_tokens_same_se = 0
            tokens_same_pct_orig = 0
            tokens_same_pct_orig_se = 0
            tokens_same_pct_intervened = 0
            tokens_same_pct_intervened_se = 0
            tokens_same_pct_min = 0
            tokens_same_pct_min_se = 0
            count = 0
        
        conditioned_metrics.append({
            'model': model,
            'vowel_consonant_correct': condition,
            'mean_tokens_same': mean_tokens_same,
            'mean_tokens_same_se': mean_tokens_same_se,
            'tokens_same_pct_orig': tokens_same_pct_orig,
            'tokens_same_pct_orig_se': tokens_same_pct_orig_se,
            'tokens_same_pct_intervened': tokens_same_pct_intervened,
            'tokens_same_pct_intervened_se': tokens_same_pct_intervened_se,
            'tokens_same_pct_min': tokens_same_pct_min,
            'tokens_same_pct_min_se': tokens_same_pct_min_se,
            'count': count
        })

conditioned_df = pd.DataFrame(conditioned_metrics)

# Plot conditioned results
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Mean tokens same
for i, condition in enumerate([True, False]):
    condition_data = conditioned_df[conditioned_df['vowel_consonant_correct'] == condition]
    label = f'Vowel & Consonant Correct: {condition}'
    color = 'green' if condition else 'red'
    axes[0, 0].errorbar(x_positions, condition_data['mean_tokens_same'], 
                        yerr=condition_data['mean_tokens_same_se'],
                        marker='o', linewidth=2, markersize=8, capsize=5,
                        label=label, color=color)

axes[0, 0].set_xlabel('Model')
axes[0, 0].set_ylabel('Mean Tokens Same')
axes[0, 0].set_title('Mean Tokens Same by Model (Conditioned)')
axes[0, 0].set_xticks(x_positions)
axes[0, 0].set_xticklabels(models, rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Percentage of original length
for i, condition in enumerate([True, False]):
    condition_data = conditioned_df[conditioned_df['vowel_consonant_correct'] == condition]
    label = f'Vowel & Consonant Correct: {condition}'
    color = 'green' if condition else 'red'
    axes[0, 1].errorbar(x_positions, condition_data['tokens_same_pct_orig'], 
                        yerr=condition_data['tokens_same_pct_orig_se'],
                        marker='s', linewidth=2, markersize=8, capsize=5,
                        label=label, color=color)

axes[0, 1].set_xlabel('Model')
axes[0, 1].set_ylabel('Tokens Same / Original Length')
axes[0, 1].set_title('Tokens Same as % of Original Length (Conditioned)')
axes[0, 1].set_xticks(x_positions)
axes[0, 1].set_xticklabels(models, rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Percentage of intervention length
for i, condition in enumerate([True, False]):
    condition_data = conditioned_df[conditioned_df['vowel_consonant_correct'] == condition]
    label = f'Vowel & Consonant Correct: {condition}'
    color = 'green' if condition else 'red'
    axes[1, 0].errorbar(x_positions, condition_data['tokens_same_pct_intervened'], 
                        yerr=condition_data['tokens_same_pct_intervened_se'],
                        marker='^', linewidth=2, markersize=8, capsize=5,
                        label=label, color=color)

axes[1, 0].set_xlabel('Model')
axes[1, 0].set_ylabel('Tokens Same / Intervention Length')
axes[1, 0].set_title('Tokens Same as % of Intervention Length (Conditioned)')
axes[1, 0].set_xticks(x_positions)
axes[1, 0].set_xticklabels(models, rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Percentage of minimum length
for i, condition in enumerate([True, False]):
    condition_data = conditioned_df[conditioned_df['vowel_consonant_correct'] == condition]
    label = f'Vowel & Consonant Correct: {condition}'
    color = 'green' if condition else 'red'
    axes[1, 1].errorbar(x_positions, condition_data['tokens_same_pct_min'], 
                        yerr=condition_data['tokens_same_pct_min_se'],
                        marker='d', linewidth=2, markersize=8, capsize=5,
                        label=label, color=color)

axes[1, 1].set_xlabel('Model')
axes[1, 1].set_ylabel('Tokens Same / Min Length')
axes[1, 1].set_title('Tokens Same as % of Min Length (Conditioned)')
axes[1, 1].set_xticks(x_positions)
axes[1, 1].set_xticklabels(models, rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tokens_same_conditioned.pdf")
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print("==================")
for _, row in metrics_df.iterrows():
    print(f"{row['model']}:")
    print(f"  Vowel & Consonant Correct: {row['vowel_consonant_correct']:.3f}")
    print(f"  Vowel & Singular Consonant Correct: {row['vowel_singular_consonant_correct']:.3f}")
    print(f"  Vowel Correct: {row['vowel_correct']:.3f}")
    print(f"  Consonant Correct: {row['consonant_correct']:.3f}")
    print(f"  Singular Consonant Correct: {row['singular_consonant_correct']:.3f}")
    print(f"  Mean Tokens Same: {row['mean_tokens_same']:.3f}")
    print(f"  Tokens Same % of Original: {row['tokens_same_pct_orig']:.3f}")
    print(f"  Tokens Same % of Intervention: {row['tokens_same_pct_intervened']:.3f}")
    print(f"  Tokens Same % of Min Length: {row['tokens_same_pct_min']:.3f}")
    print(f"  Temperature 1.0 Tokens Same: {row['temp_10_tokens_same']:.3f}")
    print(f"  Temperature 1.0 % of Original: {row['temp_10_pct_orig']:.3f}")
    print(f"  Temperature 1.0 Vowel Correct: {row['temp_10_vowel_correct']:.3f}")
    print(f"  Temperature 1.0 Consonant Correct: {row['temp_10_consonant_correct']:.3f}")
    print(f"  Temperature 1.0 Singular Consonant Correct: {row['temp_10_singular_consonant_correct']:.3f}")
    print(f"  Intervention Temperature 1.0 Tokens Same: {row['intervention_temp_10_tokens_same']:.3f}")
    print(f"  Intervention Temperature 1.0 % of Original: {row['intervention_temp_10_pct_orig']:.3f}")
    print(f"  Intervention Temperature 1.0 Vowel Correct: {row['intervention_temp_10_vowel_correct']:.3f}")
    print(f"  Intervention Temperature 1.0 Consonant Correct: {row['intervention_temp_10_consonant_correct']:.3f}")
    print(f"  Intervention Temperature 1.0 Singular Consonant Correct: {row['intervention_temp_10_singular_consonant_correct']:.3f}")
    print()

# Print conditioned summary statistics
print("\nConditioned Summary Statistics:")
print("===============================")
for model in models:
    print(f"{model}:")
    model_conditioned = conditioned_df[conditioned_df['model'] == model]
    
    for _, row in model_conditioned.iterrows():
        condition = row['vowel_consonant_correct']
        print(f"  Vowel & Consonant Correct = {condition} (n={row['count']}):")
        print(f"    Mean Tokens Same: {row['mean_tokens_same']:.3f}")
        print(f"    Tokens Same % of Original: {row['tokens_same_pct_orig']:.3f}")
        print(f"    Tokens Same % of Intervention: {row['tokens_same_pct_intervened']:.3f}")
        print(f"    Tokens Same % of Min Length: {row['tokens_same_pct_min']:.3f}")
    print()
#%%
# Plot mean generation lengths by model
plt.figure(figsize=(12, 6))
x_positions = range(len(models))

# Calculate mean lengths and standard errors for each model
mean_orig_lengths = []
mean_orig_lengths_se = []
mean_intervened_lengths = []
mean_intervened_lengths_se = []
mean_min_lengths = []
mean_min_lengths_se = []

for model in models:
    model_data = combined_df[combined_df['model'] == model]
    
    orig_len_data = model_data['orig_len']
    mean_orig_lengths.append(orig_len_data.mean())
    mean_orig_lengths_se.append(orig_len_data.sem())
    
    intervened_len_data = model_data['intervened_len']
    mean_intervened_lengths.append(intervened_len_data.mean())
    mean_intervened_lengths_se.append(intervened_len_data.sem())
    
    min_len_data = model_data[['orig_len', 'intervened_len']].min(axis=1)
    mean_min_lengths.append(min_len_data.mean())
    mean_min_lengths_se.append(min_len_data.sem())

plt.errorbar(x_positions, mean_orig_lengths, 
             yerr=mean_orig_lengths_se,
             marker='o', linewidth=2, markersize=8, capsize=5,
             label='Original Generation Length', color='blue')

plt.errorbar(x_positions, mean_intervened_lengths, 
             yerr=mean_intervened_lengths_se,
             marker='s', linewidth=2, markersize=8, capsize=5,
             label='Intervention Generation Length', color='red')

plt.errorbar(x_positions, mean_min_lengths, 
             yerr=mean_min_lengths_se,
             marker='^', linewidth=2, markersize=8, capsize=5,
             label='Min Generation Length', color='green')

plt.xlabel('Model')
plt.ylabel('Mean Token Length')
plt.title('Mean Generation Length by Model')
plt.xticks(x_positions, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mean_generation_lengths.pdf")
plt.show()

# Print length statistics
print("\nGeneration Length Statistics:")
print("=============================")
for i, model in enumerate(models):
    print(f"{model}:")
    print(f"  Mean Original Length: {mean_orig_lengths[i]:.2f}")
    print(f"  Mean Intervention Length: {mean_intervened_lengths[i]:.2f}")
    print(f"  Mean Min Length: {mean_min_lengths[i]:.2f}")
    print()
         
# %%
