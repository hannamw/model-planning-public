#%%
import pandas as pd

def format_country_capital(file='pre-data/country_capital.csv'):
    df = pd.read_csv(file)
    prompts = [f'The country containing {city} has its capital in the city of' for city in df['Largest_City']]
    intermediate_exps = [f'The country containing {city}' for city in df['Largest_City']]
    hop1s = [f'What is the country containing {city}?' for city in df['Largest_City']]
    hop2s = [f'{intermediate} has its capital in the city of' for intermediate in df['Country']]
    d = {'question': prompts, 'intermediate': df['Country'], 'answer': df['Capital'], 'intermediate_expression': intermediate_exps, "prompt_type": "country_capital", 'prompt_subtype': "country_capital", 'hop1': hop1s, 'hop2':hop2s}

    return d

def format_state_capital(file='pre-data/state_capital.csv'):
    df = pd.read_csv(file)
    prompts = [f'The state containing {city} has its capital in the city of' for city in df['Largest_City']]
    intermediate_exps = [f'The state containing {city}' for city in df['Largest_City']]
    hop1s = [f'What is the state containing {city}?' for city in df['Largest_City']]
    hop2s = [f'{intermediate} has its capital in the city of' for intermediate in df['State']]
    d = {'question': prompts, 'intermediate': df['State'], 'answer': df['Capital'], 'intermediate_expression': intermediate_exps, "prompt_type": "state_capital", 'prompt_subtype': "state_capital", 'hop1': hop1s, 'hop2':hop2s}

    return d

def format_math_data(file='pre-data/math_data.csv'):
    df = pd.read_csv(file)
    hop1s = [f'{exp}. {exp[0]}=' for exp in df['eq1']]
    hop2s = [question.replace(intermediate_expression + f". {question[0]}", str(intermediate)) for question, intermediate_expression, intermediate in zip(df['eqfull'], df['eq1'], df['intermediate'])]
    d = {'question': df['eqfull'], 'intermediate': df['intermediate'], 'answer': df['answer'], 'intermediate_expression': df['eq1'], "prompt_type": "math", 'prompt_subtype': "math", 'hop1': hop1s, 'hop2':hop2s}

    return d

def format_math_data_novar(file='pre-data/math_data_novar.csv'):
    df = pd.read_csv(file)
    hop1s = [f'{exp}=' for exp in df['eq1']]
    hop2s = [question.replace(intermediate_expression, str(intermediate)) for question, intermediate_expression, intermediate in zip(df['eqfull'], df['eq1'], df['intermediate'])]
    d = {'question': df['eqfull'], 'intermediate': df['intermediate'], 'answer': df['answer'], 'intermediate_expression': df['eq1'], "prompt_type": "math_novar", 'prompt_subtype': "math_novar", 'hop1': hop1s, 'hop2':hop2s}

    return d

def format_two_digit_addition(file='pre-data/two_digit_addition_data.csv'):
    df = pd.read_csv(file)
    hop1s = df['eqfull']
    hop2s = df['eqfull']
    d = {'question': df['eqfull'], 'intermediate': df['intermediate'], 'answer': df['answer'], 'intermediate_expression': df['eqfull'], "prompt_type": "two_digit_addition", 'prompt_subtype': "two_digit_addition", 'hop1': hop1s, 'hop2':hop2s}

    return d

def format_translation(file='pre-data/translation.csv', language='spanish'):
    df = pd.read_csv(file)
    prompt_types = [f'translation-{relation_type}-{language}' for relation_type in df['relation_type']]
    intermediate_expressions = [query.split(' is called what')[0] for query in df['foreign_language_query']]
    
    if language == 'spanish':
        answer_col = 'spanish_translation' 
    elif language == 'french':
        answer_col = 'french_translation'
    else:
        raise ValueError(f"Unknown language: {language}")

    prompts = [s.replace("in X", f'in {language.capitalize()}') for s in df['foreign_language_query']]

    hop1s = [f'What is {exp.lower()}?' for exp in intermediate_expressions]
    hop2s = [question.replace(intermediate_expression, str(intermediate)).capitalize() for question, intermediate_expression, intermediate in zip(prompts, intermediate_expressions, df['intermediate'])]

    d = {'question': prompts, 'intermediate': df['intermediate'], 'answer': df[answer_col], 'intermediate_expression': intermediate_expressions, "prompt_type": f'translation-{language}', 
    'prompt_subtype': prompt_types, 'hop1': hop1s, 'hop2':hop2s}
    return d

def format_translation_claude(file='pre-data/claude-translation.csv', language='spanish'):
    df = pd.read_csv(file)
    intermediate_expressions = [query.split(' is called what')[0] for query in df['foreign_language_query']]
    
    if language == 'spanish':
        answer_col = 'spanish_translation' 
    elif language == 'french':
        answer_col = 'french_translation'
    else:
        raise ValueError(f"Unknown language: {language}")

    prompts = [s.replace("in X", f'in {language.capitalize()}') for s in df['foreign_language_query']]

    hop1s = [f'What is {exp.lower()}?' for exp in intermediate_expressions]
    hop2s = [question.replace(intermediate_expression, str(intermediate)).capitalize() for question, intermediate_expression, intermediate in zip(prompts, intermediate_expressions, df['intermediate'])]

    d = {'question': prompts, 'intermediate': df['intermediate'], 'answer': df[answer_col], 'intermediate_expression': intermediate_expressions, 
    "prompt_type": f'claude-translation', 'prompt_subtype': f'claude-translation-{language}', 'hop1': hop1s, 'hop2':hop2s}
    return d

def format_fictional_characters(file='pre-data/fictional_characters_filtered.csv'):
    df = pd.read_csv(file, delimiter=';')

    hop1s = [f'Who is {exp.lower()}?' for exp in df['Entity_Description']]
    hop2s = [question.replace(intermediate_expression, intermediate) for question, intermediate_expression, intermediate in zip(df['Question'], df['Entity_Description'], df['Entity'])]

    d = {'question': df['Question'], 'intermediate': df['Entity'], 'answer': df['Answer'], 'intermediate_expression': df['Entity_Description'], 
        "prompt_type": "fictional_characters", 'prompt_subtype': "fictional_characters", 'hop1': hop1s, 'hop2':hop2s}

    return d

def format_socrates(file='pre-data/SOCRATES_v1.csv'):
    df = pd.read_csv(file)
    prompt_types = [f'socrates-{category}' for category in df['category']]
    hop1s = [f'What is {exp}?' for exp in df['r2(r1(e1)).subject']]
    hop2s = [question.replace(intermediate_expression, intermediate) for question, intermediate_expression, intermediate in zip(df['r2(r1(e1)).prompt'], df['r2(r1(e1)).subject'], df['r2(e2).subject'])]
    d = {'question': df['r2(r1(e1)).prompt'], 'intermediate': df['r2(e2).subject'], 'answer': df['e3.value'], 
    'intermediate_expression': df['r2(r1(e1)).subject'], "prompt_type": 'socrates', 'prompt_subtype': prompt_types, 'hop1': hop1s, 'hop2':hop2s}

    return d

# Load and combine all datasets
datasets = [
    pd.DataFrame(format_country_capital()),
    pd.DataFrame(format_state_capital()),
    pd.DataFrame(format_translation(language='spanish')),
    pd.DataFrame(format_translation(language='french')),
    pd.DataFrame(format_translation_claude(language='spanish')),
    pd.DataFrame(format_translation_claude(language='french')),
    pd.DataFrame(format_fictional_characters()),
    pd.DataFrame(format_math_data()),
    pd.DataFrame(format_math_data_novar()),
    pd.DataFrame(format_two_digit_addition()),
    pd.DataFrame(format_socrates()),
]

combined_df = pd.concat(datasets, ignore_index=True)
combined_df.to_csv('data/combined_multihop_dataset.csv', index=False)
# %%
