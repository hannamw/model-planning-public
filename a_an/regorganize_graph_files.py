#%%
from pathlib import Path
import json
import shutil

new_dir = 'graph_files_new'
dirs = ['qwen3-0.6b-relu-lowl0','qwen3-1.7b-relu-lowl0','qwen3-4b-relu',
'qwen3-8b-relu','qwen3-14b-relu-lowl0']
metadata = {'graphs':[]}
for dir in dirs:
    for exdir in Path('graph_files/' + dir).iterdir():
        with open(exdir / 'graph-metadata.json') as f:
            json_content = json.load(f)
        metadata['graphs'].append(json_content['graphs'][0])
        filename = f'{dir}-{exdir.stem}.json'
        file = exdir / filename
        
        #Path(new_dir).mkdir(exist_ok=True)
        new_path = Path(new_dir) / filename
        #shutil.copy2(file, new_path)
        print("copying", str(file), "to", str(new_path))


print(metadata)
with open(f'{new_dir}/graph-metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
# %%
