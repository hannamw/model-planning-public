#%%
from pathlib import Path
import json
import shutil

old_dir = 'graph_files_diff'
new_dir = 'graph_files_diff_new'
dirs = [x.split('/')[-1] for x in  {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"}.keys()]
metadata = {'graphs':[]}
for dir in dirs:
    for exdir in Path(old_dir + '/' + dir).iterdir():
        with open(exdir / 'graph-metadata.json') as f:
            json_content = json.load(f)
        metadata['graphs'].append(json_content['graphs'][0])
        filename = f'{dir}-{exdir.stem}.json'
        file = exdir / filename
        
        Path(new_dir).mkdir(exist_ok=True)
        new_path = Path(new_dir) / filename
        shutil.copy2(file, new_path)
        print("copying", str(file), "to", str(new_path))


print(metadata)
with open(f'{new_dir}/graph-metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
# %%
