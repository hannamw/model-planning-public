#%%
from functools import partial
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

from circuit_tracer.attribution import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.frontend.upload_graph_to_s3 import upload_graph_to_s3

def print_topk(model, logits:torch.Tensor, k=5):
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)
    for i in range(k):
        print(model.tokenizer.decode([topk.indices[i]]), ':', topk.values[i].item())

def chattify(inputs: List[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified


model_name = "Qwen/Qwen3-14B"
model_config = 'circuit-tracer-dev/circuit_tracer/configs/qwen3-14b-relu-lowl0.yaml'

model_short_name = Path(model_config).stem
print(model_short_name)
model = ReplacementModel.from_pretrained(model_name, 
                                        model_config, 
                                        transcoders_offload='disk', 
                                        dtype=torch.bfloat16)

chattify = partial(chattify, tokenizer=model.tokenizer)

Example = namedtuple("Example", ["sentence", "continuation", "name"])
#%%
prompt = """/no_think Generate a function body that counts the individual letters of a string. Write only code.

def count(s: str):
"""
response_one_line = """<think>

</think>

```python
   """

response_one_line2 = """<think>

</think>

```python
   return """

# response_one_line = """<think>

# </think>

# ```python
#     return {char: s.count(char) for char in set(s)}"""

response_multi_line = """<think>

</think>

```python
    counts = {}
    for char in s:
        counts[char] = counts.get(char, 0) + 1
    return counts
```"""

response_multi_line2 = """<think>

</think>

```python
    counts = 
```"""

# response_multi_line = """<think>

# </think>

# ```python
#     counts = {}
#     for char in s:
#         counts[char] = counts.get(char, 0) + 1
#     return counts
# ```"""
factorial_iterative_prompt = "/no_think Generate a function body that computes the factorial function iteratively, with no error raising. Write only code."
factorial_recursive_prompt = "/no_think Generate a function body that computes the factorial function recursively, with no error raising. Write only code."
factorial_iterative_response1 = """<think>

</think>

```python
def factorial(n: int) -> int:
   """

factorial_iterative_response2 = """<think>

</think>

```python
def factorial(n: int) -> int:
    result = 1
   """

factorial_iterative_response = """<think>

</think>

```python
def factorial(n: int) -> int:
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result"""

factorial_recursive_response1 = """<think>

</think>

```python
def factorial(n: int) -> int:
   """

factorial_recursive_response2 = """<think>

</think>

```python
def factorial(n: int) -> int:
    if n == 0:
        return 1
   """

factorial_recursive_response = """<think>

</think>

```python
def factorial(n: int) -> int:
    if n == 0:
        return 1
    return n * factorial(n - 1)"""

point_prompt  = """/no_think Generate a function body based on the following context and type signature. Put each call on its own line.

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y=y

class Line():
    def __init__(self, point1, point2):
        self.point1=point1
        self.point2=point2

def make_a_line(x1,y1,x2,y2):
"""

point_response = """<think>

</think>

def make_a_line(x1, y1, x2, y2):
   """

point_response2 = """<think>

</think>

def make_a_line(x1, y1, x2, y2):
    point1 = Point(x1, y1)
    point2 = Point(x2, y2)
    line ="""

point_response_full = """<think>

</think>

def make_a_line(x1, y1, x2, y2):
    point1 = Point(x1, y1)
    point2 = Point(x2, y2)
    line = Line(point1, point2)
    return line"""

plane_prompt = """/no_think Generate a function body based on the following context and type signature. Put each call on its own line.

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y=y

class Line():
    def __init__(self, point1, point2):
        self.point1=point1
        self.point2=point2

class Plane():
    def __init__(self, line1, line2):
        self.line1=line1
        self.line2=line2

def make_a_plane(x1,y1,x2,y2,x3,y3,x4,y4):
"""
plane_response = """<think>

</think>

def make_a_plane(x1, y1, x2, y2, x3, y3, x4, y4):
   """
plane_response2 = """<think>

</think>

def make_a_plane(x1, y1, x2, y2, x3, y3, x4, y4):
    point1 = Point(x1, y1)
    point2 = Point(x2, y2)
    point3 = Point(x3, y3)
    point4 = Point(x4, y4)
   """

plane_response_full = """<think>

</think>

def make_a_plane(x1, y1, x2, y2, x3, y3, x4, y4):
    point1 = Point(x1, y1)
    point2 = Point(x2, y2)
    point3 = Point(x3, y3)
    point4 = Point(x4, y4)
    line1 = Line(point1, point2)
    line2 = Line(point3, point4)
    plane = Plane(line1, line2)
    return plane"""

rectangle_prompt = """/no_think Generate a function body based on the following context and type signature.
If you instantiate any objects, put each of them on its own, separate line:

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class CoordinateArrays:
    def __init__(self, x_coords: tuple[float, float], y_coords: tuple[float, float]):
        self.x_coords = x_coords
        self.y_coords = y_coords


class Rectangle:
    def __init__(self, x0, y0, x1, y1):
        pass

    @classmethod
    def from_points(cls, top_left: Point, bottom_right: Point) -> "Rectangle":
        return cls(top_left.x, top_left.y, bottom_right.x, bottom_right.y)

    @classmethod
    def from_coordinates(cls, coordinates: CoordinateArrays) -> "Rectangle":
        return cls(
            coordinates.x_coords[0],
            coordinates.y_coords[0],
            coordinates.x_coords[1],
            coordinates.y_coords[1],
        )

def make_rectangle(x0: float, y0: float, x1: float, y1: float) -> Rectangle:
# make a rectangle using a classmethod"""
rectangle_response = """<think>

</think>

```python
    # Create a"""

rectangle_response_full = """<think>

</think>

```python
    # Create a Point object for the top-left corner
    top_left = Point(x0, y0)
    
    # Create a Point object for the bottom-right corner
    bottom_right = Point(x1, y1)
    
    # Use the from_points class method to create and return a Rectangle
    return Rectangle.from_points(top_left, bottom_right)"""

examples = [
    # Example(chattify([prompt, response_one_line]),
    # " return", "count-chars-return"),
    # Example(chattify([prompt, response_multi_line]),
    # " counts", "count-chars-counts"),
    # Example(chattify([factorial_iterative_prompt, factorial_iterative_response1]),
    # " result", "factorial-iterative1"),
    # Example(chattify([factorial_iterative_prompt, factorial_iterative_response2]),
    # " for", "factorial-iterative2"),
    # Example(chattify([factorial_recursive_prompt, factorial_recursive_response1]),
    # " if", "factorial-recursive1"),
    # Example(chattify([factorial_recursive_prompt, factorial_recursive_response2]),
    # " return", "factorial-recursive2"),
    # Example(chattify([point_prompt, point_response]),
    # " point1", "point-construction"),
    # Example(chattify([point_prompt, point_response2]),
    # " Line", "point-construction2"),
    # Example(chattify([plane_prompt, plane_response]),
    # " point", "plane-1"),
    # Example(chattify([plane_prompt, plane_response2]),
    # " line", "plane-2"),
    Example(chattify([rectangle_prompt, rectangle_response]),
    " Point", "rectangle-1"),
]


for example in examples:
    sentence, continuation, _ = example
    input_ids = model.tokenizer(sentence + continuation).input_ids
    tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
    print(tokens)
    with torch.inference_mode():
        logits = model(sentence)
        print_topk(model,logits)

    sentence, continuation, name = example
    graph = attribute(sentence, model, batch_size=32, max_feature_nodes=7500, 
                        offload='cpu', verbose=True)
    

    output_path = Path(f'attribution_graphs/{model_short_name}')
    output_path.mkdir(exist_ok=True, parents=True)
    output_path = output_path / f'{name}.pt'

    graph.to_pt(output_path)

    slug = f"{model_short_name}-{name}"

    upload_graph_to_s3(output_path, slug, node_threshold=0.85, edge_threshold=0.9)
    print(f"Graph now available at http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug={slug}")
    del graph
# %%
