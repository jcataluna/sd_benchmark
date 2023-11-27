
import random
from datasets import load_dataset

dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="test")

for i in range(5):
    random_index = random.randint(0, len(dataset) - 1)
    print(dataset[random_index]["Prompt"])
