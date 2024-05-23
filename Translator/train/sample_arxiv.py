import random
from tqdm import tqdm


file_path = "/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/arxiv_test.csv"
output_path = "/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/arxiv_sample_test.csv"

if __name__ == "__main__":
    random.seed(2023)
    line_list = []
    with open(file_path, "r") as f:
        for item in f.readlines():
            line_list.append(item)
    num_sample = 500
    selected_index = random.sample(range(1, len(line_list)), num_sample)
    selected_index = [0] + selected_index
    selected_data = [line_list[x] for x in selected_index]
    
    with open(output_path, "w") as f:
        for line in tqdm(selected_data):
            f.write(line)