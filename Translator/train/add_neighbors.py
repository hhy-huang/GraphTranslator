import random
import csv
import ast
from tqdm import tqdm


file_path = "/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/summary_embeddings.csv"
sample_neighbor_path = "/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/sample_neighbor_df.csv"
output_path = "/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/summary_neightbors_embeddings.csv"


def build_dict_from_csv(csv_file):
    result_dict = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader):
            src_node = int(row['src_node'])
            dst_node = int(row['dst_node'])

            if src_node in result_dict:
                result_dict[src_node].append(dst_node)
            else:
                result_dict[src_node] = [dst_node]
    return result_dict

def add_neighbor_ids(csv_file, dict_data, output_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    # 添加新列名
    rows[0].append('neighbor_ids')
    for row in rows[1:]:
        node_id = int(row[0])
        neighbor_list = dict_data.get(node_id, [])
        # neighbor_ids = ','.join(neighbor_list)
        row.append(neighbor_list)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)



if __name__ == "__main__":
    random.seed(2024)
    line_list = []
    with open(file_path, "r") as f:
        for item in f.readlines():
            line_list.append(item)

    result_dict = build_dict_from_csv(sample_neighbor_path)
    add_neighbor_ids(file_path, result_dict, output_path)