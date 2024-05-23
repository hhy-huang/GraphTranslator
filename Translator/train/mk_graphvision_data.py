"""
Make Graph Datasets with networkx.
"""

import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.sparse as sp
import csv
import logging
import numpy as np

from networkx.algorithms import community
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from tqdm import tqdm


colors_dict = {0: "white", 1: "blue", 2: "red", 3: "green", 4: "purple", 5: "orange", 6:"yellow", 7: "pink", 8: "black", 9: "cyan", 10: "grey", 11: "olive"}
graph_data_all_path = '/data/ChenWei/HaoyuHuang/GraphGPT-base/graph_data/graph_data_all.pt'
stage_1_train_instruct_path = '/data/ChenWei/HaoyuHuang/GraphGPT-base/data/stage_1/train_instruct_graphmatch.json'

# source files
sample_neighbor_path = '/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/sample_neighbor_df.csv'
arxiv_all_file = '/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/arxiv_nodeidx2paperid.csv'

# to-be modified files, add one column (image id)
summary_neighbor_path = '/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/summary_neighbors_embeddings.csv'
arxiv_sample_test_path = '/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/arxiv_sample_test.csv'

# dst files
image_path = '/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/img'
summary_neighbor_image_path = '/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/summary_neighbors_image_embeddings.csv'
arxiv_sample_image_test_path = '/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/arxiv_sample_image_test.csv'


def build_dict_from_csv(csv_file):
    """
    Build node id to neighbors mapping dict.
    """
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

def build_graph_img(node2neighbor_dict, arxiv_all_file):
    """
    Build img graph data for each center node used in target file
    """
    logging.info("*" * 30)
    logging.info(f"Begin building {arxiv_all_file} image graph data...")

    center_node_list = []
    with open(arxiv_all_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader):
            center_node_id = int(row['node idx'])
            center_node_list.append(center_node_id)

    for center_node in tqdm(center_node_list):
        one_hop_neighbors = node2neighbor_dict[center_node]
        two_hop_neighbors = [node2neighbor_dict[x] for x in one_hop_neighbors]
        edge_index = [[], []]
        # build edge_index
        # one hop
        edge_index[0] += [center_node] * len(one_hop_neighbors)
        edge_index[1] += one_hop_neighbors
        # two hop
        for item in one_hop_neighbors:
            edge_index[0] += [item] * len(node2neighbor_dict[item])
            edge_index[1] += node2neighbor_dict[item]
        # build node_list
        node_list = [center_node] + one_hop_neighbors
        for item in one_hop_neighbors:
            node_list += node2neighbor_dict[item]
        node_list = list(set(node_list))

        # build pyg graph
        edge_index = torch.tensor(np.array(edge_index))
        node_list = torch.tensor(np.array(node_list))
        pyg_graph = Data(x=0, edge_index=edge_index, num_nodes=node_list.shape[0])

        # trans pyg graph to graph image
        nx_graph = to_networkx(pyg_graph, to_undirected=True)
        nx_graph = nx.ego_graph(nx_graph, center_node, radius=2, undirected=True)

        # save image
        node_colors = ['blue'] * (len(node_list.numpy().tolist()))
        center_tag = [index for index, value in enumerate(list(nx_graph.nodes)) if value == center_node][0]
        node_colors[center_tag] = 'red'
        plt.figure(figsize=(3, 3))
        pos = nx.spring_layout(nx_graph, seed=2024)
        nx.draw(nx_graph, pos, with_labels=True, node_size=200, font_size=10, node_color=node_colors)
        plt.savefig(f'/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/img/{center_node}.png')

    logging.info("*" * 30)
    logging.info("Finished.")

def add_img_idx(target_file, dst_file):
    """
    Add graph image idx to the target files
    """
    logging.info("*" * 30)
    logging.info(f"Begin writing {target_file} with image idx...")

    with open(target_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    # 添加新列名
    rows[0].append('image_id')

    for row in tqdm(rows[1:]):
        node_id = int(row[0])
        image_id = node_id
        row.append(image_id)

    with open(dst_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
    
    logging.info("Finished.")
    logging.info("*" * 30)

def load_ori_graph(path):
    graph_data_all = torch.load(path)
    return graph_data_all

def load_json(path):
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data

def edgeidx2adjacencydict(edge_index):
    """
    transfer edge_index to adjacency_dict
    """
    adjacency_dict = {}
    for i in range(edge_index.shape[1]):
        src_node = edge_index[0, i].item()
        dst_node = edge_index[1, i].item()
        adjacency_dict[i] = [src_node, dst_node]
    return adjacency_dict

def process_graph_instruction(instruct_list, cora_data):
    """
    fetch subgraph info from instruction data list
    """
    subgraph_adjacency_dict = {}                                    # {center_node0: {"adjacency_dict": .., "node_list": ..}, ...}
    for i, item in enumerate(instruct_list):
        if i >= 74075 and i <= 99194:
            edge_index = torch.tensor(item['graph']['edge_index'])      # id in subgraph
            node_list = torch.tensor(item['graph']['node_list'])
            sub_graph_pyg = Data(x=cora_data.x[item['graph']['node_list']], edge_index=edge_index)
            center_node = item['graph']['node_idx']

            # adjacency_dict = edgeidx2adjacencydict(edge_index)
            # subgraph_adjacency_dict[center_node] = {"adjacency_dict": adjacency_dict, "node_list": node_list, "edge_index": edge_index}
            subgraph_adjacency_dict[center_node] = {"node_list": node_list, "edge_index": edge_index, "sub_graph_pyg": sub_graph_pyg}
    return subgraph_adjacency_dict

def make_nx_subgraph(subgraph_adjacency_dict, cora_node_label):
    """
    make a dict of nx-subgraphs, which are from the same graph dataset
    category, center node red
    """
    subgraph_nx_dict = {}                           # {center_node0: {"nx_graph": .., "node_list": ..}, center_node1: {..}}
    for center_node, graph_info in zip(subgraph_adjacency_dict.keys(), subgraph_adjacency_dict.values()):
        node_list = graph_info['node_list']
        edge_index_list = graph_info['edge_index'].numpy().tolist()
        sub_graph_pyg = graph_info['sub_graph_pyg']
        # subgraph = nx.Graph()
        # subgraph.add_nodes_from(node_list.numpy().tolist())
        # subgraph.add_edges_from(list(zip(edge_index_list[0], edge_index_list[1])))
        sub_graph_nx = to_networkx(sub_graph_pyg, to_undirected=True)
        sub_graph = nx.ego_graph(sub_graph_nx, center_node, radius=1, undirected=True)
        node_label_dict = {x: cora_node_label[x] for x in node_list.numpy().tolist()}
        subgraph_nx_dict[center_node] = {"nx_graph": sub_graph, "node_list": node_list, "label_dict": node_label_dict}
    return subgraph_nx_dict

def draw_nx_subgraph(subgraph_nx_dict):
    """
    save the image of the subgraphs
    """
    for center_node, graph_info in tqdm(zip(subgraph_nx_dict.keys(), subgraph_nx_dict.values())):
        nx_subgraph = graph_info['nx_graph']
        node_list = graph_info['node_list']
        label_dict = graph_info['label_dict']

        node_colors = [cora_colors_dict[i] for i, x in enumerate(node_list.numpy().tolist())]
        plt.figure(figsize=(3, 3))
        pos = nx.spring_layout(nx_subgraph, seed=42)
        nx.draw(nx_subgraph, pos, node_size=200, font_size=10, node_color=node_colors)
        plt.savefig(f'./image_cora_dataset/{center_node}.png')



if __name__ == "__main__":    
    colors = list(mcolors.TABLEAU_COLORS.values())
    unique_colors = colors[:40]
    node2neighbor_dict = build_dict_from_csv(sample_neighbor_path)
    build_graph_img(node2neighbor_dict, arxiv_all_file)
    """
    exit(0)
    graph_data_all = load_ori_graph(graph_data_all_path)
    ## Cora
    cora_data = graph_data_all['cora']
    cora_edge_index = cora_data['edge_index']                                           # [2, num_edges]
    cora_node_label = cora_data['y'].numpy().tolist()                                   # [25120]                                            
    # full graph
    adjacency_dict = edgeidx2adjacencydict(cora_edge_index)
    G_cora = nx.Graph(adjacency_dict)
    # multi-subgraph
    stage_1_train_instruct = load_json(stage_1_train_instruct_path)                     # list 183198
    subgraph_adjacency_dict = process_graph_instruction(stage_1_train_instruct, cora_data)         # [subgraph0_info, subgraph1_info, ...]
    subgraph_nx_dict = make_nx_subgraph(subgraph_adjacency_dict, cora_node_label)                        # 
    # draw subgraph
    draw_nx_subgraph(subgraph_nx_dict)
    """