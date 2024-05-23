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


class MakeVisionGraphData(object):
    def __init__(self):
        # nx
        self.with_label = True
        self.node_size = 200
        self.font_size = 10
        self.layout_seed = 2024
        self.font_color = "white"
        self.colors_list = ['blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'grey',
                   'lime', 'navy', 'gold', 'maroon', 'turquoise', 'olive', 'indigo', 'lightcoral',
                   'darkgreen', 'chocolate', 'magenta', 'lightseagreen', 'black']

        # data
        self.gt_label_all_path = '/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv_raw/raw/node-label.csv'
        self.node2label = self.get_gt_lable_all()


    def get_gt_lable_all(self):
        """
        return node2label dict
        """
        node2label = {}
        with open(self.gt_label_all_path, 'r') as file:
            lines = file.readlines()
            for i, row in tqdm(enumerate(lines)):
                label = int(row.strip('\n'))
                node2label[i] = label
        return node2label

    def build_dict_from_csv(self, csv_file):
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
    
    def set_colors(self, center_node, nx_graph):
        """
        Set colors for nx graph plot
        """
        node_colors = []
        color_mapping = {}
        current_color_index = 0
        for node in nx_graph.nodes():
            label = self.node2label[node]
            if node == center_node:
                node_colors.append('red')
            else:
                if label not in color_mapping:
                    color_mapping[label] = self.colors_list[current_color_index]
                    current_color_index += 1  # Move to the next color for a new label
                node_colors.append(color_mapping[label])
        return node_colors

    def build_graph_img(self, node2neighbor_dict, arxiv_all_file):
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
            # set color
            node_colors = self.set_colors(center_node=center_node, nx_graph=nx_graph)
            # set label
            display_labels = {}
            for node in nx_graph.nodes():
                if node == center_node:
                    display_labels[node] = '?'
                else:
                    display_labels[node] = self.node2label[node]
            # plot
            plt.figure(figsize=(3, 3))
            pos = nx.spring_layout(nx_graph, seed=self.layout_seed)
            nx.draw(nx_graph, pos,\
                 with_labels=self.with_label,
                 labels=display_labels, 
                 node_size=self.node_size, 
                 font_size=self.font_size, 
                 node_color=node_colors,
                 font_color=self.font_color)
            plt.savefig(f'/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/img/{center_node}.png')

        logging.info("*" * 30)
        logging.info("Finished.")

    def add_img_idx(self, target_file, dst_file):
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


if __name__ == "__main__": 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )   
    MK_VG = MakeVisionGraphData()
    node2neighbor_dict = MK_VG.build_dict_from_csv(sample_neighbor_path)
    MK_VG.build_graph_img(node2neighbor_dict, arxiv_all_file)
