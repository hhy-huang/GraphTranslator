"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np
import ast
from Translator.datasets.datasets.base_dataset import BatchIterableDataset


class ArxivTextPairDataset(BatchIterableDataset):
    def __init__(self, cfg, mode):
        super(ArxivTextPairDataset, self).__init__(cfg, mode)
        self.max_length = cfg.arxiv_processor.train.max_length
        self.vocab_size = cfg.arxiv_processor.train.vocab_size
        self.max_neighbour = 10

    def _train_data_parser(self, data):
        # 训练阶段使用
        user_id = data[0][0]
        embedding = np.array(data[0][1].split(','), dtype=np.float32)
        node_input = data[0][2]
        if len(node_input)>=1000:
            node_input=node_input[:1000]
        neighbour_input = data[0][3]
        title = data[0][4]

        if len(data[0]) == 7:
            neighbour_list = [user_id] + ast.literal_eval(data[0][6])
            length = len(neighbour_list)
            # padding
            if len(neighbour_list) <= self.max_neighbour:
                neighbour_list += (self.max_neighbour - len(neighbour_list)) * [-1]
            else:
                neighbour_list = neighbour_list[:self.max_neighbour]
            neighbour_list = np.array(neighbour_list)
        elif len(data[0]) == 6:
            # print(data[0][5])
            # neighbour_list = [user_id] + ast.literal_eval(data[0][5])
            neighbour_list = [user_id] + [data[0][5]]
            length = len(neighbour_list)
            # padding
            if len(neighbour_list) <= self.max_neighbour:
                neighbour_list += (self.max_neighbour - len(neighbour_list)) * [-1]
            else:
                neighbour_list = neighbour_list[:self.max_neighbour]
            neighbour_list = np.array(neighbour_list)
    
        text_input = 'The summary of this article is as follows:' + node_input+ '\nThere are some papers that cite this paper.' + neighbour_input

        return user_id, embedding, text_input, title, neighbour_list, length

    def __len__(self):
        return self.row_count
