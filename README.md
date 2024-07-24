# GraphTranslator-Implementation & Rethinking

The work is based on The Web Conference 2024 paper
[GraphTranslator: Aligning Graph Model to Large Language Model for Open-ended Tasks](https://arxiv.org/pdf/2402.07197.pdf)

Author: Mengmei Zhang, [Mingwei Sun](https://github.com/smw1996), [Peng Wang](https://github.com/PaulWongDlut), [Shen Fan](https://www.findshine.com), [Yanhu Mo](https://github.com/guyuisland), Xiaoxiao Xu, Hong Liu, Cheng Yang, Chuan Shi

My work concentrated on the ***implementation*** of the GraphTranslator and the ***improvement*** based on it.

### Result:

| Methods | Arxiv_Sample_200 |  |  | |
------|-----|-----|-----|-----
|  | Acc@1 | Acc@3 | Acc@5 | Legality Rate
| GraphTranslator | 8.67 | 14.29 | 22.96 | 98.94
| GraphTranslator_neighbors | **12.12** | **21.21** | **29.29** | **99.5**


## Original Model Pipeline

![image-20240129111934589](./figure/model.jpg)

- **Pre-training Graph Model Phase.**

- **Producer Phase.**

- **Translator Training Phase.(Modified)** 

  ​	*Stage 1*: Training the Translator for GraphModel-Text alignment.

  ​	*Stage 2*: Training the Translator for GraphModel-LLM alignment.

- **Translator Generate Phase.** Generate the predictions with the pre-trained Translator model.

### Installation

The experiment settings:

- NVIDIA A40
- CUDA Version: 11.4
- torch: 1.12.1+cu113
- torch-cluster             1.6.0+pt112cu113
- torch_geometric           2.5.3
- torch-scatter             2.1.0+pt112cu113
- torch-sparse              0.6.16+pt112cu113
- torch-spline-conv         1.2.1+pt112cu113

The `./requirements.txt` list all Python libraries that GraphTranslator depend on, and you can install using:

```
conda create -n graphtranslator python=3.9
conda activate graphtranslator
cd GraphTranslator/
pip install -r requirements.txt
```

### Datasets & Models

Download datasets and model checkpoints used in this project with huggingface.

**ArXiv Dataset**

Download files `bert_node_embeddings.pt`, `graphsage_node_embeddings.pt` and `titleabs.tsv` from [link](https://huggingface.co/datasets/Hualouz/GraphTranslator-arixv) and insert them to `./data/arxiv`.

```
cd ./data/arxiv
git lfs install
git clone git@hf.co:datasets/Hualouz/GraphTranslator-arxiv
```

**Translator Model**

Download `bert-base-uncased.zip` from [link](https://huggingface.co/Hualouz/Qformer/tree/main) and unzip it to `./Translator/models`.

```
cd Translator/models/
git lfs install
git clone git@hf.co:Hualouz/Qformer
unzip bert-base-uncased.zip
```

**ChatGLM2-6B Model**

Download the `ChatGLM2-6B` model from [link](https://huggingface.co/THUDM/chatglm2-6b) and insert it to `./Translator/models` 

```
cd ./Translator/models
git lfs install
git clone git@hf.co:THUDM/chatglm2-6b
```

### Run

#### Producer Phase

- Generate node summary text with LLM (ChatGLM2-6B).

```
cd ./Producer/inference
python producer.py
```

#### Training Phase

To achieve the modified performance, you need to set the parameters `self.use_neighbors = True` in the files under the folder `./Translator/models/translator_models`.

Train the Translator model with the prepared ArXiv dataset.

- Stage 1 Training

Train the Translator for GraphModel-Text alignment. The training configurations are in the file `./Translator/train/pretrain_arxiv_stage1.yaml`.

```
cd ./Translator/train
python train.py --cfg-path ./pretrain_arxiv_stage1.yaml
```

After stage 1, you will get a model checkpoint stored in `./Translator/model_output/pretrain_arxiv_stage1/checkpoint_0.pth`.

- Stage 2 Training

Train the Translator for GraphModel-LLM alignment. The training configurations are in the file `./Translator/train/pretrain_arxiv_stage2.yaml`.

GPU cost: chatglm-6b, 24000MiB

```
cd ./Translator/train
python train.py --cfg-path ./pretrain_arxiv_stage2.yaml
```

After stage 2, you will get a model checkpoint stored in `./Translator/model_output/pretrain_arxiv_stage2/checkpoint_0.pth`.

After all the training stages , you will get a model checkpoint that can translate GraphModel information into that the LLM can understand.

#### Generate Phase

- generate prediction with the pre-trained Translator model. The generate configurations are in the file `./Translator/train/pretrain_arxiv_generate_stage2.yaml`. As to the inference efficiency, I randomly selected 200 data points for the evaluation.

- GPU Cost: chatglm-6B, 14375MiB

```
cd ./Translator/train
python generate.py
```

- The generated prediction results will be saved in `./data/arxiv/pred_noNeighbors.txt` and `./data/arxiv/pred_with_neighbors.txt` 

#### Evaluation

Evaluate the accuracy of the generated predictions.

Don't forget to change the prediction file name.

```
cd ./Evaluate
python eval.py
```
