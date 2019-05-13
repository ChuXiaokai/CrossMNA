# CrossMNA
This repo contains the source code and datasets of WWW' 19 paper: Cross-Network Embedding for Multi-Network Alignment.

## Datasets
There are three datasets: ArXiv, SacchCere, and Twitter. The raw data can be found [here](https://comunelab.fbk.eu/data.php).

Two multi-network tasks in our work: multi-network alignment and link prediction.

To split the dataset into training set and test set for **network alignment**, you can use the method _split_dataset()_  in _node_matching/split_data.py_.
This will generate a special input data format for CrossMNA.
You can use _transfer()_ to transforms this data format to the input format as in method [IONE](https://github.com/ColaLL/IONE).


To split dataset for link prediction:

    >> python link_prediction/split_data.py

## Run
To train CrossMNA for network alignment, where _p_ denotes the training ratio:

    >> python main.py --task NetworkAlignment --dataset xxx --p xx

To generate multi-network embedding for intra-link prediction:

    >> python main.py --task LinkPrediction --dataset xxx --p xx


## Dependencies
* Python == 2.7
* Tensorflow >= 1.4

## Cite
If this code is helpful for you, please cite this paper:
_Xiaokai Chu, Xinxin Fan, Di Yao, Zhihua Zhu, Jianhui Huang, Jing- ping Bi. Cross-Network Embedding for Multi-Network Alignment. In Proceedings of the 2019 World Wide Web Conference (WWW ’19)._