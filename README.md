# Seq2SeqLearning
## Introduction
A fork from <a href="https://github.com/faneshion/Matchzoo">faneshion/Matchzoo</a>, add some new model in Question Answering Field. We rename the repository to show our focus. I have done lots of experiments and modification on this toolkit, please refer to this <a href="https://github.com/SongRb/Seq2SeqLearning/blob/master/docs/cntnExp.pdf">report</a> for detailed result explanation.

## Environment
* Python 2.7+
* Tensorflow 1.4.1
* Keras 2.06+
* nltk 3.2.2+
* tqdm 4.19.4+
* h5py 2.7.1+


## Overview
### Data Preparation
First please configure `nltk` resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

The data preparation module aims to convert dataset of different text matching tasks into a unified format as the input of deep matching models. Users provide datasets which contains pairs of texts along with their labels, and the module produces the following files.

+	**Word Dictionary**: records the mapping from each word to a unique identifier called *wid*. Words that are too frequent (e.g. stopwords), too rare or noisy (e.g. fax numbers) can be  filtered out by predefined rules.
+	**Corpus File**: records the mapping from each text to a unique identifier called *tid*, along with a sequence of word identifiers contained in that text. Note here each text is truncated or padded to a fixed length customized by users.
+	**Relation File**: is used to store the relationship between two texts, each line containing a pair of *tids* and the corresponding label.
+   **Detailed Input Data Format**: a detailed explaination of input data format can be found in `seq2seq/data/example/readme.md`.

Please run `bash run_data.sh` in directory `./data/WikiQA/` to get and format dataset.


### Model Construction
In the model construction module, we employ Keras library to help users build the deep matching model layer by layer conveniently. The Keras libarary provides a set of common layers widely used in neural models, such as convolutional layer, pooling layer, dense layer and so on. To further facilitate the construction of deep text matching models, we extend the Keras library to provide some layer interfaces specifically designed for text matching.

Moreover, the toolkit has implemented two schools of representative deep text matching models, namely representation-focused models and interaction-focused models [[Guo et al.]](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf).

### Training and Evaluation
For learning the deep matching models, the toolkit provides a variety of objective functions for regression, classification and ranking. For example, the ranking-related objective functions include several well-known pointwise, pairwise and listwise losses. It is flexible for users to pick up different objective functions in the training phase for optimization. Once a model has been trained, the toolkit could be used to produce a matching score, predict a matching label, or rank target texts (e.g., a document) against an input text.

## Benchmark Results:
Here, we adopt <a href="https://www.microsoft.com/en-us/download/details.aspx?id=52419">WikiQA</a> dataset for an example to inllustrate the usage of seq2seq. WikiQA is a popular benchmark dataset for answer sentence selection in question answering. We have provided <a href="./data/WikiQA/run_data.sh">a script</a> to download the dataset, and prepared it into the seq2seq data format. In the <a href="">models directory</a>, there are a number of configurations about each model for WikiQA dataset. 

Take the DRMM as an example. In training phase, you can run
```
python seq2seq/main.py --phase train --model_file examples/wikiqa/config/drmm_wikiqa.config
```
In testing phase, you can run
```
python seq2seq/main.py --phase predict --model_file examples/wikiqa/config/drmm_wikiqa.config
```

Here is my expriment result(Keep updating)  
Experiment on normal QA pair:

|     Model    |  NDCG@3  |  NDCG@5  |    MAP   |
|:------------:|:--------:|:--------:|:--------:|
|      WEC     | 0.485437 | 0.546745 | 0.508369 |
|    ARC-II    | 0.610188 |  0.66360 | 0.620367 |
|     CNTN     | 0.557843 | 0.633769 | 0.577380 |
|    MV-LSTM   | 0.613337 | 0.677977 | 0.632837 |
| Gated ARC-II | 0.602453 | 0.677129 | 0.622159 |

Experiment on short-long QA pair:

|     Model    |  NDCG@3  |  NDCG@5  | MAP      |
|:------------:|:--------:|:--------:|:--------:|
|      WEC     | 0.834926 | 0.852815 | 0.810173 |
|    ARC-II    | 0.496673 | 0.564462 | 0.507174 |
|     CNTN     | 0.383719 | 0.477619 | 0.437992 |
|    MV-LSTM   | 0.442138 | 0.543960 | 0.488529 |
| Gated ARC-II | 0.405606 | 0.488695 | 0.464356 |

## Model Detail:
1. DRMM:
    <a href="http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf">A Deep Relevance Matching Model for Ad-hoc Retrieval</a>.

- model file: `models/drmm.py`
- model config: `models/drmm_wikiqa.config`

2. MatchPyramid:
    <a href="https://arxiv.org/abs/1602.06359"> Text Matching as Image Recognition</a>

- model file: `models/matchpyramid.py`
- model config: `models/matchpyramid_wikiqa.config`

---
3. ARC-I:
    <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

- model file: `models/arci.py`
- model config: `models/arci_wikiqa.config`

---
4. DSSM:
    <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf">Learning Deep Structured Semantic Models for Web Search using Clickthrough Data</a>

- model file: `models/dssm.py`
- model config: `models/dssm_wikiqa.config`

---
5. CDSSM:
    <a href="https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/">Learning Semantic Representations Using Convolutional Neural Networks for Web Search</a>

- model file: `models/cdssm.py`
- model config: `models/cdssm_wikiqa.config`

---
6. ARC-II:
    <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

- model file: `models/arcii.py`
- model config: `models/arcii_wikiqa.config`

---
7. MV-LSTM:
    <a href="https://arxiv.org/abs/1511.08277">A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations</a>

- model file: `models/mvlstm.py`
- model config: `models/mvlstm_wikiqa.config`

-------
8. aNMM:
    <a href="http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1240">aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model</a>
- model file: `models/anmm.py`
- model config: `models/anmm_wikiqa.config`

-------
9. DUET:
    <a href="https://dl.acm.org/citation.cfm?id=3052579">Learning to Match Using Local and Distributed Representations of Text for Web Search</a>

- model file: `models/duet.py`
- model config: `models/duet_wikiqa.config`

---
10. K-NRM:
    <a href="https://arxiv.org/abs/1706.06613">End-to-End Neural Ad-hoc Ranking with Kernel Pooling</a>

- model file: `models/knrm.py`
- model config: `models/knrm_wikiqa.config`

---
11. CNTN:
<a href="https://www.ijcai.org/Proceedings/15/Papers/188.pdf">
Convolutional Neural Tensor Network Architecture for Community-Based Question Answering
</a>

- model file: `models/cntn.py`
- model config: `models/cntn_wikiqa.config`

---
12. WEC:
<a href="https://arxiv.org/abs/1511.04646">
Word Embedding Based Correlation Model for Question/Answer Matching
</a>

- model file: `models/wec.py`
- model config: `models/wec_wikiqa.config`

---
13. (Experiment) GatedARC-II:
Modification on ARC-II with Gated Linear Unit

- model file: `models/gated_arcii.py`
- model config: `models/gated_arcii_wikiqa.config`

---





