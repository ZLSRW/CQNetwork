# CQNetwork: consensuality quantification network for interpretable identification of RNA N6-methyladenosine modification sites

**Abstract:** N6-methyladenosine (m6A) is one of the most prevalent internal RNA modifications in eukaryotes, significantly influencing diverse biological processes. The computational identification of m6A typically hinges on consensus knowledge, such as motifs, present within RNA sequences. However, identifying consensus patterns is challenging, as existing computational models struggle to simultaneously capture inherent differences in RNA sequence features, spatial conformations, and physicochemical environments. Here, we introduce CQNetwork, which integrates these prior differences into an end-to-end network specifically designed for consensus pattern quantification, thereby enabling interpretable and precise identification of RNA m6A. Specifically, CQNetwork first constructs a region graph to capture the connectivity between potential key regions of input RNA sequences by integrating multi-view prior knowledge related to consensus regions and nucleotides. A distance-based consensuality-aware strategy is then designed to establish the associations between region graphs, generating a consensus graph that evaluates differences in consensus patterns between regions through its essential topological structures. Under the quantification constraint, CQNetwork employs graph wavelet convolution to align the consensus graph with sequence information, learning high-quality consensus embeddings, and utilizes a multi-layer perceptron for accurate identification of m6A modification sites. Extensive experimental results validate that CQNetwork demonstrates robust performance in identifying novel m6A modification sites, underscoring its accuracy and interpretability.

## Requirements

```
Python 3.8.20
PyTorch = 1.8.0
numpy = 1.19.5
torch-geometric = 2.0.4
transformers = 4.30.2
tqdm = 4.66.2
pandas = 1.3.5
scikit-learn = 1.0.2
torchvision = 0.9.0
matplotlib = 3.5.3
h5py = 3.8.0
scipy = 1.7.3
seaborn = 0.11.2
```

## Data Preprocess

The raw data and preprocessing process can be found in the *Pre-Encoding* folder, which contains three subfolders: *data*, *Dataset*, and *Preprocess* (Here, we present the Rat_liver dataset as an example.):

- `Dataset`: The processed data, including primary and secondary structure information.

  -  `benchmark`:  Original data of the model.
  - ` independent`:Original independent data of the model.
  - `Tissue Specific`: The folder for storing the processed data.

- `Preprocess`: Contains four Python files. Executing them in sequence will generate the input data for model training and validation.

  - `0.Structure-Graph`: Sequence processing script;

  - `1.feature_embedding`: ELMo feature generation script;

  - `2.Train-Test-Division`: Training set - test set division script;

  - `3.IntegrationDatasets`: Data integration script.

  - running:

    ```
    python 0.Structure-Graph.py
    python 1.feature_embedding.py
    python 2.Train-Test-Division.py
    python 3.IntegrationDatasets.py
    ```

## data_loader

- `SiteBinding_dataloader1.py`: Execute the file to read the training and testing dataset files.

## model

- Our model can be found in the `models`
  - `ConsensusComponents.py`: Implements components for motif recognition, graph pooling, and consensus graph construction in the CQNetwork.
  - `ConsensusNetwork.py`: Defines the CQNetwork model that integrates region and consensus graphs for m6A modification site identification.
  - `Graph_Wavelet.py`: Implements Graph Wavelet Neural Network layers for processing graph-based features using wavelet convolutions.
  - `SemanticNet.py`: Builds a semantic network using convolutional and LSTM layers to extract and process RNA sequence features.
  - `handler.py`: Contains utility functions for saving/loading models, training, and validation within the CQNetwork framework.
  - `configure.py`: Configuration file for some settings.
  - `Utils.py`: Utility file.

- Run the following command to complete the evaluation of the dataset.

  ```
  	python main.py
  ```

  

