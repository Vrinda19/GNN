from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
import numpy as np

class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        layers = []
        layer_sizes = [n_inputs] + n_hidden
        for idx in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[idx-1], layer_sizes[idx]),
                        nn.ReLU()]
      
        layers += [nn.Linear(layer_sizes[-1], n_outputs)]
        self.netLayers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
            x: input to the network
        Returns:
            out: outputs of the network
        """

        out = x
        for value in self.netLayers:
          out = value(out)


        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device



class GNN(nn.Module):
    """implements a graphical neural network in pytorch. In particular, we will use pytorch geometric's nn_conv module so we can apply a neural network to the edges.
    """

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        n_hidden: int,
        n_output: int,
        num_convolution_blocks: int,
    ) -> None:
        
        # n_node_features = x


        super().__init__()
        in_channels, out_channels2 = n_node_features, n_hidden

        self.embed = nn.Sequential(
                      nn.Linear(in_channels, n_hidden))
        self.gnn_layers = []
        #gnn_layers += [torch.nn.Embedding(64,64)]
        for idx in range(1,(num_convolution_blocks+1)):
          #print(idx)
          inc = n_hidden

          self.gnn_layers += [nn.ReLU(inplace=True), 
                        geom_nn.RGCNConv(in_channels = inc, 
                                        out_channels = n_hidden, 
                                        num_relations = n_edge_features),
                        nn.ReLU(inplace=True), 
                        geom_nn.MFConv(in_channels=n_hidden, 
                             out_channels=n_hidden,)
                        ]
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.head = nn.Sequential(
                      nn.Linear(n_hidden, n_hidden), 
                      nn.ReLU(inplace=True), 
                      nn.Linear(n_hidden, n_output))
        

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        #convert to int from one hot 
        edge_attr = edge_attr.argmax(-1)

        x = self.embed(x)
        for i, layer in enumerate(self.gnn_layers):
            if 'RGCN' in str(layer):
                x = layer(x, edge_index,  edge_attr)
            elif 'MFC' in str(layer):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        x = geom_nn.global_add_pool(x, batch)
        x = self.head(x)
        
        return x

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return next(self.parameters()).device
