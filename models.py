from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from utils import create_norm
from gat import GAT


class Pretwins(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            concat_hidden: bool = False,
    ):
        super(Pretwins, self).__init__()

        self._encoder_type = encoder_type
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

    def forward(self, g, x):
        # ---- attribute reconstruction ----
        g1 = self.mask_attr_prediction(g, x)
        return g1

    def mask_attr_prediction(self, use_g, use_x):
        #enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        enc_rep= self.encoder(use_g, use_x)#, return_hidden=True)
        #if self._concat_hidden:
        #    enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

class twins_process(nn.Module):

    def __init__(self,args):
        super(twins_process, self).__init__()
        self.alpha = 1.0
        num_heads = args.num_heads
        num_out_heads = args.num_out_heads
        num_hidden = args.num_hidden
        num_layers = args.num_layers
        residual = args.residual
        attn_drop = args.attn_drop
        in_drop = args.in_drop
        norm = args.norm
        negative_slope = args.negative_slope
        encoder_type = args.encoder
        decoder_type = args.decoder
        drop_edge_rate = args.drop_edge_rate
        activation = args.activation
        alpha_l = args.alpha_l
        concat_hidden = args.concat_hidden
        num_features = args.num_features
        #n_clusters = args.n_cluster
        self.ae = Pretwins(
            in_dim=num_features,
            num_hidden=num_hidden,
            num_layers=num_layers,
            nhead=num_heads,
            nhead_out=num_out_heads,
            activation=activation,
            feat_drop=in_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            norm=norm,
            concat_hidden=concat_hidden)

    def forward(self, graph, x):

        g1 = self.ae(graph, x)

        return g1






def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError

    return mod


class AE_Model(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            concat_hidden: bool = False,
    ):
        super(AE_Model, self).__init__()

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._mask_rate= mask_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,

        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)


    def encoding_mask_noise(self, g, x, mask_rate):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)

        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, g, x):
        # ---- attribute reconstruction ----
        g1,x_init,x_rec  = self.mask_attr_prediction(g, x)
        return g1
    def mask_attr_prediction(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g
        enc_rep = self.encoder(use_g, use_x)

        rep = self.encoder_to_decoder(enc_rep)


        rep[mask_nodes] = 0

        recon = self.decoder(pre_use_g, rep)

        return recon

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

import numpy as np
import dgl
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx
def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng

class AE_process(nn.Module):

    def __init__(self,args):
        super(AE_process, self).__init__()
        self.alpha = 1.0
        num_heads = args.num_heads
        num_out_heads = args.num_out_heads
        num_hidden = args.num_hidden
        num_layers = args.num_layers
        residual = args.residual
        attn_drop = args.attn_drop
        in_drop = args.in_drop
        norm = args.norm
        negative_slope = args.negative_slope
        encoder_type = args.encoder
        decoder_type = args.decoder
        drop_edge_rate = args.drop_edge_rate
        activation = args.activation
        alpha_l = args.alpha_l
        concat_hidden = args.concat_hidden
        num_features = args.num_features
        mask_rate = args.mask_rate
        self.ae =AE_Model(
            in_dim=num_features,
            num_hidden=num_hidden,
            num_layers=num_layers,
            nhead=num_heads,
            nhead_out=num_out_heads,
            activation=activation,
            feat_drop=in_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            norm=norm,
            mask_rate=mask_rate,
            drop_edge_rate=drop_edge_rate,
            alpha_l=alpha_l,
            concat_hidden=concat_hidden)


    def forward(self, graph, x):

        g1 = self.ae(graph, x)

        return g1