import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from typing import List, Optional

from tpp.models.encoders.base.fixed_history import FixedHistoryEncoder
from tpp.pytorch.models import MLP
from tpp.utils.events import Events
from tpp.utils.history import build_histories

class RecurrentFixedEncoder(FixedHistoryEncoder):
    """Abstract classes for recurrent encoders operating on a fixed size window.

    Args:
        name: The name of the encoder class.
        rnn: RNN encoder function.
        units_mlp: List of hidden layers sizes for MLP.
        activations: MLP activation functions. Either a list or a string.
        emb_dim: Size of the embeddings. Defaults to 1.
        embedding_constraint: Constraint on the weights. Either `None`,
            'nonneg' or 'softplus'. Defaults to `None`.
        temporal_scaling: Scaling parameter for temporal encoding
        padding_id: Id of the padding. Defaults to -1.
        encoding: Way to encode the events: either times_only, marks_only,
                  concatenate or temporal_encoding. Defaults to times_only
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            name: str,
            rnn: nn.Module,
            # MLP args
            units_mlp: List[int],
            activation_mlp: Optional[str] = "relu",
            dropout_mlp: Optional[float] = 0.,
            constraint_mlp: Optional[str] = None,
            activation_final_mlp: Optional[str] = None,
            # Other args
            emb_dim: Optional[int] = 1,
            embedding_constraint: Optional[str] = None,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            history_size: Optional[int] = 2,
            **kwargs):
        super(RecurrentFixedEncoder, self).__init__(
            name=name,
            output_size=units_mlp[-1],
            history_size=history_size,
            emb_dim=emb_dim,
            embedding_constraint=embedding_constraint,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.rnn = rnn
        self.mlp = MLP(
            units=units_mlp,
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            input_shape=self.rnn.hidden_size,
            activation_final=activation_final_mlp)  

    def forward(self, events:Events):
        encoded_history, history_mask, prev_events_idxs = self.get_encoded_history_representations(events=events) #[B,1+L, H, D]
        
        seq_lens = th.sum(prev_events_idxs >= 0, dim=-1) #[B,1+L]
        seq_lens = th.flatten(seq_lens) #[B*(1+L)]
        seq_lens[seq_lens == 0] = 1 #Pack sequences does not allow 0 length sequences. 
        b,l,h,d = encoded_history.shape
        
        encoded_history_3d = encoded_history.view(b*l, h, d) #[B*(1+L), H, D]
        packed_history = pack_padded_sequence(encoded_history_3d, lengths=seq_lens.to('cpu'), batch_first=True, enforce_sorted=False)

        _,output = self.rnn(packed_history)

        
        hidden = output.squeeze() # [B*(1+L), D_hidden]
        hidden = hidden.view(b, l, self.hidden_size) #[B,1+L,D_hidden]
        hidden = F.normalize(hidden, dim=-1, p=2)
        representations = self.mlp(hidden) #
        
        return (representations, history_mask, dict())
