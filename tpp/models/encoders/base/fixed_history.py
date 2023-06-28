import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple

from tpp.utils.events import Events
from tpp.models.encoders.base.encoder import Encoder
from tpp.utils.history import build_histories
from tpp.utils.index import take_3_by_3

from tpp.pytorch.models import LAYER_CLASSES, MLP


from tpp.utils.encoding import SinusoidalEncoding, event_encoder, encoding_size


class FixedHistoryEncoder(Encoder):
    """A parametric encoder process with a fixed history size representation.

    Args
        name: The name of the encoder class.
        net: The network used to encode the history.
        history_size: The size of each history.
        output_size: The output size (dimensionality) of the representations
            formed by the encoder.
        marks: The distinct number of marks (classes) for the process.
            Defaults to 1.
    """
    def __init__(
            self,
            name: str,
            output_size: int,
            emb_dim: Optional[int] = 1,
            embedding_constraint: Optional[str] = None,
            encoding: Optional[str] = 'times_only',
            temporal_scaling: Optional[float] = 1.,
            time_encoding: Optional[str] = 'relative',
            history_size: Optional[int] = 2,
            marks: Optional[int] = 1,
            **kwargs):
        super(FixedHistoryEncoder, self).__init__(
            name=name, output_size=output_size, marks=marks, **kwargs)
        self.emb_dim = emb_dim
        self.embedding_constraint = embedding_constraint
        self.encoding = encoding
        self.time_encoding = time_encoding
        self.temporal_scaling = temporal_scaling
        self.history_size = history_size
        self.embedding = None
        self.encoding_size = encoding_size(
            encoding=self.encoding, emb_dim=self.emb_dim)
        if encoding in ["marks_only", "concatenate", "temporal_with_labels",
                        "learnable_with_labels", "log_concatenate"]:
            embedding_layer_class = nn.Linear
            if self.embedding_constraint is not None:
                embedding_layer_class = LAYER_CLASSES[
                    self.embedding_constraint]
            self.embedding = embedding_layer_class(
                in_features=self.marks, out_features=self.emb_dim, bias=False)

        self.temporal_enc = None
        if encoding in ["temporal", "temporal_with_labels"]:
            self.temporal_enc = SinusoidalEncoding(
                emb_dim=self.emb_dim, scaling=temporal_scaling)
        elif encoding in ["learnable", "learnable_with_labels"]:
            self.temporal_enc = MLP(
                units=[self.emb_dim],
                activations=None,
                constraint=self.embedding_constraint,
                dropout_rates=0,
                input_shape=1,
                activation_final=None)

    def get_encoded_history_representations(    
            self, events:Events):
        event_representations, events_mask = self.get_events_representations(events=events) #[B,1+L,D], [B,1+L]
        
        query = events.get_times(prepend_window=True) #[B,1+L]
        history , history_mask, prev_events_idx = build_histories(query=query, events=events, history_size=self.history_size, 
        allow_partial_history=True, neg_index_last=True, 
        aligned=True, allow_window=True, return_history=False) #[B,1+L,H]
        
        nonneg_prev_events_idx = prev_events_idx.clone() 
        nonneg_prev_events_idx[prev_events_idx < 0] = 0 #[B,L,H,D]
        
        encoded_history = take_3_by_3(event_representations, nonneg_prev_events_idx) #[B,L,H,D]

        

        return encoded_history, history_mask, prev_events_idx

        
    
    
    def get_history_representations(
            self, events: Events) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the history vectors.

        Args:
            events: [B,L] Times and labels of events.

        Returns:
            histories: [B,L+1,H] Histories of each event.
            histories_mask: [B,L+1] Mask indicating which histories
                are well-defined.

        """
        histories = events.times.unsqueeze(dim=-1)  # [B,L,1]
        histories_mask = events.mask  # [B,L]
        batch_size, _ = histories_mask.shape

        if self.history_size > 1:
            h_prev, h_prev_mask = build_histories(
                query=events.times, events=events,
                history_size=self.history_size - 1)          # [B,L,H-1], [B,L]
            histories = th.cat([h_prev, histories], dim=-1)  # [B,L,H] 
            histories_mask = histories_mask * h_prev_mask    # [B,L]

        window_history = th.zeros(
            [batch_size, 1, self.history_size],
            dtype=histories.dtype,
            device=histories.device)
        histories = th.cat([window_history, histories], dim=1)  # [B,L+1,H]
        window_mask = th.zeros(
            [batch_size, 1],
            dtype=histories_mask.dtype,
            device=histories.device)
        histories_mask = th.cat(
            [window_mask, histories_mask], dim=1)               # [B,L+1]

        return histories, histories_mask                   # [B,L+1,H], [B,L+1]

    
