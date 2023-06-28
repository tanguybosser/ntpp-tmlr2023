import abc

import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple

from tpp.utils.events import Events
from tpp.utils.history import get_prev_times
from tpp.utils.encoding import SinusoidalEncoding, event_encoder, encoding_size

class Encoder(nn.Module, abc.ABC):
    """An encoder for a TPP.

    Args:
        name: The name of the encoder class.
        output_size: The output size (dimensionality) of the representations
            formed by the encoder.
        marks: The distinct number of marks (classes) for the process.
            Defaults to 1.

    """
    def __init__(
            self,
            name: str,
            output_size: int,
            marks: Optional[int] = 1,
            **kwargs):
        super(Encoder, self).__init__()
        self.name = name
        self.marks = marks
        self.output_size = output_size

    def get_events_representations(
            self, events: Events) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the history vectors.

        Args:
            events: [B,L] Times and labels of events.

        Returns:
            merged_embeddings: [B,L+1,emb_dim] Histories of each event.
            histories_mask: [B,L+1] Mask indicating which histories
                are well-defined.
        """
        times = events.get_times(prepend_window=True)      # [B,L+1]
        histories_mask = events.get_mask(prepend_window=True)  # [B,L+1]

        prev_times, is_event, pos_delta_mask = get_prev_times(
            query=times,
            events=events,
            allow_window=True)            # ([B,L+1],[B,L+1]), [B,L+1], [B,L+1]

        if self.time_encoding == "relative":
            prev_times, prev_times_idxs = prev_times  # [B,L+1], [B,L+1]
            times = times - prev_times

        histories_mask = histories_mask * pos_delta_mask

        labels = events.labels
        labels = th.cat(
            (th.zeros(
                size=(labels.shape[0], 1, labels.shape[-1]),
                dtype=labels.dtype, device=labels.device),
             labels), dim=1)  # [B,L+1,M]

        return event_encoder(
            times=times,
            mask=histories_mask,
            encoding=self.encoding,
            labels=labels,
            embedding_layer=self.embedding,
            temporal_enc=self.temporal_enc)
    
    
    @abc.abstractmethod
    def forward(self, events: Events) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the (query time independent) event representations.

        Args:
            events: [B,L] Times and labels of events.

        Returns:
            representations: [B,L+1,D] Representations of each event,
                including the window start.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined.
            artifacts: A dictionary of whatever else you might want to return.

        """
        pass
