import torch as th
from tpp.models.encoders.base.encoder import Encoder

from typing import Dict, Optional, Tuple

from tpp.utils.encoding import encoding_size
from tpp.utils.events import Events


class ConstantEncoder(Encoder):
    """Encoder that passes a constant history representation to the decoder (i.e. making the decoder independent of the hsitory of the process).

    Args:
        emb_dim: Size of the embeddings. This becomes the constant history vector size. Defaults to 1.
    """
    def __init__(
            self,
            # Other args
            units_mlp,
            **kwargs):
        super(ConstantEncoder, self).__init__(
            name="constant",
            output_size=units_mlp[-1],
            **kwargs)

    def forward(self, events: Events) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Returns vectors of ones as history reprensentations.

        Args:
            events: [B,L] Times and labels of events.

        Returns:
            histories: [B,L+1,M+1] Representations of each event.
            histories_mask: [B,L+1] Mask indicating which representations
                are well-defined.

        """
        b, l = events.times.shape[0], events.times.shape[1]
        histories = th.ones(b, l+1, self.output_size).to(events.times.device) #[B,L+1, D]
        histories_mask = th.ones(b, l+1).to(events.times.device) #[B,L+1]

        
        return (histories, histories_mask, 
                dict())  # [B,L+1,D], [B,L+1], Dict
