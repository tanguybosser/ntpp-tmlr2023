import abc

import torch as th
import torch.nn as nn

from typing import Optional, Tuple

from tpp.utils.events import Events

from tpp.models.encoders.base.encoder import Encoder

from tpp.pytorch.models import LAYER_CLASSES, MLP

from tpp.utils.history import get_prev_times
from tpp.utils.encoding import SinusoidalEncoding, event_encoder, encoding_size


class VariableHistoryEncoder(Encoder, abc.ABC):
    """Variable history encoder. Here, the size H depends on the encoding type.
       It can be either 1, emb_dim or emb_dim+1.

    Args:
        name: The name of the encoder class.
        output_size: The output size (dimensionality) of the representations
            formed by the encoder.
        emb_dim: Size of the embeddings. Defaults to 1.
        embedding_constraint: Constraint on the weights. Either `None`,
            'nonneg' or 'softplus'. Defaults to `None`.
        temporal_scaling: Scaling parameter for temporal encoding
        encoding: Way to encode the events: either times_only, marks_only,
                  concatenate or temporal_encoding. Defaults to times_only
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            name: str,
            output_size: int,
            emb_dim: Optional[int] = 1,
            embedding_constraint: Optional[str] = None,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            **kwargs):
        super(VariableHistoryEncoder, self).__init__(
            name=name, output_size=output_size, marks=marks, **kwargs)
        self.emb_dim = emb_dim
        self.encoding = encoding
        self.time_encoding = time_encoding
        self.embedding_constraint = embedding_constraint
        self.encoding_size = encoding_size(
            encoding=self.encoding, emb_dim=self.emb_dim)

        self.embedding = None
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

    
