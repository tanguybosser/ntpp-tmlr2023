import torch as th

from typing import List, Optional, Tuple, Dict

from tpp.models.decoders.base.cumulative import CumulativeDecoder
from tpp.models.base.process import Events

from tpp.pytorch.models import MLP

from tpp.utils.encoding import encoding_size
from tpp.utils.index import take_2_by_2, take_3_by_2


class ConditionalPoissonCMDecoder(CumulativeDecoder):
    """A mlp decoder based on the cumulative approach.

    Args:
        units_mlp: List of hidden layers sizes, including the output size.
        activation_mlp: Activation functions. Either a list or a string.
        constraint_mlp: Constraint of the network. Either none, nonneg or
            softplus.
        dropout_mlp: Dropout rates, either a list or a float.
        activation_final_mlp: Last activation of the MLP.

        mc_prop_est: Proportion of numbers of samples for the MC method,
                     compared to the size of the input. (Default=1.).
        do_zero_subtraction: If `True` the class computes
            Lambda(tau) = Lambda'(tau) - Lambda'(0)
            in order to enforce Lambda(0) = 0. Defaults to `True`.
        emb_dim: Size of the embeddings (default=2).
        encoding: Way to encode the events: either times_only, or temporal.
            Defaults to times_only.
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            # MLP
            units_mlp: List[int],
            activation_mlp: Optional[str] = "relu",
            dropout_mlp: Optional[float] = 0.,
            constraint_mlp: Optional[str] = "nonneg",
            activation_final_mlp: Optional[str] = "parametric_softplus",
            # Other params
            do_zero_subtraction: Optional[bool] = True,
            model_log_cm: Optional[bool] = False,
            emb_dim: Optional[int] = 2,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            **kwargs):

        if constraint_mlp is None:
            print("Warning! MLP decoder is unconstrained. Setting to `nonneg`")
            constraint_mlp = "nonneg"

        input_size = units_mlp[0]
        super(ConditionalPoissonCMDecoder, self).__init__(
            name="mlp-cm",
            do_zero_subtraction=do_zero_subtraction,
            model_log_cm=model_log_cm,
            input_size=input_size,
            emb_dim=emb_dim,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.mlp = MLP(
            units=units_mlp[1:],
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            input_shape=input_size,
            activation_final=activation_final_mlp)

    def cum_intensity(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None,
            update_running_stats: Optional[bool] = True
    ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the cumulative log intensity and a mask

        Args:
            events: [B,L] Times and labels of events.
            query: [B,T] Times to evaluate the intensity function.
            prev_times: [B,T] Times of events directly preceding queries.
            prev_times_idxs: [B,T] Indexes of times of events directly
                preceding queries. These indexes are of window-prepended
                events.
            pos_delta_mask: [B,T] A mask indicating if the time difference
                `query - prev_times` is strictly positive.
            is_event: [B,T] A mask indicating whether the time given by
                `prev_times_idxs` corresponds to an event or not (a 1 indicates
                an event and a 0 indicates a window boundary).
            representations: [B,L+1,D] Representations of each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined. If `None`, there is no mask. Defaults to
                `None`.
            artifacts: A dictionary of whatever else you might want to return.
            update_running_stats: whether running stats are updated or not.

        Returns:
            intensity_integral: [B,T,M] The cumulative intensities for each
                query time for each mark (class).
            intensities_mask: [B,T]   Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: Some measures.
        """
        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)                   # [B,T,D]

        intensity_itg = self.mlp(history_representations)   # [B,T,output_size]

        delta_t = query - prev_times  # [B,T]
        delta_t = delta_t.unsqueeze(dim=-1)  # [B,T,1]
        intensity_integrals = intensity_itg * delta_t  # [B,T,M]

        intensity_mask = pos_delta_mask  # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)  # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        return intensity_integrals, intensity_mask, artifacts
