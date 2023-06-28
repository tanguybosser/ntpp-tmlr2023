import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple, List

from tpp.models.decoders.base.variable_history import VariableHistoryDecoder
from tpp.utils.events import Events
from tpp.utils.index import take_3_by_2, take_2_by_2
from tpp.utils.stability import epsilon, subtract_exp, check_tensor

class RMTPPDecoder(VariableHistoryDecoder):
    """Analytic decoder process, uses a closed form for the intensity
    to train the model.
    See https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            units_mlp: List[int],
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            encoding: Optional[str] = "times_only",
            **kwargs):
        if encoding not in ["times_only", "log_times_only"]:
            raise ValueError("Wrong encoding for RMTPP decoder")
        super(RMTPPDecoder, self).__init__(
            name="rmtpp",
            input_size=units_mlp[0],
            encoding=encoding,
            marks=marks)
        self.w = nn.Parameter(th.Tensor(1))
        self.w_h = nn.Linear(self.input_size, marks)
        self.w_t  = nn.Linear(self.input_size, 1)
        self.multi_labels = multi_labels
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.w, b=0.001)

    def forward(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.LongTensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None,
            sampling: Optional[bool] = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Compute the intensities for each query time given event
        representations.

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
            representations: [B,L+1,D] Representations of window start and
                each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined. If `None`, there is no mask. Defaults to
                `None`.
            artifacts: A dictionary of whatever else you might want to return.

        Returns:
            log_intensity: [B,T,M] The intensities for each query time for
                each mark (class).
            intensity_integrals: [B,T,M] The integral of the intensity from
                the most recent event to the query time for each mark.
            intensities_mask: [B,T] Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: A dictionary of whatever else you might want to return.

        """
        
        (query_representations,
         intensity_mask) = self.get_query_representations(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            pos_delta_mask=pos_delta_mask,
            is_event=is_event,
            representations=representations, 
            representations_mask=representations_mask)  # [B,T,enc_size], [B,T]


        history_representations = take_3_by_2(                            
            representations, index=prev_times_idxs)                   # [B,T,D]

        v_h_t = self.w_t(history_representations)                         #[B,T,1]
        v_h_t = v_h_t.squeeze()                                         #[B,T]

        v_h_m = self.w_h(history_representations)                         #[B,T,M]

        if self.encoding == 'times_only':
            w_delta_t = self.w * (query - prev_times)                     # [B,T]
        else:
            w_delta_t = self.w * query_representations.squeeze(-1)                      #[B,T]
        
        base_log_intensity = v_h_t + w_delta_t                        # [B,T]


        if self.multi_labels:
            p_m = th.sigmoid(v_h_m)                                   # [B,T,M]
        else:
            p_m = th.softmax(v_h_m, dim=-1)                           # [B,T,M]
        regulariser = epsilon(dtype=p_m.dtype, device=p_m.device)
        p_m = p_m + regulariser

        marked_log_intensity = base_log_intensity.unsqueeze(
            dim=-1)  # [B,T,1]
        marked_log_intensity = marked_log_intensity + th.log(p_m)     # [B,T,M]

        intensity_mask = pos_delta_mask                                 # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)            # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        if self.encoding == "times_only":
            exp_1, exp_2 = v_h_t + w_delta_t, v_h_t                         # [B,T]
            # Avoid exponentiating to get masked infinity
            exp_1, exp_2 = exp_1 * intensity_mask, exp_2 * intensity_mask   # [B,T]
            base_intensity_itg = subtract_exp(exp_1, exp_2)
            base_intensity_itg = base_intensity_itg / self.w                # [B,T]
            base_intensity_itg = th.relu(base_intensity_itg)
        else:
            delta_t = (query - prev_times) * intensity_mask
            delta_t = delta_t + (delta_t == 0).float() * epsilon(
            dtype=delta_t.dtype, device=delta_t.device)
            base_intensity_itg  = th.exp(v_h_t) * th.pow(delta_t, self.w+1)/(self.w + 1)

        marked_intensity_itg = base_intensity_itg.unsqueeze(dim=-1)   # [B,T,1]
        marked_intensity_itg = marked_intensity_itg * p_m             # [B,T,M]

        artifacts_decoder = {
            "base_log_intensity": base_log_intensity,
            "base_intensity_integral": base_intensity_itg,
            "mark_probability": p_m}
        if artifacts is None:
            artifacts = {'decoder': artifacts_decoder}
        else:
            artifacts['decoder'] = artifacts_decoder

        check_tensor(marked_log_intensity)
        check_tensor(marked_intensity_itg * intensity_mask.unsqueeze(-1),
                     positive=True)
        return (marked_log_intensity,
                marked_intensity_itg,
                intensity_mask,
                artifacts)                      # [B,T,M], [B,T,M], [B,T], Dict
