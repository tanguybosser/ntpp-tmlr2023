import math
import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple, List

from tpp.models.decoders.base.variable_history import VariableHistoryDecoder
from tpp.pytorch.models import LAYER_CLASSES

from tpp.utils.events import Events
from tpp.utils.index import take_3_by_2, take_2_by_2
from tpp.utils.stability import epsilon, check_tensor
from tpp.utils.encoding import encoding_size

class LogNormalMixtureDecoder(VariableHistoryDecoder):
    """Analytic decoder process, uses a closed form for the intensity
    to train the model.
    See https://arxiv.org/pdf/1909.12127.pdf.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            n_mixture: int,
            units_mlp: List[int],
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            encoding: Optional[str] = "times_only",
            embedding_constraint: Optional[str] = None,
            emb_dim: Optional[int] = 2,
            **kwargs):
        if encoding not in ["times_only", "learnable"]:
            raise ValueError("Invalid event encoding for LogNormMix decoder.")
        if encoding == 'learnable' and embedding_constraint is None:
            print("Warning, embedding are unconstrained for LongNormMix decoder, setting to 'nonneg'")
            embedding_constraint = 'nonneg'
        super(LogNormalMixtureDecoder, self).__init__(
            name="log-normal-mixture",
            input_size=units_mlp[0],
            marks=marks,
            encoding=encoding, 
            emb_dim=emb_dim,
            embedding_constraint=embedding_constraint,  
            **kwargs)
        if len(units_mlp) < 2:
            raise ValueError("Units of length at least 2 need to be specified")
        enc_size = encoding_size(encoding=encoding, emb_dim=emb_dim)
        self.mu = nn.Linear(in_features=units_mlp[0], out_features=n_mixture)
        self.s = nn.Linear(in_features=units_mlp[0], out_features=n_mixture)
        self.w = nn.Linear(in_features=units_mlp[0], out_features=n_mixture)
        if self.encoding == "learnable":
            w_t_layer = LAYER_CLASSES[self.embedding_constraint]
            self.w_t = w_t_layer(in_features=enc_size, out_features=1)
        self.marks1 = nn.Linear(
            in_features=units_mlp[0], out_features=units_mlp[1])
        self.marks2 = nn.Linear(
            in_features=units_mlp[1], out_features=marks)
        self.multi_labels = multi_labels

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
        
        query.requires_grad = True
        
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
            representations, index=prev_times_idxs)  # [B,T,D]
        if self.encoding == "times_only":
            delta_t = query - prev_times  # [B,T]
            delta_t = delta_t.unsqueeze(-1)  # [B,T,1]
            delta_t = th.relu(delta_t) 
            delta_t = delta_t + (delta_t == 0).float() * epsilon(
            dtype=delta_t.dtype, device=delta_t.device)
            delta_t = th.log(delta_t)
        else:
            delta_t = self.w_t(query_representations) #[B,T]

        mu = self.mu(history_representations)  # [B,T,K]
        std = th.exp(self.s(history_representations))
        w = th.softmax(self.w(history_representations), dim=-1)
        if self.multi_labels:
            p_m = th.sigmoid(
                self.marks2(th.tanh(self.marks1(history_representations))))
        else:
            p_m = th.softmax(
                self.marks2(
                    th.tanh(self.marks1(history_representations))), dim=-1)
        cum_f = w * 0.5 * (
                1 + th.erf((delta_t - mu) / (std * math.sqrt(2))))
        cum_f = th.clamp(th.sum(cum_f, dim=-1), max=1-1e-6)
        one_min_cum_f = 1. - cum_f
        one_min_cum_f = th.relu(one_min_cum_f) + epsilon(
            dtype=cum_f.dtype, device=cum_f.device)
        f = th.autograd.grad(
            outputs=cum_f,
            inputs=query,
            grad_outputs=th.ones_like(cum_f),
            retain_graph=True,
            create_graph=True)[0]
        query.requires_grad = False
    
        f = f + epsilon(dtype=f.dtype, device=f.device)

        
        base_log_intensity = th.log(f / one_min_cum_f)
        marked_log_intensity = base_log_intensity.unsqueeze(
            dim=-1)  # [B,T,1]
        marked_log_intensity = marked_log_intensity + th.log(p_m)  # [B,T,M]

        base_intensity_itg = - th.log(one_min_cum_f)
        marked_intensity_itg = base_intensity_itg.unsqueeze(dim=-1)  # [B,T,1]
        marked_intensity_itg = marked_intensity_itg * p_m  # [B,T,M] 

        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)  # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        artifacts_decoder = {
            "base_log_intensity": base_log_intensity,
            "base_intensity_integral": base_intensity_itg,
            "mark_probability": p_m}
        if artifacts is None:
            artifacts = {'decoder': artifacts_decoder}
        else:
            artifacts['decoder'] = artifacts_decoder

        check_tensor(marked_log_intensity * intensity_mask.unsqueeze(-1))
        check_tensor(marked_intensity_itg * intensity_mask.unsqueeze(-1),
                     positive=True)
        artifacts["mu"] = mu.detach()[:,-1,:].squeeze().cpu().numpy() 
        artifacts["sigma"] = std.detach()[:,-1,:].squeeze().cpu().numpy()
        artifacts["w"] = w.detach()[:,-1,:].squeeze().cpu().numpy()

        return (marked_log_intensity,
                marked_intensity_itg,
                intensity_mask,
                artifacts)                      # [B,T,M], [B,T,M], [B,T], Dict
