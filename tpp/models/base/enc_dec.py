import torch as th

from typing import Dict, Optional, Tuple

from tpp.models.decoders.base.decoder import Decoder
from tpp.models.encoders.base.encoder import Encoder
from tpp.models.base.process import Process
from tpp.utils.events import Events
from tpp.utils.history_bst import get_prev_times
from tpp.utils.index import take_2_by_1
from tpp.utils.logical import xor
from tpp.utils.stability import epsilon

class EncDecProcess(Process):
    """A parametric encoder decoder process.

    Args
        encoder: The encoder.
        decoder: The decoder.

    """
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 multi_labels: Optional[bool] = False,
                 **kwargs):
        if encoder is not None:
            assert encoder.marks == decoder.marks
            name = '_'.join([encoder.name, decoder.name])
            marks = encoder.marks

            if decoder.input_size is not None:
                assert encoder.output_size == decoder.input_size

        else:
            name = kwargs.pop("name")
            marks = kwargs.pop("marks")
        super(EncDecProcess, self).__init__(name=name, marks=marks, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.multi_labels = multi_labels

        if self.encoder is not None:
            self.enc_dec_hidden_size = self.encoder.output_size

    def intensity(
            self, query: th.Tensor, events: Events
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the intensities at query times given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            intensity: [B,T,M] The intensities for each query time for each
                mark (class).
            intensity_mask: [B,T,M] Which intensities are valid for further
                computation based on e.g. sufficient history available.

        """
        log_intensity, _, intensity_mask, _ = self.artifacts(
            query=query, events=events)
        return th.exp(log_intensity), intensity_mask

    def log_density(
            self, query: th.Tensor, events: Events
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the log densities at query times given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            log_density: [B,T,M] The densities for each query time for each
                mark (class).
            density_mask: [B,T,M] Which intensities are valid for further
                computation based on e.g. sufficient history available.

        """
        log_intensity, intensity_integral, intensity_mask, _ = self.artifacts(
            query=query, events=events)
        ground_intensity_integral = intensity_integral.sum(-1).unsqueeze(-1)
        log_density = log_intensity - ground_intensity_integral
        log_mark_density = log_intensity - th.logsumexp(log_intensity, dim=-1).unsqueeze(-1)
        return log_density, log_mark_density, intensity_mask

    def neg_log_likelihood(
            self, events: Events, test:Optional[bool]=False) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the negative log likelihood of events.

        Args:
            events: [B,L] Times and labels of events.
        Returns:
            nll: [B] The negative log likelihoods for each sequence.
            nll_mask: [B] Which neg_log_likelihoods are valid for further
                computation based on e.g. at least one element in sequence has
                a contribution.
            artifacts: Other useful items, e.g. the relevant window of the
                sequence.

        """
        events_times = events.get_times(postpend_window=True)         # [B,L+1] 
        log_intensity, intensity_integral, intensity_mask, artifacts_modular = self.artifacts(
            query=events_times, events=events)  # [B,L+1,M], [B,L+1,M], [B,L+1]
        # For the interval normalisation
        shift = 1. + th.max(events_times) - th.min(events_times)
        shifted_events = events_times + (1 - intensity_mask) * shift 
        interval_start_idx = th.min(shifted_events, dim=-1).indices
        interval_start_times = events.get_times(prepend_window=True)
        interval_start_times = take_2_by_1(
            interval_start_times, index=interval_start_idx)

        interval_end_idx = th.max(events_times, dim=-1).indices
        interval_end_times = take_2_by_1(
            events_times, index=interval_end_idx)

        interval = interval_end_times - interval_start_times

        artifacts = {
            "interval_start_times": interval_start_times,
            "interval_end_times": interval_end_times,
            "interval": interval}

        log_intensity = log_intensity[:, :-1, :]                 # [B,L,M]
        
        intensity_integral = th.sum(intensity_integral, dim=-1)  # [B,L+1] 
        window_integral = intensity_integral[:, -1]              # [B]
        exp_window_integral = th.exp(-window_integral)
        intensity_integral = intensity_integral[:, :-1]          # [B,L]
        #Cumulative density  
        if test:
            cumulative_density = 1 - th.exp(-intensity_integral) #[B, L]
            cumulative_density[~intensity_mask[:,:-1].bool()] = -1      #[B,L]
            artifacts["cumulative density"] = cumulative_density.detach()
        
        window_intensity_mask = intensity_mask[:, -1]          # [B]
        intensity_mask = intensity_mask[:, :-1]                # [B,L]
        labels = events.labels                                        # [B,L,M]

        if 'alpha' in artifacts_modular:
            for k in ['log_intensity_0', 'log_intensity_1']:
                log_intensity_k = artifacts_modular[k]
                log_intensity_k = log_intensity_k[:,:-1,:]
                intensity_k = th.exp(log_intensity_k)
                intensity_k = intensity_k * labels * intensity_mask.unsqueeze(dim=-1)
                intensity_k = th.sum(intensity_k, dim=(1,2))
                artifacts[k.lstrip('log_')] = intensity_k
            artifacts['alpha'] = artifacts_modular['alpha']
        for k in artifacts_modular.keys():
            if isinstance(artifacts_modular[k], dict):
                if 'mu' in artifacts_modular[k]:
                    artifacts["mu"] = artifacts_modular[k]["mu"]
                    artifacts["sigma"] = artifacts_modular[k]["sigma"]
                    artifacts["w"] = artifacts_modular[k]["w"]
        log_density = (log_intensity
                       - intensity_integral.unsqueeze(dim=-1))        # [B,L,M]
        log_density = log_density * intensity_mask.unsqueeze(dim=-1)  # [B,L,M]

        log_ground_density = th.logsumexp(log_density, dim=-1) * intensity_mask #[B,L]
        log_ground_density_unsqueezed = log_ground_density.unsqueeze(-1).repeat(1,1, log_density.shape[-1]) #[B,L,M]        
        
        log_mark_density = log_density - log_ground_density_unsqueezed            
        log_mark_density = log_mark_density * intensity_mask.unsqueeze(-1)
        log_mark_density = log_mark_density * labels #[B,L,M]
        true_log_mark_density = th.sum(log_mark_density, dim=-1)

        artifacts["true log density"] = th.sum(log_ground_density, dim=-1).detach()
        artifacts["true mark density"] = th.sum(log_mark_density, dim=(1,2)).detach()
        if self.multi_labels:
            eps = epsilon(dtype=log_density.dtype, device=log_density.device)
            log_density = th.clamp(log_density, max=-eps)
            one_min_density = 1. - th.exp(log_density) + eps  # [B,L,M]
            log_one_min_density = th.log(one_min_density)  # [B,L,M]
            log_one_min_density = (log_one_min_density *
                                   intensity_mask.unsqueeze(dim=-1))
            one_min_true_log_density = (1. - labels) * log_one_min_density
            one_min_true_log_density_flat = one_min_true_log_density.reshape(
                one_min_true_log_density.shape[0], -1)  # [B,L*M]
            log_likelihood = log_likelihood + th.sum(
                one_min_true_log_density_flat, dim=-1)  # [B]

        add_window_integral = 1 - events.final_event_on_window.type(
            log_ground_density.dtype)                                     # [B]
        window_integral = window_integral * add_window_integral       # [B]
        exp_window_integral = exp_window_integral * add_window_integral
        
        artifacts["window integral"] = window_integral.detach()             # [B]
        
        log_likelihood = th.sum(log_ground_density, dim=-1) + th.sum(true_log_mark_density, dim=-1) - window_integral #[B]
        loss =  - log_likelihood
        loss_mask = th.sum(intensity_mask, dim=-1)                     # [B]
        loss_mask = (loss_mask > 0.).type(loss.dtype)                    # [B]
        defined_window_integral = window_intensity_mask * add_window_integral
        no_window_integral = 1 - add_window_integral                    # [B]
        window_mask = xor(defined_window_integral, no_window_integral)  # [B]
        loss_mask = loss_mask * window_mask
        return loss, loss_mask, artifacts 

    def artifacts(
            self, query: th.Tensor, events: Events
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Compute the (log) intensities and intensity integrals at query times
        given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            log_intensity: [B,T,M] The log intensities for each query time for
                each mark (class).
            intensity_integrals: [B,T,M] The integral of the intensity from
                the most recent event to the query time for each mark.
            intensities_mask: [B,T,M] Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: A dictionary of whatever else you might want to return.

        """
        representations, representations_mask, artifacts = self.encode(
            events=events)                            # [B,L+1,D] [B,L+1], Dict
        prev_times, is_event, pos_delta_mask = get_prev_times( #
            query=query,
            events=events,
            allow_window=True)# ([B,T],[B,T]), [B,T], [B,T]
        
        prev_times, prev_times_idxs = prev_times  # [B,T], [B,T]
        return self.decode(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            is_event=is_event,
            pos_delta_mask=pos_delta_mask,
            representations=representations,
            representations_mask=representations_mask,
            artifacts=artifacts)

    def encode(self, events: Events) -> Tuple[th.Tensor, th.Tensor, Dict]:
        return self.encoder(events=events) 

    def decode(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            is_event: th.Tensor,
            pos_delta_mask: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None
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
        return self.decoder(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            is_event=is_event,
            pos_delta_mask=pos_delta_mask,
            representations=representations,
            representations_mask=representations_mask,
            artifacts=artifacts)
