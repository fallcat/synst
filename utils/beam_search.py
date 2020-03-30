'''
A module that implements beam search
'''
import torch
import pdb
import utils

class BeamHypothesis(object):
    ''' A class that represents a particular hypothesis in a beam '''
    def __init__(self, sequence, score, cache=None):
        self.score = score
        self.sequence = sequence
        self.cache = cache or {}

    def __len__(self):
        ''' The length of the hypothesis is the length of the sequence '''
        return len(self.sequence)


class Beam(object):
    ''' A class that represents a beam in the search '''
    def __init__(self, start_sequence, initial_score=0, max_sequence_length=0):
        ''' Initialize the beam '''
        self.max_sequence_length = max_sequence_length
        self.hypotheses = [BeamHypothesis(start_sequence, initial_score)]

    @property
    def best_hypothesis(self):
        ''' Returns the current best hypothesis given the score comparator '''
        return max(self.hypotheses, key=lambda h: h.score)

    def all_done(self, eos_idx):
        ''' Determine if the given beams have completed '''
        return all(
            self.finished_decoding(hypothesis, eos_idx)
            for hypothesis in self.hypotheses
        )

    def finished_decoding(self, hypothesis, eos_idx):
        ''' Check if the hypothesis has finished decoding '''
        return (
            eos_idx in hypothesis.sequence or
            (
                self.max_sequence_length > 0 and
                len(hypothesis.sequence) >= self.max_sequence_length
            )
        )


class BeamSearchDecoder(object):
    ''' Class that encapsulates decoding using beam search '''
    def __init__(self, model, eos_idx, config, span=1, ensemble=False):
        ''' Initialize the beam search decoder '''
        self.span = span
        self.model = model
        self.config = config
        self.eos_idx = eos_idx
        self.ensemble = ensemble

    @property
    def beam_width(self):
        ''' Get the beam width '''
        return self.config.beam_width

    def all_done(self, beams):
        ''' Determine if the given beams have completed '''
        return all(
            beam.all_done(self.eos_idx)
            for beam in beams
        )

    def collate(self, encoded, beams):
        ''' Collate beams into a batch '''
        """
            [sent-0-beam-0, sent-0-beam-1, sent-0-beam-2, sent-0-beam-3, sent-1-beam-0,...]
        """
        batch = []
        cache = []
        beam_map = {}
        encoded_batch = []
        beam_count = []
        for i, beam in enumerate(beams):
            hypothesis_map = {}
            cnt=0
            for hypothesis in beam.hypotheses:
                if beam.finished_decoding(hypothesis, self.eos_idx):
                    continue

                batch_idx = len(batch)
                cache.append(hypothesis.cache)
                encoded_batch.append(encoded[i])
                hypothesis_map[hypothesis] = batch_idx
                batch.append(hypothesis.sequence)
                cnt += 1

            if hypothesis_map:
                beam_map[beam] = hypothesis_map

            beam_count.append(cnt)

        batch = torch.LongTensor(batch)
        encoded_batch = utils.cat(encoded_batch)
        cache = utils.cat(cache) if not self.config.disable_cache else None

        return encoded_batch, batch, beam_map, cache, beam_count

    def initialize_search(self, start_sequences, max_lengths=0, initial_scores=0):
        ''' Initialize a batch of beams '''
        # pdb.set_trace()
        beams = []
        if isinstance(max_lengths, int):
            max_lengths = [max_lengths] * len(start_sequences)

        if isinstance(initial_scores, int):
            initial_scores = [initial_scores] * len(start_sequences)

        for sequence, score, max_length in zip(start_sequences, initial_scores, max_lengths):
            beams.append(Beam(sequence, score, max_length))

        return beams

    def normalized_score(self, score, length):
        '''
        Calculate the normalized score of the hypothesis

        https://arxiv.org/abs/1609.08144
        See equation #14
        '''
        return score * ((5 + 1) / (5 + length)) ** self.config.length_penalty

    def update_beam(self, scores, indices, beam, hypothesis_map, cache=None):
        ''' Update a particular beam '''
        lengths = []
        cache_lines = []
        beam_scores = []
        beam_indices = []
        hypotheses_scores = []
        for hypothesis in beam.hypotheses:
            batch_idx = hypothesis_map.get(hypothesis, -1)
            hypotheses_scores.append(hypothesis.score)
            if batch_idx >= 0:
                beam_scores.append(scores[batch_idx])
                beam_indices.append(indices[batch_idx])
                lengths.append(len(hypothesis) + self.span)

                if cache:
                    cache_lines.append(cache[batch_idx])
            else:
                beam_scores.append(scores.new_zeros((self.beam_width, self.span,)))
                beam_indices.append(indices.new_full((self.beam_width, self.span,), self.eos_idx))
                lengths.append(len(hypothesis))

                if cache:
                    cache_lines.append(hypothesis.cache)

        scores = torch.stack(beam_scores)
        indices = torch.stack(beam_indices)

        hypotheses_scores = scores.new_tensor(hypotheses_scores)
        scores = torch.sum(scores, -1) + hypotheses_scores[:, None]
        indices = indices.reshape(-1, self.span)
        scores = scores.reshape(-1)

        # pylint:disable=unused-variable
        lengths = torch.stack([scores.new_tensor(lengths)] * self.beam_width, 1).reshape(-1)
        normalized_scores = self.normalized_score(scores, lengths)
        normalized_scores, hypotheses_indices = torch.topk(normalized_scores, self.beam_width)
        # pylint:enable=unused-variable

        # need to convert each index into a hypothesis index
        # numpy searchsorted is a faster version of python's bisect.bisect[_left|_right]
        # that returns insertion points for multiple values
        new_hypotheses = []
        # pdb.set_trace()
        for new_hypothesis_idx in hypotheses_indices:
            base_hypothesis_idx = new_hypothesis_idx // self.beam_width
            base_hypothesis = beam.hypotheses[base_hypothesis_idx]
            if beam.finished_decoding(base_hypothesis, self.eos_idx):
                new_hypotheses.append(base_hypothesis)
                continue

            new_score = scores[new_hypothesis_idx]
            predictions = indices[new_hypothesis_idx]
            new_sequence = base_hypothesis.sequence + predictions.tolist()
            new_cache = cache_lines[base_hypothesis_idx] if cache else None
            new_hypotheses.append(BeamHypothesis(new_sequence, new_score, new_cache))

        beam.hypotheses = new_hypotheses

    def update_greedy(self, scores, indices, beam, hypothesis_map, cache=None):
        ''' Update a particular beam in a greedy fashion '''
        new_hypotheses = []
        for hypothesis in beam.hypotheses:
            batch_idx = hypothesis_map.get(hypothesis, -1)
            if batch_idx >= 0:
                new_cache = cache[batch_idx] if cache else None
                new_score = hypothesis.score + scores[batch_idx, 0]
                new_sequence = hypothesis.sequence + indices[batch_idx, 0].tolist()
                new_hypotheses.append(BeamHypothesis(new_sequence, new_score, new_cache))
            else:
                new_hypotheses.append(hypothesis)

        beam.hypotheses = new_hypotheses

    def update_beams(self, log_prob, beam_map, cache=None):
        ''' Update the beam batch '''
        # pdb.set_trace()
        scores, indices = torch.topk(log_prob, self.beam_width, 1)
        for beam, hypothesis_map in beam_map.items():
            if beam.all_done(self.eos_idx):
                continue

            if self.beam_width > 1:
                self.update_beam(scores, indices, beam, hypothesis_map, cache=cache)
            else:
                self.update_greedy(scores, indices, beam, hypothesis_map, cache=cache)

    def decode(self, encoded, beams, raw_layermask):
        ''' Decodes the given inputs '''
        self.model.eval()
        with torch.no_grad():
            encoded = utils.split_or_chunk(encoded, len(beams))
            while not self.all_done(beams):
                encoded_batch, batch, beam_map, cache, beam_count = self.collate(encoded, beams)

                logits = []
                updated_cache = []
                # expand raw_layermask
                if self.ensemble:
                    new_raw_layermask = raw_layermask
                elif len(raw_layermask.size()) == 1:
                    new_raw_layermask = raw_layermask
                else:
                    # pdb.set_trace()
                    assert len(raw_layermask) == len(beam_count)
                    if sum(beam_count) != raw_layermask.shape[0]:
                        new_raw_layermask = []
                        for eg, rtimes in zip(raw_layermask, beam_count):
                            if rtimes == 0:
                                continue
                            new_raw_layermask.append(eg.repeat(rtimes, 1))
                        new_raw_layermask = torch.cat(new_raw_layermask)
                    else:
                        new_raw_layermask = raw_layermask

                chunks = [(encoded_batch, batch, new_raw_layermask)]
                while chunks:
                    try:
                        encoded_batch, batch, r_layermask = chunks.pop()
                        
                        result = self.model(encoded_batch, batch, cache=cache, raw_layermask=r_layermask)


                        new_cache = result.get('cache')
                        if new_cache:
                            # pdb.set_trace()
                            updated_cache.extend(utils.split_or_chunk(new_cache, len(batch)))

                        full_logits = result['logits']
                        logits.append(full_logits[:, :, -self.span:])
                    except RuntimeError as rte:
                        if 'out of memory' in str(rte):
                            # This is the EAFP (easier to ask forgiveness than permission) approach
                            # to decoding. When the sequences being decoded become too long, it is
                            # possible to start running out of memory trying to decode all the
                            # sequences at once. This may for example happen early in training
                            # before the model has converged to output <EOS> tokens. Just split the
                            # current batch into two chunks and try again.

                            chunks.extend(zip(
                                utils.split_or_chunk(encoded_batch, 2),
                                utils.split_or_chunk(batch, 2),
                                [r_layermask] * 2 if self.ensemble else utils.split_or_chunk(r_layermask, 2)
                            ))
                            # pdb.set_trace()

                            # Additionally clear the cache in case the issue is related to allocator
                            # fragmentation.
                            torch.cuda.empty_cache()
                        else:
                            raise rte
                log_prob = torch.cat(logits)

                # normal
                log_prob = log_prob.log_softmax(1)

                # 1. naive

                # log_prob = log_prob.log_softmax(1)
                #
                # if self.ensemble:
                #     log_prob = log_prob.view(int(log_prob.shape[0] / raw_layermask.shape[0]), raw_layermask.shape[0], log_prob.shape[1], log_prob.shape[2]).mean(dim=1)

                # 2. nonnaive

                # if self.ensemble:
                #     log_prob = log_prob.view(int(log_prob.shape[0] / raw_layermask.shape[0]), raw_layermask.shape[0], log_prob.shape[1], log_prob.shape[2]).mean(dim=1)
                #
                # log_prob = log_prob.log_softmax(1)

                # 3. rank
                # log_prob = log_prob.log_softmax(1)
                # if self.ensemble:
                #     batch_size = int(log_prob.shape[0] / raw_layermask.shape[0])
                #     log_prob = log_prob.view(batch_size, raw_layermask.shape[0],
                #                              log_prob.shape[1], log_prob.shape[2])
                #     max_indices = log_prob.max(dim=2)[0].argmax(dim=1).view(-1)
                #     log_prob = log_prob[range(batch_size), max_indices]

                # 4. rank with avg topk
                # log_prob = log_prob.log_softmax(1)
                # if self.ensemble:
                #     # pdb.set_trace()
                #     batch_size = int(log_prob.shape[0] / raw_layermask.shape[0])
                #     log_prob = log_prob.view(batch_size, raw_layermask.shape[0],
                #                              log_prob.shape[1], log_prob.shape[2])
                #     max_indices = log_prob.topk(self.beam_width, dim=2)[0].mean(2).argmax(dim=1).view(-1)
                #     log_prob = log_prob[range(batch_size), max_indices]

                    # pdb.set_trace()

                self.update_beams(log_prob, beam_map, updated_cache)
            return beams

