import torch
import torchaudio as ta
import augment
import numpy as np
from torch import Tensor
import os
import time




class ChainRunner:
    """
    Takes an instance of augment.EffectChain and applies it on pytorch tensors.
    """

    def __init__(self, chain):
        self.chain = chain

    def __call__(self, x):
        """
        x: torch.Tensor, (channels, length). Must be placed on CPU.
        """
        src_info = {'channels': x.size(0),  # number of channels
                    'length': x.size(1),   # length of the sequence
                    'precision': 16,       # precision (16, 32 bits)
                    'rate': 16000.0,       # sampling rate
                    'bits_per_sample': 16}  # size of the sample

        target_info = {'channels': 1,
                       'length': x.size(1),
                       'precision': 16,
                       'rate': 16000.0,
                       'bits_per_sample': 16}

        y = self.chain.apply(
            x, src_info=src_info, target_info=target_info)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()
        return y


# Generate a random shift applied to the speaker's pitch
def random_pitch_shift():
    return np.random.randint(-300, 300)


def complex_norm(
        complex_tensor: Tensor,
        power: float = 1.0
) -> Tensor:
    r"""Compute the norm of complex tensor input.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`).

    Returns:
        Tensor: Power of the normed input tensor. Shape of `(..., )`
    """

    # Replace by torch.norm once issue is fixed
    # https://github.com/pytorch/pytorch/issues/34279
    return complex_tensor.pow(2.).sum(-1).pow(0.5 * power)

if __name__ == '__main__':

    path = r'/home/ljj/PycharmProjects/TransformerAndLm/test/OS00013867.wav'

    effect_chain_past = augment.EffectChain()
    # The pitch effect changes the sampling ratio; we have to compensate for that.
    # Here, we specify 'quick' options on both pitch and rate effects, to speed up things
    # effect_chain_past.pitch("-q", random_pitch_shift).rate("-q", 16_000)
    effect_chain_past.pitch("-q", -500).rate("-q", 16_000)
    # Futher, we add an effect that randomly drops one 50ms subsequence
    effect_chain_past.time_dropout(max_seconds=50 / 1000)

    effect_chain_past_runner = ChainRunner(effect_chain_past)

    waveform, sample_frequency = ta.load(path)
    augmented, original = waveform, waveform
    print(waveform.shape)

    featureOld = ta.compliance.kaldi.fbank(waveform, num_mel_bins=40, sample_frequency=sample_frequency,
                            dither=0.0)
    featureNew = ta.transforms.Spectrogram(win_length=400, hop_length=160, power=None)(waveform)
    print(f'featureNew1 = {featureNew.shape}')
    featureNew = ta.transforms.TimeStretch(hop_length=160, n_freq=201, fixed_rate=1.1)(featureNew)
    print(f'featureNew2 = {featureNew.shape}')
    featureNew = complex_norm(featureNew, power=2.)
    print(f'featureNew3 = {featureNew.shape}')

    featureNew = ta.transforms.MelScale(n_mels=40, sample_rate=sample_frequency)(featureNew)
    print(f'featureNew4 = {featureNew.shape}')
    featureNew = featureNew.squeeze(dim=0)
    print(f'featureNew4 = {featureNew.shape}')
    featureNew = featureNew.T
    print(f'featureNew5 = {featureNew.shape}')
    print(f'featureOld = {featureOld.shape}')

    # augmented = effect_chain_past_runner(augmented)
    #
    #
    # ta.save('test/augmented.wav', augmented, sample_frequency)
    # print(augmented.shape)
    # ta.save('test/original.wav', original, sample_frequency)
    # print('Saved examples of augmented and non-augmented sequences to augmented.wav and original.wav')

