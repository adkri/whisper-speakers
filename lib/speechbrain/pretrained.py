import mlx.core as mx

from functools import cached_property
import numpy as np

from lib.speechbrain.embedding import load_embedding_model
from lib.speechbrain.input_normalization import load_mean_var_norm
from lib.speechbrain.features import Fbank


class Encoder:
    """A ready-to-use class for utterance-level classification (e.g, speaker-id,
    language-id, emotion recognition, keyword spotting, etc).

    The class assumes that an encoder called "embedding_model" and a model
    called "classifier" are defined in the yaml file. If you want to
    convert the predicted index into a corresponding text label, please
    provide the path of the label_encoder in a variable called 'lab_encoder_file'
    within the yaml.

    The class can be used either to run only the encoder (encode_batch()) to
    extract embeddings or to run a classification step (classify_batch()).
    ```

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.pretrained import EncoderClassifier
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> classifier = EncoderClassifier.from_hparams(
    ...     source="speechbrain/spkrec-ecapa-voxceleb",
    ...     savedir=tmpdir,
    ... )
    >>> classifier.hparams.label_encoder.ignore_len()

    >>> # Compute embeddings
    >>> signal, fs = torchaudio.load("tests/samples/single-mic/example1.wav")
    >>> embeddings = classifier.encode_batch(signal)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.compute_features = Fbank(n_mels=80)
        self.embedding_model = load_embedding_model()
        self.mean_var_norm = load_mean_var_norm()

    def encode_batch(self, wavs, wav_lens=None):
        """Encodes the input audio into a single vector embedding.

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = <this>.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        normalize : bool
            If True, it normalizes the embeddings with the statistics
            contained in mean_var_norm_emb.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = mx.expand_dims(wavs, axis=0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = mx.ones(wavs.shape[0])

        # Storing waveform in the specified device
        print(type(wavs))
        # wavs = wavs.astype(mx.fl)

        # Computing features and embeddings
        print("wav shape", wavs.shape)
        features = self.compute_features(wavs)
        print("features.shape after compute", features.shape)

        print("feats.shape", features.shape)
        print("wav_lens.shape", wav_lens.shape)
        features = self.mean_var_norm(mx.array(features), mx.array(wav_lens))

        print("Calling embedding model")
        embeddings = self.embedding_model(mx.array(features), mx.array(wav_lens))
        print("embeddings.shape", embeddings.shape)
        return embeddings

    def __call__(self, wavs, wav_lens=None):
        return self.encode_batch(wavs, wav_lens)


class PretrainedSpeakerEmbedding:
    def __init__(self) -> None:
        self.encoder = Encoder()

    @cached_property
    def sample_rate(self) -> int:
        return 16000  # self.classifier_.audio_normalizer.sample_rate

    @cached_property
    def dimension(self) -> int:
        dummy_waveforms = mx.random.normal([1, 16000])
        *_, dimension = self.encoder.encode_batch(dummy_waveforms).shape
        return dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        lower, upper = 2, round(0.5 * self.sample_rate)
        middle = (lower + upper) // 2
        while lower + 1 < upper:
            try:
                _ = self.encoder.encode_batch(mx.random.normal([1, middle]))
                upper = middle
            except RuntimeError:
                lower = middle

            middle = (lower + upper) // 2

        return upper

    def __call__(self, waveforms, masks=None) -> np.ndarray:
        """

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional

        Returns
        -------
        embeddings : (batch_size, dimension)

        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1, "Only mono waveforms are supported."

        waveforms = np.array(waveforms)
        waveforms = np.squeeze(waveforms, axis=1)

        print("waveforms.shape", waveforms.shape)

        if masks is None:
            signals = np.squeeze(waveforms, axis=1)
            wav_lens = signals.shape[1] * mx.ones(batch_size)
        else:
            batch_size_masks, _ = masks.shape
            assert batch_size == batch_size_masks

            # TODO: speed up the creation of "signals"
            # preliminary profiling experiments show
            # that it accounts for 15% of __call__
            # (the remaining 85% being the actual forward pass)

            # imasks = F.interpolate(
            #     masks.unsqueeze(dim=1), size=num_samples, mode="nearest"
            # ).squeeze(dim=1)
            # interpolate masks
            masks = np.array(masks)
            imasks = custom_zoom_nearest_numpy(masks, num_samples)

            imasks = imasks > 0.5
            # pad sequences
            max_length = max(imask.sum() for imask in imasks)
            # signals = np.array([np.pad(waveform[imask])])
            signals = np.array(
                [
                    np.pad(
                        waveform[imask], (0, max_length - imask.sum()), mode="constant"
                    )
                    for waveform, imask in zip(waveforms, imasks)
                ]
            )

            # masks_expaneded = np.expand_dims(masks, axis=1)
            # scale_factor = num_samples / masks_expaneded.shape[2]
            # imasks = np.repeat(masks_expaneded, scale_factor, axis=2)
            # imasks = np.squeeze(imasks, axis=1)

            # imasks = imasks > 0.5
            # imasks = mx.array(imasks)

            # signals = pad_sequence_mx(
            #     [waveform[imask] for waveform, imask in zip(waveforms, imasks)],
            #     batch_first=True,
            # )

            wav_lens = imasks.sum(axis=1)

        max_len = wav_lens.max()

        # corner case: every signal is too short
        if max_len < self.min_num_samples:
            return np.NAN * np.zeros((batch_size, self.dimension))

        too_short = wav_lens < self.min_num_samples
        wav_lens = wav_lens / max_len
        wav_lens[too_short] = 1.0

        embeddings = mx.squeeze(
            self.encoder.encode_batch(signals, wav_lens=wav_lens), axis=1
        )

        print("copying embeddings...")
        print("batch embeddings.shape", embeddings.shape)
        embeddings = np.array(embeddings)
        embeddings[too_short] = np.NAN

        return embeddings


def custom_zoom_nearest_numpy(array, new_length):
    """
    Resize a 2D array along its second axis using nearest neighbor interpolation with NumPy.

    :param array: 2D input array.
    :param new_length: New length of the second dimension.
    :return: Resized array.
    """
    old_length = array.shape[1]
    scale_factor = new_length / old_length

    # Create an array of indices
    indices = np.floor(np.arange(new_length) / scale_factor).astype(int)
    # Clamp the indices to be within the array bounds
    indices = np.clip(indices, 0, old_length - 1)

    # Use advanced indexing to create the new array
    return array[:, indices]


def pad_sequence_mx(sequences, batch_first=False, padding_value=0.0):
    max_length = max([s.shape[0] for s in sequences])
    padded_sequences = []

    for seq in sequences:
        padded_seq = mx.pad(
            seq, ((0, max_length - seq.shape[0]), (0, 0)), constant_values=padding_value
        )
        padded_sequences.append(padded_seq)

    if batch_first:
        return mx.stack(padded_sequences)
    else:
        return mx.transpose(mx.stack(padded_sequences), (1, 0, 2))


if __name__ == "__main__":
    # num_samples = 80000  # or your target length
    # imasks = custom_zoom_nearest_numpy(
    #     np.array(mx.random.normal([64, 293])), num_samples
    # )
    # print(imasks.shape)

    feats = mx.random.normal([64, 63071])
    wav_lens = mx.random.normal([64])
    emb_outputs = Encoder().encode_batch(feats, wav_lens)
    print(emb_outputs.shape)

    batch_feats = mx.random.normal([64, 1, 80000])
    batch_mask = mx.random.normal([64, 293])
    emb_outputs = PretrainedSpeakerEmbedding()(batch_feats, batch_mask)
    print(emb_outputs.shape)
    # import torch
    # from torch.nn.utils.rnn import pad_sequence

    # a = torch.ones(25, 300)
    # b = torch.ones(22, 300)
    # c = torch.ones(15, 300)

    # result = pad_sequence([a, b, c], batch_first=True)
    # print(result.shape)

    # a = mx.ones((25, 300))
    # b = mx.ones((22, 300))
    # c = mx.ones((15, 300))
    # result = pad_sequence_mx([a, b, c], batch_first=True)
    # print(result.shape)
