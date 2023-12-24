import mlx.core as mx
import mlx.nn as nn


class InputNormalization(nn.Module):
    """Performs mean and variance normalization of the input tensor.

    Arguments
    ---------
    mean_norm : True
         If True, the mean will be normalized.
    norm_type : str
         It defines how the statistics are computed ('sentence' computes them
         at sentence level, 'batch' at batch level, 'speaker' at speaker
         level, while global computes a single normalization vector for all
         the sentences in the dataset). Speaker and global statistics are
         computed with a moving average approach.
    avg_factor : float
         It can be used to manually set the weighting factor between
         current statistics and accumulated ones.

    Example
    -------
    >>> import torch
    >>> norm = InputNormalization()
    >>> inputs = torch.randn([10, 101, 20])
    >>> inp_len = torch.ones([10])
    >>> features = norm(inputs, inp_len)
    """

    from typing import Dict

    spk_dict_mean: Dict[int, mx.array]
    spk_dict_std: Dict[int, mx.array]
    spk_dict_count: Dict[int, int]

    def __init__(
        self,
        mean_norm=True,
        norm_type="global",
        avg_factor=None,
        requires_grad=False,
        update_until_epoch=3,
    ):
        super().__init__()
        self.mean_norm = mean_norm
        self.norm_type = norm_type
        self.avg_factor = avg_factor
        self.requires_grad = requires_grad
        self.glob_mean = mx.array([0])
        self.glob_std = mx.array([0])
        self.spk_dict_mean = {}
        self.spk_dict_std = {}
        self.spk_dict_count = {}
        self.weight = 1.0
        self.count = 0
        self.eps = 1e-10
        self.update_until_epoch = update_until_epoch

    def forward(self, x, lengths, spk_ids=mx.array([]), epoch=0):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : tensor
            A batch of tensors.
        lengths : tensor
            A batch of tensors containing the relative length of each
            sentence (e.g, [0.7, 0.9, 1.0]). It is used to avoid
            computing stats on zero-padded steps.
        spk_ids : tensor containing the ids of each speaker (e.g, [0 10 6]).
            It is used to perform per-speaker normalization when
            norm_type='speaker'.
        """

        N_batches = x.shape[0]

        current_means = []
        current_stds = []

        for snt_id in range(N_batches):
            # Avoiding padded time steps
            actual_size = int(mx.round(lengths[snt_id] * x.shape[1]).item())

            # computing statistics
            current_mean, current_std = self._compute_current_stats(
                x[snt_id, 0:actual_size, ...]
            )

            current_means.append(current_mean)
            current_stds.append(current_std)

            if self.norm_type == "sentence":
                x[snt_id] = (x[snt_id] - current_mean) / current_std

        return x

    def _compute_current_stats(self, x):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : tensor
            A batch of tensors.
        """
        # Compute current mean
        if self.mean_norm:
            current_mean = mx.mean(x, axis=0)
        else:
            current_mean = mx.array([0.0])

        # Compute current std
        current_std = mx.array([1.0])

        # Improving numerical stability of std
        current_std = mx.maximum(current_std, (self.eps * mx.ones_like(current_std)))

        return current_mean, current_std

    def _load_statistics_dict(self, state):
        """Loads the dictionary containing the statistics.

        Arguments
        ---------
        state : dict
            A dictionary containing the normalization statistics.
        """
        self.count = state["count"]
        if isinstance(state["glob_mean"], int):
            self.glob_mean = state["glob_mean"]
            self.glob_std = state["glob_std"]
        else:
            self.glob_mean = state["glob_mean"]
            self.glob_std = state["glob_std"]

        # Loading the spk_dict_mean in the right device
        self.spk_dict_mean = {}
        for spk in state["spk_dict_mean"]:
            self.spk_dict_mean[spk] = state["spk_dict_mean"][spk]

        # Loading the spk_dict_std in the right device
        self.spk_dict_std = {}
        for spk in state["spk_dict_std"]:
            self.spk_dict_std[spk] = state["spk_dict_std"][spk]

        self.spk_dict_count = state["spk_dict_count"]

        return state

    def _load(self, path=None, end_of_epoch=False, device=None):
        """Load statistic dictionary.

        Arguments
        ---------
        path : str
            The path of the statistic dictionary
        device : str, None
            Passed to torch.load(..., map_location=device)
        """
        del end_of_epoch  # Unused here.
        # stats = torch.load(path, map_location=device)

        stats = {
            "glob_std": mx.array([1]),
            "glob_mean": mx.array(
                [
                    -1.0373159646987915,
                    0.004313964396715164,
                    3.2695112228393555,
                    2.3305561542510986,
                    -0.15005721151828766,
                    -2.0687949657440186,
                    -2.6122326850891113,
                    -1.1647052764892578,
                    -3.6624019145965576,
                    -2.2515218257904053,
                    1.6357080936431885,
                    -1.119925618171692,
                    0.006497358437627554,
                    -1.6516363620758057,
                    2.9809319972991943,
                    2.3903005123138428,
                    0.4542860686779022,
                    -2.5051052570343018,
                    0.9176740646362305,
                    -0.733771562576294,
                    0.16936202347278595,
                    -2.124788761138916,
                    1.6698431968688965,
                    1.8387510776519775,
                    -0.08849123865365982,
                    -1.2149461507797241,
                    0.7280610203742981,
                    2.271894693374634,
                    -0.43160873651504517,
                    -2.8749468326568604,
                    -0.559111475944519,
                    1.6032946109771729,
                    -0.9148209691047668,
                    1.4720821380615234,
                    1.1198285818099976,
                    -1.4893202781677246,
                    -0.6628988981246948,
                    2.8217077255249023,
                    -0.4240037798881531,
                    0.034323032945394516,
                    0.8499836325645447,
                    -0.4805501699447632,
                    0.5753277540206909,
                    -1.8313301801681519,
                    -0.4765237867832184,
                    -0.4623653292655945,
                    -2.476926565170288,
                    -2.5266642570495605,
                    -0.43583226203918457,
                    -0.13821762800216675,
                    -3.5822761058807373,
                    1.2447354793548584,
                    -3.0051748752593994,
                    -1.1562477350234985,
                    1.4163520336151123,
                    -0.08225936442613602,
                    -1.3980894088745117,
                    1.4378186464309692,
                    -2.0002541542053223,
                    -0.7509810328483582,
                    0.5473135113716125,
                    1.4607231616973877,
                    -2.9042270183563232,
                    -0.06595407426357269,
                    0.490867555141449,
                    -0.13157211244106293,
                    2.7749292850494385,
                    -1.4899675846099854,
                    -1.4631725549697876,
                    -3.5752434730529785,
                    -3.735719680786133,
                    1.813037633895874,
                    -2.9033639430999756,
                    -1.5314199924468994,
                    0.044556256383657455,
                    -0.8697152733802795,
                    -0.2519349157810211,
                    -0.2272966057062149,
                    -2.2982749938964844,
                    0.3571125268936157,
                    0.4964159429073334,
                    -1.3591055870056152,
                    1.8846709728240967,
                    0.11263932287693024,
                    2.0511837005615234,
                    1.550254464149475,
                    0.12858016788959503,
                    -0.8767280578613281,
                    -1.5414769649505615,
                    -0.19298221170902252,
                    0.7146889567375183,
                    -2.9431040287017822,
                    3.112628936767578,
                    2.13198184967041,
                    -0.8264098763465881,
                    0.4612421989440918,
                    0.3205318748950958,
                    0.3296632766723633,
                    -3.2684059143066406,
                    0.23895543813705444,
                    0.3636946678161621,
                    -0.12437890470027924,
                    1.3755600452423096,
                    0.255165696144104,
                    1.0819262266159058,
                    -0.13521458208560944,
                    0.537755012512207,
                    -0.09236566722393036,
                    -1.8614071607589722,
                    3.5387070178985596,
                    1.0389963388442993,
                    0.6034169793128967,
                    1.4116590023040771,
                    0.7125404477119446,
                    0.0900835171341896,
                    0.2674475312232971,
                    1.1767133474349976,
                    5.2197651863098145,
                    2.3130791187286377,
                    0.10807593911886215,
                    -1.0737806558609009,
                    -0.5112389326095581,
                    0.4291580021381378,
                    -3.940735101699829,
                    1.948137879371643,
                    -1.6503609418869019,
                    0.8884319067001343,
                    0.5542525053024292,
                    -2.9716341495513916,
                    -1.6699655055999756,
                    1.1195788383483887,
                    -0.2553310692310333,
                    -3.874058485031128,
                    0.9672127366065979,
                    -1.5142803192138672,
                    0.6499041318893433,
                    3.5598418712615967,
                    0.15086598694324493,
                    -2.3249998092651367,
                    -3.2102720737457275,
                    -1.9018909931182861,
                    -2.468729257583618,
                    0.640428364276886,
                    -0.7220006585121155,
                    3.6187126636505127,
                    1.3778181076049805,
                    -0.9557759761810303,
                    -0.8157440423965454,
                    -2.300356149673462,
                    -1.9137400388717651,
                    -0.2345997393131256,
                    -1.2996282577514648,
                    -3.1549272537231445,
                    1.8740365505218506,
                    2.1052587032318115,
                    -0.7936500310897827,
                    0.5830777883529663,
                    0.22441917657852173,
                    -2.363966703414917,
                    0.435719758272171,
                    -1.1352617740631104,
                    -3.0252480506896973,
                    -1.654279351234436,
                    -2.3213794231414795,
                    -0.5806937217712402,
                    -1.5413265228271484,
                    -0.09943648427724838,
                    3.261835813522339,
                    -0.6869571805000305,
                    -2.8344998359680176,
                    1.7905585765838623,
                    -0.18342450261116028,
                    -1.8079222440719604,
                    -1.2684203386306763,
                    -1.1088321208953857,
                    -0.5944979190826416,
                    -1.595962405204773,
                    -1.4731144905090332,
                    -0.632537305355072,
                    0.29566100239753723,
                    -1.044356107711792,
                    -1.7863552570343018,
                    -0.33273231983184814,
                    0.10807638615369797,
                    1.7756731510162354,
                    1.6043795347213745,
                    1.055188775062561,
                    -0.05681168660521507,
                    -0.4484957456588745,
                    1.1689050197601318,
                    1.621566653251648,
                    -0.2107294797897339,
                ]
            ),
            "spk_dict_mean": {},
            "spk_dict_std": {},
            "spk_dict_count": {},
            "count": 18750,
        }
        self._load_statistics_dict(stats)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def load_mean_var_norm() -> InputNormalization:
    norm = InputNormalization(norm_type="sentence")
    norm._load()
    return norm


if __name__ == "__main__":
    norm = InputNormalization(norm_type="sentence")
    norm._load()

    inputs = mx.random.normal([64, 501, 80])
    inp_len = mx.ones([64])
    features = norm(inputs, inp_len)
    print(features)
