# import shortcuts
try:
    from neon.models.autoencoder import Autoencoder  # noqa
    from neon.models.balance import Balance  # noqa
    from neon.models.balance import BalanceMP  # noqa
    from neon.models.dbn import DBN  # noqa
    from neon.models.mlp import MLP  # noqa
    from neon.models.rbm import RBM  # noqa
    from neon.models.rnn import RNN  # noqa
except ImportError:
    pass
