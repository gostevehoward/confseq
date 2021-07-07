"""
These "unit tests" are purely visual, and won't be run via pytest.
The main goal here is just to make sure that the confidence sequences
look reasonable. It is advisable to run these before pushing to GitHub.
"""

from confseq.cs_plots import *
from confseq.betting_strategies import *
from confseq.betting import *
from confseq.predmix import (
    predmix_empbern_cs_wor,
    predmix_hoeffding_cs_wor,
)
from scipy.stats import binom, beta

alpha = 0.05
N = 10000

cs_list = [
    ConfseqToPlot(
        lambda x: predmix_hoeffding_cs_wor(
            x, N=N, alpha=alpha, running_intersection=False
        ),
        "PM-H",
        "tab:orange",
        "-.",
    ),
    ConfseqToPlot(
        lambda x: predmix_empbern_cs_wor(
            x, N=N, alpha=alpha, running_intersection=False
        ),
        "PM-EB",
        "tab:blue",
        "--",
    ),
    ConfseqToPlot(
        lambda x: betting_cs(
            x,
            N=N,
            breaks=500,
            trunc_scale=1 / 2,
            alpha=alpha,
            parallel=True,
            running_intersection=False,
        ),
        r"Hedged",
        "tab:green",
        "-",
    ),
]


dgp_list = [
    DataGeneratingProcess(
        data_generator_fn=lambda: np.random.binomial(1, 0.5, N),
        name="highvar",
        title="$High-variance",
    ),
    DataGeneratingProcess(
        data_generator_fn=lambda: np.random.binomial(1, 0.1, N),
        name="evenly_distributed",
        title="Evenly-distributed",
    ),
    DataGeneratingProcess(
        data_generator_fn=lambda: np.random.beta(10, 30, N),
        name="asymm_lowvar",
        title="Asymmetric, low-variance",
    ),
]

plot_CSs(dgp_list=dgp_list, cs_list=cs_list, folder=".", nsim=5)
