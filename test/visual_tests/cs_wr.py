"""
These "unit tests" are purely visual, and won't be run via pytest.
The main goal here is just to make sure that the confidence sequences
look reasonable. It is advisable to run these before pushing to GitHub.
"""

from confseq.cs_plots import *
from confseq.betting_strategies import *
from confseq.betting import *
from confseq.conjmix_bounded import conjmix_empbern_cs
from confseq.predmix import *
from scipy.stats import binom, beta

alpha = 0.05
n = 10000

cs_list = [
    ConfseqToPlot(
        lambda x: predmix_hoeffding_cs(x, alpha=alpha, running_intersection=False),
        "PM-H",
        "tab:orange",
        "-.",
    ),
    ConfseqToPlot(
        lambda x: predmix_empbern_cs(
            x, truncation=0.5, alpha=alpha, running_intersection=False
        ),
        "PM-EB",
        "tab:blue",
        "--",
    ),
    ConfseqToPlot(
        lambda x: betting_cs(
            x,
            breaks=1000,
            trunc_scale=1 / 2,
            alpha=alpha,
            parallel=True,
            running_intersection=False,
        ),
        r"Hedged",
        "tab:green",
        "-",
    ),
    ConfseqToPlot(
        lambda x: conjmix_empbern_cs(x, v_opt=500 / 4, running_intersection=False),
        r"CM-EB",
        "tab:red",
        ":",
    ),
]


dgp_list = [
    DataGeneratingProcess(
        data_generator_fn=lambda: np.random.binomial(1, 0.5, n),
        dist_fn=lambda x: binom.pmf(x, 1, 0.5),
        mean=0.5,
        name="Bernoulli_0.5_",
        discrete=True,
        title="$X_i \sim$ Bernoulli(1/2)",
    ),
    DataGeneratingProcess(
        data_generator_fn=lambda: np.random.binomial(1, 0.1, n),
        dist_fn=lambda x: binom.pmf(x, 1, 0.1),
        mean=0.1,
        name="Bernoulli_0.1_",
        discrete=True,
        title="$X_i \sim$ Bernoulli(1/10)",
    ),
    DataGeneratingProcess(
        data_generator_fn=lambda: np.random.beta(10, 30, n),
        dist_fn=lambda x: beta.pdf(x, 10, 30),
        mean=1 / 4,
        name="Beta_10,_30_",
        discrete=False,
        title="$X_i \sim$ Beta(10, 30)",
    ),
]

plot_CSs(dgp_list=dgp_list, cs_list=cs_list, folder=".", nsim=5)
