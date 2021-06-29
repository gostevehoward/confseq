from confseq.conjmix_bounded import *


def test_nested_cs():
    # Check to make sure that both Hoeffding and empirical Bernstein
    # confidence sequences are nested sets with respect to alpha
    n = 10000
    x = np.random.binomial(1, 0.5, n)

    alphas = [0.01, 0.011, 0.5, 0.51, 0.1, 0.11]

    confseqs_h = map(
        lambda alpha: conjmix_hoeffding_cs(x, t_opt=500, alpha=alpha), alphas
    )
    lower_CSs, upper_CSs = zip(*confseqs_h)

    assert all([all(lower_CSs[i] <= lower_CSs[i]) for i in range(len(alphas) - 1)])
    assert all([all(upper_CSs[i] >= upper_CSs[i]) for i in range(len(alphas) - 1)])

    confseqs_eb = map(
        lambda alpha: conjmix_empbern_cs(x, v_opt=500 / 4, alpha=alpha), alphas
    )
    lower_CSs, upper_CSs = zip(*confseqs_eb)

    assert all([all(lower_CSs[i] <= lower_CSs[i]) for i in range(len(alphas) - 1)])
    assert all([all(upper_CSs[i] >= upper_CSs[i]) for i in range(len(alphas) - 1)])
