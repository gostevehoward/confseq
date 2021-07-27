from confseq.betting import *
import matplotlib
import matplotlib.pyplot as plt
import os


class ConfseqToPlot:
    """
    Confidence sequences to plot in the `plot_cs` function.

    Attributes
    ----------
    name, String
        Name of the confidence sequence

    cs_fn, function : array-like of reals -> (array-like of reals, array-like of reals)
        The confidence sequence function which accepts the list of
        real numbers and outputs a lower- and upper-confidence bound

    color, String
        The matplotlib-compatible name of the color to use in the plot

    dashes, [int]
        The matplotlib-compatible description of dashes to use in the plot

    linestyle, String
        The matplotlib-compatible linestyle to use (overwrites dashes)
    """

    def __init__(self, cs_fn, name, color, linestyle=None, dashes=None):
        self.name = name
        self.cs_fn = cs_fn
        self.color = color
        if linestyle == "-":
            dashes = []
        elif linestyle == "--":
            dashes = [4, 1]
        elif linestyle == "-.":
            dashes = [6, 2, 1, 2]
        elif linestyle == ":":
            dashes = [1, 2]
        self.dashes = dashes if dashes is not None else []


class DataGeneratingProcess:
    """
    Data-generating process to use for `plot_cs`

    Attributes
    ----------
    name, String
        Name of the dgp

    data_generator_fn, function : () -> array-like of [0, 1]-bounded reals
        Data-generator function which takes no parameters and outputs a
        numpy array of random [0, 1]-bounded real numbers according to some
        distribution.

    discrete, boolean or None
        Should the with-replacement distribution be plotted as a discrete
        distribution (pmf) or continuous (pdf)? If sampling without
        replacement, leave as None, and the empirical distribution will
        be plotted.

    dist_fn, univariate function or None
        Either the probability density function (pdf) or probability mass
        function (pmf) for the data. If sampling without replacement, leave
        as None, since there is no superpopulation pmf/pdf.

    mean, [0, 1]-valued real or None
        The mean of the with-replacement distribution. If sampling without
        replacement, leave as None since the mean can be determined from
        `data_generator_fn()`.

    title, String or None
        Title to be used for this dgp in the plots. If left as None, `title`
        will be set to `name`. Importantly, `name` will be used for file output
        so it needs to be rather simple, but `title` can be complex (e.g.
        including LaTeX).

    WoR, boolean
        Are the data being sampled without replacement from a finite population
        rather than with replacement from a superpopulation?
    """

    def __init__(
        self,
        name,
        data_generator_fn,
        discrete=None,
        dist_fn=None,
        mean=None,
        title=None,
        WoR=False,
    ):
        self.data_generator_fn = data_generator_fn
        self.dist_fn = dist_fn
        self.mean = mean if mean is not None else np.mean(data_generator_fn())
        self.name = name
        self.discrete = discrete
        self.title = title if title is not None else name
        self.WoR = WoR


def plot_cs(
    data_generator,
    cs_list,
    name,
    pdf,
    folder,
    times=None,
    display_start=1,
    time_uniform=False,
    nsim=1,
    title="",
    null_mean=math.nan,
    hist_discrete=True,
    log_scale=True,
    include_density=True,
    include_legend=True,
    legend_outside_plot=False,
    legend_columns=None,
    bbox_to_anchor=(-0.75, -0.7),
):

    computation_time_dict = {}
    miscoverage_dict = {}

    if times is None:
        N = len(data_generator())
        t = np.arange(1, N + 1)
    else:
        N = len(times)
        t = times

    # Dictionary whose key is the name of the CS, and the value
    # is a (nsim by N) matrix containing nsim simulations of
    # lower confidence sequences.
    l_mtx_dict = {}
    u_mtx_dict = {}

    # Populate the dictionaries
    for cs_obj in cs_list:
        l_mtx_dict[cs_obj.name] = np.zeros((nsim, N))
        u_mtx_dict[cs_obj.name] = np.zeros((nsim, N))
        computation_time_dict[cs_obj.name] = np.repeat(math.inf, nsim)
        miscoverage_dict[cs_obj.name] = np.repeat(1, nsim)

    for i in np.arange(0, nsim):
        x = data_generator()
        for cs_obj in cs_list:
            start = time()
            l, u = cs_obj.cs_fn(x)
            end = time()
            computation_time_dict[cs_obj.name][i] = end - start
            miscoverage_dict[cs_obj.name][i] = any(
                np.logical_or(l > null_mean, u < null_mean)
            )
            l_mtx_dict[cs_obj.name][i, :] = l
            u_mtx_dict[cs_obj.name][i, :] = u

    grid = np.arange(0, 1.01, step=0.01)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 13
    if include_density:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        # Index of the density plot in the 'axs' object
        axs_dens_idx = 0
        # Index of the confidence sequence plot in the 'axs' object
        axs_cs_idx = 1
        # Index of the width plot in the 'axs' object
        axs_width_idx = 2

        if hist_discrete is None:
            axs[axs_dens_idx].hist(data_generator())
        elif hist_discrete:
            axs[axs_dens_idx].bar(grid, pdf(grid), width=0.1)
            axs[axs_dens_idx].set_ylabel("pmf")
        else:
            axs[axs_dens_idx].plot(grid, pdf(grid))
            axs[axs_dens_idx].set_ylabel("pdf")

    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs_cs_idx = 0
        # Index of the width plot in the 'axs' object
        axs_width_idx = 1

    for cs_obj in cs_list:
        l_avg = l_mtx_dict[cs_obj.name].mean(axis=0)
        u_avg = u_mtx_dict[cs_obj.name].mean(axis=0)

        axs[axs_cs_idx].plot(
            t[(display_start - 1) :],
            l_avg[(display_start - 1) :],
            color=cs_obj.color,
            dashes=cs_obj.dashes,
        )
        axs[axs_cs_idx].plot(
            t[(display_start - 1) :],
            u_avg[(display_start - 1) :],
            color=cs_obj.color,
            dashes=cs_obj.dashes,
            label=cs_obj.name,
        )
        axs[axs_width_idx].plot(
            t[(display_start - 1) :],
            u_avg[(display_start - 1) :] - l_avg[(display_start - 1) :],
            color=cs_obj.color,
            dashes=cs_obj.dashes,
            label=cs_obj.name,
        )

    axs[axs_cs_idx].axhline(null_mean, linestyle=":", color="gray")
    axs[axs_width_idx].set_ylabel("CI width")
    if log_scale:
        axs[axs_cs_idx].set_xscale("log")
        axs[axs_width_idx].set_xscale("log")

    if time_uniform:
        axs[axs_cs_idx].set_ylabel("Confidence sequence")
        axs[axs_cs_idx].set_xlabel("time $t$")
        axs[axs_width_idx].set_ylabel("CS width")
        axs[axs_width_idx].set_xlabel("time $t$")
    else:
        axs[axs_cs_idx].set_ylabel("Confidence interval")
        axs[axs_cs_idx].set_xlabel("sample size $n$")
        axs[axs_width_idx].set_ylabel("CI width")
        axs[axs_width_idx].set_xlabel("sample size $n$")

    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=6)
    axs[axs_cs_idx].xaxis.set_major_locator(locmaj)
    axs[axs_width_idx].xaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0, subs=np.arange(0.1, 1, step=0.1), numticks=12
    )
    axs[axs_cs_idx].xaxis.set_minor_locator(locmin)
    axs[axs_width_idx].xaxis.set_minor_locator(locmin)
    axs[axs_cs_idx].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[axs_width_idx].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.tight_layout()
    if include_legend:
        if legend_outside_plot:
            if legend_columns is None:
                legend_columns = int(len(cs_list) / 2)
            axs[axs_cs_idx].legend(
                loc="lower left",
                bbox_to_anchor=bbox_to_anchor,
                ncol=legend_columns,
            )
        else:
            axs[axs_width_idx].legend(loc="best")

    """
    Strictly speaking this should occur before plt.tight_layout() but
    to remain consistent with the previous plots I'm going to leave it after.
    """
    if include_density:
        axs[axs_dens_idx].set_title(title)

    if time_uniform:
        plt.savefig(
            os.path.join(folder, name + "_time-uniform.pdf"), bbox_inches="tight"
        )
    else:
        plt.savefig(os.path.join(folder, name + "_fixed-time.pdf"), bbox_inches="tight")

    # Print out computation times
    for name in computation_time_dict:
        print(
            str(name)
            + " took an average of "
            + str(np.mean(computation_time_dict[name]))
            + " seconds"
        )
        print(
            str(name)
            + " miscovered in "
            + str(np.sum(miscoverage_dict[name]))
            + " out of "
            + str(nsim)
            + " experiments"
        )


def plot_CSs(
    dgp_list,
    cs_list,
    folder,
    display_start=1,
    times=None,
    time_uniform=False,
    nsim=1,
    log_scale=True,
    include_density=True,
    legend_on_last_only=False,
    legend_outside_plot=False,
    legend_columns=None,
    bbox_to_anchor=(-0.75, -0.7),
):
    for dgp in dgp_list:
        if not legend_on_last_only or dgp_list[-1] == dgp:
            plot_cs(
                dgp.data_generator_fn,
                cs_list,
                folder=folder,
                nsim=nsim,
                pdf=dgp.dist_fn,
                title=dgp.title,
                name=dgp.name,
                display_start=display_start,
                times=times,
                null_mean=dgp.mean,
                hist_discrete=dgp.discrete,
                time_uniform=time_uniform,
                log_scale=log_scale,
                include_density=include_density,
                include_legend=True,
                legend_outside_plot=legend_outside_plot,
                legend_columns=legend_columns,
                bbox_to_anchor=bbox_to_anchor,
            )
        else:
            plot_cs(
                dgp.data_generator_fn,
                cs_list,
                folder=folder,
                nsim=nsim,
                pdf=dgp.dist_fn,
                title=dgp.title,
                name=dgp.name,
                display_start=display_start,
                times=times,
                null_mean=dgp.mean,
                hist_discrete=dgp.discrete,
                time_uniform=time_uniform,
                log_scale=log_scale,
                include_density=include_density,
                include_legend=False,
                legend_outside_plot=legend_outside_plot,
                legend_columns=legend_columns,
                bbox_to_anchor=bbox_to_anchor,
            )
