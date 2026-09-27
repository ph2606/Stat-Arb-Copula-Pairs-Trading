"""Build publication-quality figures directly from the saved research outputs.

Run from any directory: python scripts/build_figures.py
All PnL curves use arithmetic sums over fixed initial capital; none compound.
"""

from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
PAIRS = [("AAPL", "GOOG"), ("IBM", "SPY"), ("DIA", "SPY")]
PAIR_NAMES = ["-".join(pair) for pair in PAIRS]
FAMILIES = ["Gaussian", "Student-t", "Clayton", "Frank", "Gumbel", "CFG", "CtG"]
NAVY, TEAL, GOLD, SLATE = "#18334D", "#147D88", "#C58B35", "#798AA0"
COLORS = {"Selected": NAVY, "Best-single": TEAL, "Distance": GOLD, "Cointegration": SLATE}
LABELS = {"Selected": "Selected copula", "Best-single": "Best single copula",
          "Distance": "Distance", "Cointegration": "Cointegration"}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.titlepad": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#BCC6CF", "axes.labelcolor": NAVY,
        "text.color": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
        "grid.color": "#DDE4EA", "grid.linewidth": .65,
        "axes.axisbelow": True, "legend.frameon": False,
        "legend.fontsize": 9, "lines.linewidth": 1.55,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=210, bbox_inches="tight", pad_inches=.14)
    plt.close(fig)
    print(f"Created {name}.png and {name}.pdf", flush=True)


def dates_axis(ax, *, short=False):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator) if short else mdates.DateFormatter("%Y"))
    ax.grid(axis="y")
    ax.margins(x=.01)


def percentage_axis(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.axhline(0, color="#ABB7C3", lw=.8, zorder=0)


def capital_barrier(ax):
    if ax.get_ylim()[0] <= -100:
        ax.axhline(-100, color="#AB3448", ls="--", lw=1.2)
        ax.text(.99, -100, "Initial capital exhausted", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="#AB3448",
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 1.5})


def read_table(name, dates=()):
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing. Run scripts/run_analysis.py before building figures.")
    frame = pd.read_csv(path, parse_dates=list(dates))
    if frame.empty:
        raise ValueError(f"{path.name} has no observations")
    return frame


def load_prices():
    symbols = dict.fromkeys(symbol for pair in PAIRS for symbol in pair)
    prices = pd.concat({s: pd.read_csv(ROOT / "data" / f"{s}.csv", index_col="Date", parse_dates=True)["Adj Close"]
                        for s in symbols}, axis=1)
    prices = prices.loc["2006-01-01":"2025-12-31"]
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Figures require complete, positive adjusted prices")
    return prices


def context(prices):
    returns = 100 * np.log(prices).diff().dropna()
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.0), layout="constrained")
    for row, (first, second) in enumerate(PAIRS):
        ax, scatter = axes[row]
        normalized = 100 * prices[[first, second]] / prices[[first, second]].iloc[0]
        for symbol, color in ((first, NAVY), (second, TEAL)):
            ax.plot(normalized.index, normalized[symbol], color=color, label=symbol)
        ax.set_yscale("log")
        ax.set_title(f"{first} / {second}: adjusted price paths")
        ax.set_ylabel("First 2006 close = 100\n(log scale)")
        ax.legend(loc="upper left", ncol=2)
        dates_axis(ax)
        scatter.scatter(returns[first], returns[second], s=7, alpha=.25, color=TEAL,
                        linewidths=0, rasterized=True)
        limit = np.ceil(np.max(np.abs(returns[[first, second]].to_numpy())))
        scatter.set(xlim=(-limit, limit), ylim=(-limit, limit),
                    xlabel=f"{first} daily log return (%)", ylabel=f"{second} daily log return (%)")
        scatter.axhline(0, color="#B5C0CA", lw=.7)
        scatter.axvline(0, color="#B5C0CA", lw=.7)
        scatter.grid(alpha=.5)
        correlation = returns[first].corr(returns[second])
        scatter.set_title(f"Daily returns: Pearson correlation {correlation:.2f}")
    save(fig, "context")


def dependence():
    cache = json.loads((RESULTS / "fit_cache" / "2025-01-01.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    for ax, (first, second) in zip(axes, PAIRS):
        data = cache["pairs"][f"{first}-{second}"]
        model = max(data["models"], key=lambda name: data["models"][name]["loglik"])
        u, v = data["train_u"], data["train_v"]
        tau = kendalltau(u, v).statistic
        ax.scatter(u, v, s=14, alpha=.55, color=TEAL, linewidths=0, rasterized=True)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"{first}: formation empirical PIT",
               ylabel=f"{second}: formation empirical PIT", aspect="equal")
        ax.set_title(f"{first} / {second}\n{model} selected; Kendall τ = {tau:.2f}")
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.grid(alpha=.5)
    fig.suptitle("Formation dependence for January–June 2025 trading", fontsize=14, fontweight="bold")
    save(fig, "dependence")


def pair_equity(daily):
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.5), sharex=True, layout="constrained")
    for ax, pair in zip(axes, PAIR_NAMES):
        for method in COLORS:
            data = daily.loc[(daily.pair == pair) & (daily.method == method)].sort_values("Date")
            ax.plot(data.Date, 100 * data.return_net.cumsum(), label=LABELS[method], color=COLORS[method])
        ax.set_title(pair.replace("-", " / "), loc="left")
        ax.set_ylabel("Net PnL / initial capital")
        dates_axis(ax)
        percentage_axis(ax)
        capital_barrier(ax)
    axes[0].legend(ncol=4, loc="upper left")
    fig.suptitle("Pair strategies, 2006–2025: uncapped fixed-stake PnL\n$100,000 per pair; 5 bp per leg fill; arithmetic cumulative PnL\nContinuation below −100% requires added capital",
                 fontsize=12, fontweight="bold")
    save(fig, "pair_equity")


def portfolio_equity(portfolio):
    methods = ("Selected", "Best-single", "Distance")
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.5), sharex=True, layout="constrained")
    for ax, method in zip(axes, methods):
        data = portfolio.loc[portfolio.method == method].sort_values("Date")
        ax.plot(data.Date, 100 * data.return_gross.cumsum(), color=SLATE, ls="--", label="Before costs")
        ax.plot(data.Date, 100 * data.return_net.cumsum(), color=COLORS[method], label="After 5 bp per fill")
        ax.set_title(LABELS[method], loc="left")
        ax.set_ylabel("Portfolio PnL / initial capital")
        ax.legend(loc="upper left", ncol=2)
        dates_axis(ax)
        percentage_axis(ax)
        capital_barrier(ax)
    fig.suptitle("Portfolio cost drag, 2006–2025: uncapped fixed-stake PnL\n$300,000 initial capital; equal fixed stakes; arithmetic cumulative PnL\nContinuation below −100% requires added capital",
                 fontsize=12, fontweight="bold")
    save(fig, "portfolio_equity")


def shade_positions(ax, dates, position):
    values = np.asarray(position, dtype=int)
    edges = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)]
    for left, right in zip(edges[:-1], edges[1:]):
        if values[left] == 0:
            continue
        end = dates.iloc[right] if right < len(dates) else dates.iloc[-1]
        ax.axvspan(dates.iloc[left], end, color=TEAL if values[left] == 1 else GOLD,
                   alpha=.13, lw=0, zorder=0)


def signal_example(daily, signals):
    sig = signals.loc[(signals.pair == "AAPL-GOOG") & (signals.window == "2020-01-01")].sort_values("Date")
    ledger = daily.loc[(daily.pair == "AAPL-GOOG") & (daily.method == "Selected")
                       & (daily.window == "2020-01-01")].sort_values("Date")
    if sig.empty or not np.array_equal(sig.Date.to_numpy(), ledger.Date.to_numpy()):
        raise ValueError("2020 H1 signal example and ledger must align exactly")
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.5), sharex=True, layout="constrained")
    for ax in axes:
        shade_positions(ax, ledger.Date, ledger.position)
        dates_axis(ax, short=True)
    axes[0].plot(sig.Date, sig.flag1, color=NAVY, label="AAPL cumulative flag")
    axes[0].plot(sig.Date, sig.flag2, color=TEAL, label="GOOG cumulative flag")
    axes[0].axhline(0, color=SLATE, lw=.8)
    for value in (-.2, .2):
        axes[0].axhline(value, color=GOLD, ls=":", lw=1.1)
    axes[0].set_ylabel("Cumulative mispricing flag")
    axes[0].legend(loc="upper left", ncol=2)
    axes[1].plot(ledger.Date, 100 * ledger.return_net.cumsum(), color=NAVY)
    axes[1].set_ylabel("Net PnL / initial capital")
    percentage_axis(axes[1])
    axes[1].legend(handles=[Patch(color=TEAL, alpha=.2, label="Holding long AAPL / short GOOG"),
                            Patch(color=GOLD, alpha=.2, label="Holding short AAPL / long GOOG")],
                   loc="upper left", ncol=2)
    family = sig.family.iloc[0]
    fig.suptitle(f"AAPL / GOOG, January–June 2020: {family} copula\nEntry barriers ±0.20; either flag exits at zero; execution one close later",
                 fontsize=13, fontweight="bold")
    save(fig, "signal_example")


def model_selection(fits):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), sharey=True, layout="constrained")
    x = np.arange(len(FAMILIES))
    for ax, pair in zip(axes, PAIR_NAMES):
        data = fits.loc[fits.pair == pair]
        for column, label, offset, color in (("selected", "Raw likelihood", -.19, NAVY),
                                              ("selected_bic", "BIC", .19, TEAL)):
            indicator = data[column]
            if indicator.dtype != bool:
                indicator = indicator.astype(str).str.lower().eq("true")
            counts = data.loc[indicator, "family"].value_counts().reindex(FAMILIES, fill_value=0)
            ax.bar(x + offset, 100 * counts.to_numpy() / data.window.nunique(), width=.36,
                   color=color, label=label)
        ax.set_title(pair.replace("-", " / "))
        ax.set_xticks(x, FAMILIES, rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis="y")
    axes[0].set_ylabel("Share of 40 formation windows")
    axes[0].legend(loc="upper left")
    fig.suptitle("Model selection: fit quality and complexity penalty", fontsize=14, fontweight="bold")
    save(fig, "model_selection")


def threshold_sensitivity(sensitivity):
    grouped = sensitivity.groupby(["pair", "threshold", "exit_rule"])[["return_net", "days"]].sum()
    grouped["annual_mean_pct"] = 100 * 252 * grouped.return_net / grouped.days
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), sharex=True, layout="constrained")
    for ax, pair in zip(axes, PAIR_NAMES):
        for rule, label, color, marker in (("either", "Either flag at zero", NAVY, "o"),
                                           ("both", "Both flags across zero", TEAL, "s")):
            values = grouped.xs((pair, rule), level=("pair", "exit_rule")).sort_index()
            ax.plot(values.index, values.annual_mean_pct, color=color, label=label, marker=marker, ms=3.4)
        ax.axvline(.2, color=GOLD, ls=":", lw=1.2, label="Main threshold 0.20")
        ax.set_title(pair.replace("-", " / "))
        ax.set_xlabel("Absolute entry threshold")
        ax.set_xticks([.1, .2, .3, .4, .5])
        ax.grid(axis="y")
        percentage_axis(ax)
    axes[0].set_ylabel("Annualized mean net PnL / initial capital")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Preset threshold and exit-rule sensitivity, 2006–2025\nSame formation fits; 5 bp per fill; thresholds are not selected from these results",
                 fontsize=12.5, fontweight="bold")
    save(fig, "threshold_sensitivity")


def cost_sensitivity(costs):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    portfolio = costs.loc[(costs.pair == "Portfolio") & (costs.borrow_bps == 0)]
    for method in ("Selected", "Best-single", "Distance"):
        data = portfolio.loc[portfolio.method == method].sort_values("cost_bps")
        axes[0].plot(data.cost_bps, data.annual_mean_pct, marker="o", ms=4,
                     color=COLORS[method], label=LABELS[method])
    for pair, color in zip(PAIR_NAMES, (NAVY, TEAL, GOLD)):
        data = costs.loc[(costs.pair == pair) & (costs.method == "Selected")
                         & (costs.borrow_bps == 0)].sort_values("cost_bps")
        axes[1].plot(data.cost_bps, data.annual_mean_pct, marker="o", ms=4, color=color,
                     label=pair.replace("-", " / "))
    for ax in axes:
        ax.axvline(5, color=SLATE, ls=":", lw=1)
        ax.set_xlabel("Trading cost per dollar traded, per fill (bp)")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(axis="y")
        percentage_axis(ax)
        ax.legend(loc="best")
    axes[0].set_title("Three-pair portfolio")
    axes[1].set_title("Selected copula, by pair")
    axes[0].set_ylabel("Annualized mean net PnL / initial capital")
    fig.suptitle("Transaction-cost sensitivity, 2006–2025\nFixed signals and entry stakes; borrowing cost set to zero",
                 fontsize=13, fontweight="bold")
    save(fig, "cost_sensitivity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=["context", "dependence", "pair_equity", "portfolio_equity",
                        "signal_example", "model_selection", "threshold_sensitivity", "cost_sensitivity"])
    selected = set(parser.parse_args().only or ["context", "dependence", "pair_equity", "portfolio_equity",
                   "signal_example", "model_selection", "threshold_sensitivity", "cost_sensitivity"])
    style()
    if "context" in selected:
        context(load_prices())
    if "dependence" in selected:
        dependence()
    if selected & {"pair_equity", "signal_example"}:
        daily = read_table("daily_ledger.csv", dates=["Date"])
        if "pair_equity" in selected:
            pair_equity(daily)
        if "signal_example" in selected:
            signal_example(daily, read_table("selected_signals.csv", dates=["Date"]))
    if "portfolio_equity" in selected:
        portfolio_equity(read_table("portfolio_daily.csv", dates=["Date"]))
    if "model_selection" in selected:
        model_selection(read_table("copula_fits.csv"))
    if "threshold_sensitivity" in selected:
        threshold_sensitivity(read_table("sensitivity_windows.csv"))
    if "cost_sensitivity" in selected:
        cost_sensitivity(read_table("cost_sensitivity.csv"))


if __name__ == "__main__":
    main()
