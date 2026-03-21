# /// script
# dependencies = [
#     "marimo",
#     "plotnine==0.15.3",
#     "polars==1.38.1",
#     "pyarrow==23.0.1",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotnine as gg
    import pyarrow

    return gg, mo, pl


@app.cell
def _(mo):
    mo.md(
        r"""
    # Data Visualization Assignment

    In this notebook are shown the plots I'm using in my PHD research project on MIP Formulations for Delete-Free AI Planning

    ## Project Abstract
    We investigate existing Mixed Integer Programming (MIP) formulations for cost-optimal delete-free STRIPS Planning: these models are built by enforcing acyclicity in the underlying causal relation graphs using time labeling and vertex elimination methods. We then propose two new approaches to modeling acyclicity, one based on dynamically identifying subtour elimination constraints, and the other based on disjunctive landmark constraints. In addition, we propose to warm start the models with the landmarks computed by the LM-cut heuristic, and describe a simple greedy primal heuristic to provide a starting feasible solution to the MIP solver. Our results demonstrate that the proposed techniques outperform the current state of the art, both in space and time efficiency.

    ## Repository of the project
    https://github.com/Zanzibarr/MIPxHPLUS
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Downloading and parsing data
    """
    )
    return


@app.cell
def _(mo, pl):
    # SIMPLE FUNCTIONS FOR READING/DOWNLOADING DATA
    git_repo = "https://raw.githubusercontent.com/Zanzibarr/DataVisualization_Assignment/refs/heads/main"
    data_folder = git_repo + "/data/"
    images_folder = git_repo + "/images/"

    def load_data(name: str) -> pl.DataFrame:
        return (
            pl.read_csv(
                data_folder + name,
                schema_overrides={
                    "Unit_Costs": pl.Boolean,
                    "Initial_LB": pl.Float64,
                    "Root_LB": pl.Float64,
                    "Final_LB": pl.Float64,
                    "Initial_UB": pl.Float64,
                    "Final_UB": pl.Float64,
                    "N_RCB_Calls": pl.Int64,
                    "N_CCB_Calls": pl.Int64,
                    "N_CL_It": pl.Int64,
                    "CL_Time": pl.Float64,
                },
            )
            .filter(pl.col("Status").is_in([0, 1, 2, 3]))
            .with_columns(
                Status=pl.col("Status").is_in([0, 1]),
                Time=pl.when(pl.col("Status").is_in([0, 1]))
                .then(pl.col("Time") - pl.col("Parsing_Time"))
                .otherwise(pl.lit(900)),
            )
            .rename({"N_Nodes": "Nodes"})
        )

    def load_image(name: str):
        return mo.image(src=images_folder + name)

    return data_folder, load_data, load_image


@app.cell
def _(load_data, pl):
    # LOAD ALL DATASETS
    # TIME LABELING
    tl = load_data("run_tl_e__250611.csv").with_columns(Model=pl.lit("TL"))
    tl_h = load_data("run_tl_e_h__250609.csv").with_columns(Model=pl.lit("TL_h"))
    tl_hLM = load_data("run_tl_e+lmcut_h__251112.csv").with_columns(
        Model=pl.lit("TL_hLM")
    )
    # VERTEX ELIMINATION
    ve = load_data("run_ve_e__250612.csv").with_columns(Model=pl.lit("VE"))
    ve_h = load_data("run_ve_e_h__250608.csv").with_columns(Model=pl.lit("VE_h"))
    ve_hLM = load_data("run_ve_e+lmcut_h__251113.csv").with_columns(
        Model=pl.lit("VE_hLM")
    )
    # SUBTOUR ELIMINATION CONSTRAINTS
    sec = load_data("run_sec_e__251121.csv").with_columns(Model=pl.lit("SEC"))
    sec_h = load_data("run_sec_e_h__251123.csv").with_columns(Model=pl.lit("SEC_h"))
    sec_hLM = load_data("run_sec_e+lmcut_h__251115.csv").with_columns(
        Model=pl.lit("SEC_hLM")
    )
    # LANDMARK CONSTRAINTS
    lmc = load_data("run_comp_e__251120.csv").with_columns(Model=pl.lit("LMC"))
    lmc_h = load_data("run_comp_e_h__251122.csv").with_columns(Model=pl.lit("LMC_h"))
    lmc_hLM = load_data("run_comp_e+lmcut_h__251112.csv").with_columns(
        Model=pl.lit("LMC_hLM")
    )
    flm_hLM = load_data("run_base_min_1k_noworse_sortvf_lh10__260204.csv").with_columns(
        Model=pl.lit("LMCf_hLM")
    )
    # LANDMARK CONSTRAINTS THROUGH LMCUT
    run_best = load_data("run_best.csv").with_columns(Model=pl.lit("LMC_b"))
    return (
        flm_hLM,
        lmc,
        lmc_h,
        lmc_hLM,
        run_best,
        sec,
        sec_hLM,
        tl,
        tl_h,
        tl_hLM,
        ve,
        ve_h,
        ve_hLM,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Data manipulation
    """
    )
    return


@app.cell
def _(pl):
    """
    Divide data in different Time Brackets: each time bracket is computed by taking the minimum running time among different models.
    Time Bracket computed: [0,1),[1,10),[10,100),[100,900),[900,+inf)
    On top of that, the Category is computed, which is computed by a logical operation (AND/OR) over the solvability of different models.
    Categories computed: all-solvable (AND), solvable (OR), non-solvable (¬OR)
    """

    def categorize(data: pl.DataFrame) -> pl.DataFrame:

        # Time Brackets
        data = data.with_columns(
            Time_Bracket=pl.when(pl.col("Time").min().over("Instance") < 1)
            .then(pl.lit("[0,1)"))
            .when(pl.col("Time").min().over("Instance") < 10)
            .then(pl.lit("[1,10)"))
            .when(pl.col("Time").min().over("Instance") < 100)
            .then(pl.lit("[10,100)"))
            .when(pl.col("Time").min().over("Instance") < 900)
            .then(pl.lit("[100,900)"))
            .otherwise(pl.lit("[900,+inf)"))
        )

        # Categories
        data = data.with_columns(
            Category=(
                pl.when(pl.col("Status").all().over("Instance"))
                .then(pl.lit("all-solvable"))
                .when(pl.col("Status").any().over("Instance"))
                .then(pl.lit("solvable"))
                .otherwise(pl.lit("non-solvable"))
            )
        )

        return data

    """
    Compare data of different runs according to a metric
    Shifted Geometric Means (shift = 1) are computed among the same Time Bracket or Category, then the ratio between the SGM of all methods over that of the baseline is computed (so to normalize the results between different Time Brackets and Categories)
    """

    def compare_data(
        data: pl.DataFrame, metric: str, base_alias: str, run_aliases: list[str]
    ) -> pl.DataFrame:

        # Create aggregated groups
        all_data = (
            data.group_by("Model")
            .agg(shifted_geom_mean=(pl.col(metric) + 1).log().mean().exp() - 1)
            .with_columns(Group=pl.lit("all"))
            .select(["Group", "Model", "shifted_geom_mean"])
        )

        solvable_data = (
            data.filter(pl.col("Category") != pl.lit("non-solvable"))
            .group_by("Model")
            .agg(shifted_geom_mean=(pl.col(metric) + 1).log().mean().exp() - 1)
            .with_columns(Group=pl.lit("solvable"))
            .select(["Group", "Model", "shifted_geom_mean"])
        )

        all_solvable_data = (
            data.filter(pl.col("Category") == "all-solvable")
            .group_by("Model")
            .agg(shifted_geom_mean=(pl.col(metric) + 1).log().mean().exp() - 1)
            .with_columns(Group=pl.lit("all-solvable"))
            .select(["Group", "Model", "shifted_geom_mean"])
        )

        # Time bracket groups
        time_bracket_data = (
            data.group_by("Time_Bracket", "Model")
            .agg(shifted_geom_mean=(pl.col(metric) + 1).log().mean().exp() - 1)
            .rename({"Time_Bracket": "Group"})
            .select(["Group", "Model", "shifted_geom_mean"])
        )

        # Combine all groups
        dfs = [all_data, solvable_data, all_solvable_data, time_bracket_data]
        dfs = [
            df.with_columns(
                pl.col("Group").cast(pl.String), pl.col("Model").cast(pl.String)
            )
            for df in dfs
        ]

        # Combine all groups
        data = pl.concat(dfs)

        data = data.join(
            (
                data.filter(pl.col("Model") == base_alias)
                .select(["Group", "shifted_geom_mean"])
                .rename({"shifted_geom_mean": "baseline_sgm"})
            ),
            on="Group",
        ).with_columns(sgm_ratio=pl.col("shifted_geom_mean") / pl.col("baseline_sgm"))

        # Set order (with categorical value types) so that plots have results in order
        order_map = {
            base_alias: 0,
            **{alias: i + 1 for i, alias in enumerate(run_aliases)},
        }

        # Define group order
        group_order = [
            "all",
            "solvable",
            "all-solvable",
            "[0,1)",
            "[1,10)",
            "[10,100)",
            "[100,900)",
            "[900,+inf)",
        ]
        group_order_map = {g: i for i, g in enumerate(group_order)}

        data = (
            data.with_columns(
                pl.col("Model")
                .replace(order_map)
                .alias(
                    "model_order"
                ),  # Associate to each line a number for the model order
                pl.col("Group")
                .replace(group_order_map)
                .alias(
                    "group_order"
                ),  # Associate to each line a number for the group order
            )
            .sort(
                by=["group_order", "model_order"]
            )  # Sort first based on group order, then on model order
            .drop(["model_order", "group_order"])  # Remove unnecessary columns
        )

        # pl.Categorical maps the data type into integers... this is more memory efficient and when plotting the order is respected because of the mapping (the order used when the mapping is created is alphabetically ordered in the Categorical data type, so the plotting library has no reason to change it)
        data = data.with_columns(
            pl.col("Model").cast(pl.Categorical), pl.col("Group").cast(pl.Categorical)
        ).with_columns(Model_Class=pl.col("Model").cast(pl.String).str.slice(0, 2))

        return data

    return categorize, compare_data


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Plotting
    """
    )
    return


@app.cell
def _(categorize, compare_data, data_folder, gg, pl):
    """
    Produce a bar plot over the categories and time brackets produced by the comparison between different runs.
    If the comparisons are made among different metrics, a faceted plot is returned.
    """

    def bar_plot(
        data: pl.DataFrame,
        base_alias: str,
        metric: str | list[str],
        facet_on: str = "Metric",
    ):
        # Normalize metric to always be a list for uniform handling below
        metrics = [metric] if isinstance(metric, str) else metric
        use_facets = len(metrics) > 1

        data = data.filter(
            (pl.col("Model") != pl.lit(base_alias))
            & (pl.col("Group") != pl.lit("[900,+inf)"))
        )

        subtitle = (
            f"{', '.join(metrics)} Ratio by Model (w.r.t. {base_alias})"
            if use_facets
            else f"{metrics[0]} Ratio by Model (w.r.t. {base_alias})"
        )

        plot = (
            gg.ggplot(
                data, gg.aes(x="Group", y="sgm_ratio", fill="Model", group="Model")
            )
            + gg.geom_bar(stat="identity", position="dodge", width=0.7)
            +
            # Horizontal reference line at ratio=1 (i.e. on-par with the baseline)
            gg.geom_hline(
                gg.aes(yintercept=1, linetype=f'"{base_alias}"'),
                color="red",
                size=0.8,
                alpha=0.5,
            )
            + gg.scale_linetype_manual(
                name="Reference", values={f"{base_alias}": "dashed"}
            )
            + gg.labs(
                title="Model Performance Comparison",
                subtitle=subtitle,
                x="Category / Time Bracket",
                y="SGM Ratio" if use_facets else f"{metrics[0]} Ratio",
                fill="Model",
            )
            + gg.scale_fill_brewer(type="qual", palette="Set2")
            + gg.theme_minimal()
            + gg.theme(
                figure_size=(10, 6),
                plot_title=gg.element_text(ha="center"),
                plot_subtitle=gg.element_text(ha="center"),
                axis_text_x=gg.element_text(rotation=45, ha="right"),
            )
        )

        if use_facets:
            plot = plot + gg.facet_wrap(facet_on, scales="free_y")

        return plot

    """
    Same as bar_plot, but handles multiple baselines — one per model class.
    Each class gets its own facet column, with its own reference line.
    """

    def bar_plot_multi_baseline(
        data: pl.DataFrame,
        class_baselines: dict[str, str],
        metric: list[str],
        class_order: list[str] | None = None,
    ):
        # Fall back to dict insertion order if no explicit ordering is provided
        ordered_classes = (
            class_order if class_order is not None else list(class_baselines.keys())
        )

        data = data.filter(
            # Exclude baseline models themselves — we only plot ratios of comparisons
            ~pl.col("Model").cast(pl.String).is_in(list(class_baselines.values()))
            & (pl.col("Group") != pl.lit("[900,+inf)"))
        ).with_columns(
            # Cast to Enum to enforce panel column order in facet_grid
            pl.col("Model_Class").cast(pl.Enum(ordered_classes))
        )

        # One hline per class panel, labeled with the baseline name for the legend
        hlines = pl.DataFrame(
            {
                "Model_Class": ordered_classes,
                "baseline_label": [class_baselines[c] for c in ordered_classes],
                "yintercept": [1.0] * len(ordered_classes),
            }
        ).with_columns(pl.col("Model_Class").cast(pl.Enum(ordered_classes)))

        return (
            gg.ggplot(
                data, gg.aes(x="Group", y="sgm_ratio", fill="Model", group="Model")
            )
            + gg.geom_bar(stat="identity", position="dodge", width=0.7)
            + gg.geom_hline(
                gg.aes(yintercept="yintercept", linetype="baseline_label"),
                data=hlines,
                color="red",
                size=0.8,
                alpha=0.5,
            )
            + gg.scale_linetype_manual(
                name="Reference", values={v: "dashed" for v in class_baselines.values()}
            )
            + gg.labs(
                title="Model Performance Comparison",
                subtitle=f"{', '.join(metric)} Ratio by Model (w.r.t. respective class baseline)",
                x="Category / Time Bracket",
                y="SGM Ratio",
                fill="Model",
            )
            + gg.scale_fill_brewer(type="qual", palette="Set2")
            + gg.theme_minimal()
            + gg.theme(
                figure_size=(10, 6),
                plot_title=gg.element_text(ha="center"),
                plot_subtitle=gg.element_text(ha="center"),
                axis_text_x=gg.element_text(rotation=45, ha="right"),
            )
            +
            # Rows = Metric (Time / Nodes), Columns = Model_Class (TL / VE / LM)
            gg.facet_grid("Metric", "Model_Class", scales="free_y")
        )

    """
    Wrapper that prepares and routes data to the appropriate bar plot function.
    Handles both the single-baseline and per-class-baseline cases.
    """

    def time_nodes_plot(
        data_list: list[pl.DataFrame],
        baseline: str,
        comparisons: list[str],
        facet_on: str = "Metric",
        class_baselines: dict[str, str] | None = None,
        class_order: list[str] | None = None,
    ):
        metric_order = ["Time", "Nodes"]

        if class_baselines is None:
            # Keep only instances that appear in all runs (inner join semantics)
            data = pl.concat(data_list).filter(
                pl.col("Model").count().over("Instance") == len(data_list)
            )
            categorized = categorize(data)

            compared = pl.concat(
                [
                    compare_data(categorized, m, baseline, comparisons).with_columns(
                        pl.lit(m).alias("Metric")
                    )
                    for m in metric_order
                ]
            ).with_columns(pl.col("Metric").cast(pl.Enum(metric_order)))

            return bar_plot(compared, baseline, metric_order, facet_on)

        else:
            data = pl.concat(data_list)
            frames = []

            for cls, cls_base in class_baselines.items():
                # NOTE: this assumes that all models belonging to a class have
                # names prefixed by the baseline name (e.g. "TL", "TL_h", "TL_hLM")
                cls_models = [cls_base] + [
                    m for m in comparisons if m.startswith(cls_base)
                ]
                cls_comparisons = cls_models[1:]

                # Keep only instances solved by every model in this class
                cls_data = categorize(
                    data.filter(
                        pl.col("Model").cast(pl.String).is_in(cls_models)
                    ).filter(
                        pl.col("Model").count().over("Instance") == len(cls_models)
                    )
                )

                for metric in metric_order:
                    frames.append(
                        compare_data(
                            cls_data, metric, cls_base, cls_comparisons
                        ).with_columns(pl.lit(metric).alias("Metric"))
                    )

            compared = pl.concat(frames).with_columns(
                pl.col("Metric").cast(pl.Enum(metric_order))
            )

            return bar_plot_multi_baseline(
                compared, class_baselines, metric_order, class_order
            )

    """
    Plot the primal gap of the greedy algorithm wrt the best known solutions
    """

    def gap_plot(data: pl.DataFrame):

        optimal_data = (
            pl.read_csv(data_folder + "best_known.csv")
            .filter(pl.col("Incumbent") < 1e20)
            .filter(pl.col("Incumbent") > 0)
        )
        ph_data = (
            categorize(data.select("Instance", "Status", "Initial_UB", "Nodes", "Time"))
            .filter(pl.col("Initial_UB") < 1e20)
            .filter(pl.col("Initial_UB") > 0)
        )

        time_order = ["[0,1)", "[1,10)", "[10,100)", "[100,900)", "[900,+inf)"]

        df = (
            ph_data.join(
                optimal_data.rename({"Problem": "Instance"}),
                on="Instance",
                how="inner",
            )
            .with_columns(
                [
                    (1 - pl.col("Incumbent") / pl.col("Initial_UB")).alias("gap"),
                    (pl.col("Nodes") + 1).alias("Nodes_plot"),
                    pl.col("Time_Bracket").cast(pl.Enum(time_order)),
                ]
            )
            .sort("gap")
            .with_row_index("rank")
        )

        present = [t for t in time_order if t in df["Time_Bracket"].unique()]
        palette = {
            "[0,1)": "#4fc3f7",
            "[1,10)": "#81c784",
            "[10,100)": "#e8c547",
            "[100,900)": "#ff8a65",
            "[900,+inf)": "#e57373",
        }
        color_values = {k: v for k, v in palette.items() if k in present}

        plot = (
            gg.ggplot(
                df, gg.aes(x="rank", y="gap", color="Time_Bracket", size="Nodes_plot")
            )
            + gg.geom_point(alpha=0.75, fill="none")
            + gg.scale_color_manual(values=color_values, name="Time Bracket (s)")
            + gg.scale_size_continuous(
                name="Nodes explored",
                range=(1, 8),
                trans="log10",
                breaks=[1, 10, 100, 1000, 10000, 100000],
                labels=["1", "10", "100", "1K", "10K", "100K"],
            )
            + gg.scale_y_continuous(labels=lambda lst: [f"{v*100:.0f}%" for v in lst])
            + gg.labs(
                title="Optimality Gap per Instance",
                x="Instances",
                y="Optimality Gap",
            )
            + gg.theme_minimal()
            + gg.theme(
                figure_size=(10, 6),
                plot_title=gg.element_text(ha="center"),
                plot_subtitle=gg.element_text(ha="center"),
                axis_text_x=gg.element_text(rotation=45, ha="right"),
            )
            + gg.facet_wrap("Time_Bracket")
        )

        return plot

    return gap_plot, time_nodes_plot


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Formulations composition

    All existing and proposed approaches are composed of two different parts of the formulation:
    - base model (basic constraints modeling the nature of a planning task)
    - acyclicity (modeling causal acyclicity between actions)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Basic Model:
    """
    )
    return


@app.cell
def _(load_image):
    load_image("basic_model.png")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Causal Acyclicity Problem Example:
    """
    )
    return


@app.cell
def _(load_image):
    load_image("causal_acyc.png")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Comparisons between existing formulations
    In the literature there are two methods used to model acyclicity:
    - Time Labeling (assigning to each fact a time label, then ensuring that for each action the time labels of achieved facts is greater than the time label of the preconditions)
    - Vertex Elimination (using the concept of vertex elimination graphs to model acyclicity)

    ---

    The current state-of-the-art uses the vertex elimination approach due to its stronger LP relaxation:
    """
    )
    return


@app.cell
def _(time_nodes_plot, tl, ve):
    time_nodes_plot([tl, ve], "TL", ["VE"])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Branch and Cut approaches
    Other approaches to modeling acyclicity may require an exponential number of constraints, so the two methods we proposed require the use of a Branch and Cut approach:
    - Subtour Elimination Constraints (find cycles in infeasible solutions and add the Subtour Elimination Constraint to prevent it from appearing) (denoted as "SEC")
    - Landmark Constraints (it has been proven that the Delete-Free relaxation of a Planning task can be solved as an hitting set over all its landmarks - a set of actions such that each feasible plan must contain at least one action from it...) (denoted as "LMC")

    ---

    Computational results show that the approach that uses landmark constraints is more efficient with respect to the one using subtour elimination constraints, and overall it manages to improve the computational time (at the expense of an increase of number of nodes, which is expected given that this is a B&C approach)
    """
    )
    return


@app.cell
def _(lmc, sec, time_nodes_plot, ve):
    time_nodes_plot([ve, sec, lmc], "VE", ["SEC", "LMC"])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Providing Warm Starts to each formulation

    ### Primal Heuristic as MIP Start
    Finding a feasible plan is trivial, since a simple dive will either find a plan or prove the task's infeasibility... however all these formulations had some difficulties in finding any feasible solution sometimes.
    We can use a greedy algorithm ourselves to provide a good initial feasible solution to the solver.
    The use of the primal heuristic is denoted by "_h".

    ### Initial set of Landmark Constraints through LMcut
    The B&C models primarily, but also the Time Labeling one, have a weak LP relaxation, so adding a set of initial constraints could improve the initial bound.
    LMcut is a state-of-the-art heuristic for classical planning problems, which works by iteratively computing landmarks: we can get those landmarks at the beginning of our method and add them as an initial set of landmark constraints.
    The use of these inital set of constraints is denoted by "_LM".

    ---

    The greedy heuristic manages to return an optimal solution in ~60% of the cases and there's no clear connection between the time needed to compute the optimal solution/the number of nodes explored and the optimality gap.
    """
    )
    return


@app.cell
def _(gap_plot, ve_hLM):
    gap_plot(ve_hLM)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The use of these warm start techniques are very helpful, especially to the TL and LMC models: this is to be attributed by the quality of both the primal heuristic function and the quality of the landmarks computed by the LMcut algorithm.
    """
    )
    return


@app.cell
def _(
    lmc,
    lmc_h,
    lmc_hLM,
    time_nodes_plot,
    tl,
    tl_h,
    tl_hLM,
    ve,
    ve_h,
    ve_hLM,
):
    time_nodes_plot(
        [tl, tl_h, tl_hLM, ve, ve_h, ve_hLM, lmc, lmc_h, lmc_hLM],
        baseline="IGNORED SINCE CLASS BASELINES IS SET",
        comparisons=["TL_h", "TL_hLM", "VE_h", "VE_hLM", "LMC_h", "LMC_hLM"],
        facet_on="Model_Class",
        class_baselines={"TL": "TL", "VE": "VE", "LM": "LMC"},
        class_order=["TL", "VE", "LM"],
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Separating landmark constraints from fractional solutions
    Typical B&C approaches don't just separate violated constraints on integer points, but have separation procedures in place also for fractional solutions...
    Our separation procedure for fractional solutions is based on the Max-Flow problem, extracting the violated landmark as the minimum cut associated to the max-flow computation.

    ---

    As shown in the following plot, we aren't able to provide a significant time improvement, but we manage to reduce the number of nodes explored, an aspect in which our approach wasn't competitive with the previous state of the art
    """
    )
    return


@app.cell
def _(flm_hLM, lmc_hLM, time_nodes_plot):
    time_nodes_plot([lmc_hLM, flm_hLM], "LMC_hLM", ["LMCf_hLM"])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Overall Comparisons

    On top of the explained methods, our current best method (with slight variations/additions to the proposed ones) is shown as "LMC_b"
    """
    )
    return


@app.cell
def _(flm_hLM, lmc_hLM, run_best, time_nodes_plot, ve, ve_hLM):
    time_nodes_plot(
        [ve, ve_hLM, lmc_hLM, flm_hLM, run_best],
        "VE",
        ["VE_hLM", "LMC_hLM", "LMCf_hLM", "LMC_b"],
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Considering simply the number of instances solved to optimality, we can see how (except for the SEC approaches) each proposed approach manages to solve more instances, with the most number of instances solved by our best implementation of the LMC model.
    """
    )
    return


@app.cell
def _(
    flm_hLM,
    gg,
    lmc,
    lmc_hLM,
    pl,
    run_best,
    sec,
    sec_hLM,
    tl,
    tl_hLM,
    ve,
    ve_hLM,
):
    def solved_bar_plot(data: pl.DataFrame) -> gg.ggplot:
        plot_data = (
            data.filter(pl.col("Status") == True)
            .group_by("Model")
            .agg(pl.len().alias("Solved"))
            .sort(
                "Solved", descending=False
            )  # ascending so that highest is on top in coord_flip
            .with_columns(
                pl.col("Model").cast(
                    pl.Enum(
                        data.filter(pl.col("Status") == True)
                        .group_by("Model")
                        .agg(pl.len().alias("Solved"))
                        .sort("Solved", descending=False)
                        .get_column("Model")
                        .to_list()
                    )
                )
            )
            .with_columns(model_type=pl.col("Model").cast(pl.String).str.slice(0, 2))
        )

        return (
            gg.ggplot(plot_data, gg.aes(x="Model", y="Solved", fill="model_type"))
            + gg.geom_bar(stat="identity", width=0.8)
            + gg.annotate(
                "segment", x=0.4, xend=10.6, y=500, yend=500, color="white", size=0.8
            )
            + gg.annotate(
                "segment", x=0.4, xend=10.6, y=1000, yend=1000, color="white", size=0.8
            )
            + gg.annotate(
                "segment", x=0.4, xend=10.6, y=1500, yend=1500, color="white", size=0.8
            )
            + gg.annotate(
                "segment", x=0.4, xend=10.6, y=2000, yend=2000, color="white", size=0.8
            )
            + gg.annotate(
                "segment", x=0.4, xend=10.6, y=2500, yend=2500, color="white", size=0.8
            )
            + gg.geom_text(gg.aes(label="Solved"), ha="right", nudge_y=0)
            + gg.coord_flip()
            + gg.labs(
                title="Solved Instances per Model",
                x="Model",
                y="Solved Instances",
            )
            + gg.scale_fill_brewer(type="qual", palette="Set2")
            + gg.theme_minimal()
            + gg.theme(
                figure_size=(10, 6),
                plot_title=gg.element_text(ha="center"),
                plot_subtitle=gg.element_text(ha="center"),
            )
        )

    solved_bar_plot(
        pl.concat(
            [tl, tl_hLM, ve, ve_hLM, sec, sec_hLM, lmc, lmc_hLM, flm_hLM, run_best]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The next graph shows the cumulative number of instances solved by each model within a certain time limit (the plot is capped at 0.01s since differences in running time are dominated by noise below this threshold)
    """
    )
    return


@app.cell
def _(flm_hLM, gg, lmc, lmc_hLM, pl, run_best, ve, ve_hLM):
    def cumulative_plot(data: pl.DataFrame, x_min: float = 0.01) -> gg.ggplot:
        solved = data.filter(pl.col("Status") == True)
        solved = solved.sort(["Model", "Time"])
        solved = solved.with_columns(
            pl.col("Time").rank("ordinal").over("Model").alias("cumulative_solved")
        )
        solved = solved.filter(pl.col("Time") >= x_min)

        # Compute total solved per model and sort ascending
        model_order = (
            solved.group_by("Model")
            .agg(pl.col("cumulative_solved").max().alias("total"))
            .sort("total", descending=False)
            .get_column("Model")
            .to_list()
        )

        df = solved.with_columns(pl.col("Model").cast(pl.Enum(model_order)))

        return (
            gg.ggplot(
                df,
                gg.aes(
                    x="Time", y="cumulative_solved", color="Model", linetype="Model"
                ),
            )
            + gg.geom_step(size=1.2)
            + gg.scale_x_log10()
            + gg.scale_color_manual(
                values=[
                    "#e41a1c",
                    "#377eb8",
                    "#4daf4a",
                    "#984ea3",
                    "#ff7f00",
                    "#a65628",
                    "#f781bf",
                    "#999999",
                ],
                breaks=model_order,
            )
            + gg.scale_linetype_manual(
                values=[
                    "solid",
                    "dashed",
                    "dotted",
                    "dashdot",
                    "solid",
                    "dashed",
                    "dotted",
                    "dashdot",
                ],
                breaks=model_order,
            )
            + gg.labs(
                x="Time (s)",
                y="Instances Solved",
                title="Cumulative Instances Solved Over Time",
                color="Model",
                linetype="Model",
            )
            + gg.theme_minimal()
            + gg.theme(
                figure_size=(10, 6),
                plot_title=gg.element_text(ha="center"),
                plot_subtitle=gg.element_text(ha="center"),
            )
        )

    cumulative_plot(pl.concat([ve, ve_hLM, lmc, lmc_hLM, flm_hLM, run_best]))
    return


if __name__ == "__main__":
    app.run()
