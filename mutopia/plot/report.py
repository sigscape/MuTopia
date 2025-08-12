import matplotlib.pyplot as plt
from collections import defaultdict
from .signature_plot import plot_component
from .coef_matrix_plot import plot_interaction_matrix
from ..gtensor import fetch_component
from .bubble_plot import plot_shap_summary

def plot_signature_report(
        dataset,
        component,
        width=5.25,
        height=2.0,
        show=True,
        bubble_scale=300,
    ):
        """
        Generate a comprehensive report for a specific signature component.

        This method creates a figure with signature plots for mesoscale states and an interaction matrix
        for the specified component, providing a visual representation of the signature's characteristics.

        Parameters
        ----------
        component : int or str
            The signature component to visualize. Can be an integer index or a string identifier.
        normalization : str, default="global"
            The normalization method to use for the signatures.
        width : float, default=5.25
            The base width of the figure in inches. The actual figure width may be adjusted based on the number of states.
        height : float, default=2.0
            The base height per signature group in inches.
        show : bool, default=True
            Whether to display the figure immediately.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure containing signature plots and interaction matrix.

        Notes
        -----
        The report organizes mesoscale states into groups based on their prefix (before the colon),
        and displays them in separate rows. For singleton state groups (except Baseline),
        the Baseline state is automatically added as a reference.
        """
        signatures = fetch_component(dataset, component)
        n_rows = len(signatures.genome_state)
        state_groups = defaultdict(list)
        for state in signatures.genome_state.values:
            state_groups[state.split(":")[0]].append(state)

        #for k, v in state_groups.items():
        #    if not k == "Baseline" and len(v) == 1:
        #        state_groups[k].append("Baseline")

        max_n_states = max(map(len, state_groups.values()))
        n_sigs = len(state_groups)
        fig = plt.figure(figsize=(max(width * max_n_states, 10), height * n_sigs + 3))

        gs = fig.add_gridspec(
            3,
            1,
            height_ratios=[height * n_sigs + 1, 1.5, 2.5 + 0.35 * n_rows],
            hspace=0.35,
        )

        gs0 = gs[0].subgridspec(
            n_sigs + 1,
            max_n_states,
            hspace=0.75,
            wspace=0.5,
            width_ratios=[3] + [1] * (max_n_states - 1),
        )

        for i, states in enumerate(state_groups.values()):
            ax = fig.add_subplot(gs0[i, : len(states)])
            plot_component(
                signatures,
                *states,
                ax=ax,
            )

        shap_ax = fig.add_subplot(gs[1, 0])
        plot_shap_summary(
             dataset,
             component_order=[component],
             ax=shap_ax,
             scale=bubble_scale,
        )
        shap_ax.set(
             ylabel="Feature impacts",
             xlabel="",
        )
        shap_ax.set_yticklabels([])

        plot_interaction_matrix(
            dataset,
            component,
            gridspec=gs[2],
        )

        fig.suptitle(f"Component {component} report", fontsize=12, y=0.95)
        if show:
            plt.show()

        return fig