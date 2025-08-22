from matplotlib.colors import LinearSegmentedColormap

# Create a custom diverging colormap
diverging_palette = LinearSegmentedColormap.from_list(
    "custom_diverging",
    ["#427aa1ff", "#FAFAFA", "#e07a5fff"],  # White or a neutral color at center
)

categorical_palette = ["#427aa1ff", "#e07a5fff", "#acacacff", "#83c5beff"]
