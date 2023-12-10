# project config
DEV_MODE = True  # by default it's False. Set to True to enable dev mode.
SUPPRESS_PLOT = DEV_MODE  # set to True to suppress plot, False to show plot

DEFAULT_PLOT_BLOCK = (
    not SUPPRESS_PLOT
)  # True to pause after each plot, False to not pause
