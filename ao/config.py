"""Constants of the instrument model."""

NUMBER_OF_CHANNELS = 4
# Physical channels (1-based) driven for an N-primary color, in primary order
# (descending wavelength): channel 1 is the weakest and is left off at N = 3.
PRIMARY_CHANNEL_ASSIGNMENT = {3: [2, 3, 4], 4: [1, 2, 3, 4]}
# Adaptation luminance (cd/m^2): XYZ is divided by it before CIELAB.
ADAPTATION_BRIGHTNESS = 100
