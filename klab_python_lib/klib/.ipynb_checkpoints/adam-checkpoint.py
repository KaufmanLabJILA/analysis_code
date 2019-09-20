from .imports import *

# from .expfile import *
# from .analysis import *
# from .mathutil import *
# from .plotutil import *
# from .imagutil import *
# from .mako import *
# from .adam import *
# Data export functions for Adam.

def toCSV(xname, yname, x, y, filename = "tmpdata"):
    np.savetxt(filename, data, delimiter = ',')