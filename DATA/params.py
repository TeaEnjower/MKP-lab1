import numpy as np
import numpy.typing as npt

MU: float = 398600.4418 # km^3/s^2
EARTH_RADIUS: float = 6378.136 # km 
EARTH_WEIGHT: float = 5.97219e24 # kg
G_FORCE: float = 6.67430e-17 # H * km^2 / (kg^2)
DRAG_FORCE: float = 3.5
CROSS_SECTIONAL_AREA: float = 0.000023 # km^2
SATELLITE_WEIGHT: float = 1650.0 # kg
NIGHT_DENSITY_H_120: float = 1.58868e-8 # kg/m^3

# выражения
PRECISION_EQUATION: float = 1e-6

# numpy types
NDArrayFloat = npt.NDArray[np.float64]

# элементы орбит
HEIGHT_P: float = 340.0 # km
HEIGHT_A: float = 450.0 # km
INCLINATION: float = np.radians(20.0) # radians
LONGITUDE_ASCENDING_NODE: float = np.radians(10.0) # radians
PERICENTER_ARGUMENT: float = 0.0 # radians
AVERAGE_ANOMALY: float = np.radians(180.0) # radians

# выражения
PRECISION_EQUATION: float = 1e-6

# numpy types
NDArrayFloat = npt.NDArray[np.float64]
