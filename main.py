import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from DATA.params import*
from DATA.data import data

"""
1.
На основе исходных параметров орбиты ha
𝒉п, i, Ω, ω, М, которые
отвечают заданному моменту времени t, по формулам найдём
координаты заданной точки в АГЭСК (𝒙𝒂, 𝒚𝒂, 𝒛𝒂);
"""
def solve_kepler_equation(anomaly: float, e: float):
    last_root: float = anomaly
    root: float = e*np.sin(last_root) + last_root
    true_anomaly: float

    while abs(root - last_root) >= PRECISION_EQUATION:
        last_root = root
        root = e * np.sin(last_root) + anomaly

    true_anomaly = 2*np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(root / 2))
    if true_anomaly < 0  or anomaly > np.pi:
        true_anomaly += 2*np.pi

    return true_anomaly

def get_coordinates_in_AGERC(h_a: float, h_p: float, i: float, omega: float, omega_p: float, m: float):
    r_a: float
    r_p: float
    a: float
    e: float
    r: float
    x_a: float
    y_a: float
    z_a: float
    r_a = h_a + EARTH_RADIUS
    r_p = h_p + EARTH_RADIUS
    a = (r_a + r_p) / 2
    e = (r_a - r_p) / (r_a + r_p)
    r = a * (1 - e**2) / (1 + e * np.cos(m))
    x_a = r * (np.cos(omega_p) * np.cos(omega + m) - np.sin(omega_p) * np.sin(omega + m) * np.cos(i))
    y_a = r * (np.sin(omega_p) * np.cos(omega + m) + np.cos(omega_p) * np.sin(omega + m) * np.cos(i))
    z_a = r * (np.sin(i) * np.sin(omega + m))
    return np.array([x_a, y_a, z_a])

"""
2.
Найдём трансверсальную и радиальную скорости;
"""
def get_speed_components(mu: float, p: float, e: float, nu: float):
    v_r: float
    v_t: float
    v_r = np.sqrt(mu / p) * e * np.sin(nu)
    v_t = np.sqrt(mu / p) * (1 + e * np.cos(nu))
    return v_r, v_t

"""
3.
найдём положение
точки В ГСК, заданное координатами x, y, z, и рассчитать
соответствующие им геодезические координаты 𝑳, 𝑩, 𝑯;
"""
def convert_to_geocentric_coordinates(
    absolute_coordinates: NDArrayFloat, inclination: float, raan: float, perigee_argument: float):
    cos_o: float = np.cos(-raan)
    sin_o: float = np.sin(-raan)
    cos_i: float = np.cos(-inclination)
    sin_i: float = np.sin(-inclination)
    cos_w: float = np.cos(-perigee_argument)
    sin_w: float = np.sin(-perigee_argument)

    rotation_matrix: NDArrayFloat = np.array(
        [
            [cos_o * cos_w - sin_o * sin_w * cos_i, -cos_o * sin_w - sin_o * cos_w * cos_i, sin_o * sin_i],
            [sin_o * cos_w + cos_o * sin_w * cos_i, -sin_o * sin_w + cos_o * cos_w * cos_i, -cos_o * sin_i],
            [sin_w * sin_i, cos_w * sin_i, cos_i],
        ]
    )
    return rotation_matrix @ absolute_coordinates

def geocentric_to_geodetic(geocentric_coordinates: NDArrayFloat):
    x: float
    y: float
    z: float
    B: float
    L: float
    H: float
    x, y, z = geocentric_coordinates
    D = np.sqrt(x**2 + y**2)
    if D == 0:
        B = (np.pi / 2) * np.sign(z)
        L = 0
        H = z * np.sin(B) - EARTH_RADIUS * np.sqrt(1 - 0.0067385254**2 * np.sin(B)**2)
    else:
        L_a = np.arcsin(y / D)
        if (y < 0 and x > 0):
            L = 2 * np.pi - L_a
        elif (y < 0 and x < 0):
            L = np.pi + L_a
        elif(y > 0 and x < 0):
            L = np.pi - L_a
        else:
            L = L_a
        if z == 0:
            B = 0
            H = D - EARTH_RADIUS
        else:
            r = np.sqrt(x**2 + y**2 + z**2)
            c = np.arcsin(z / r)
            p = 0.0067385254**2 * EARTH_RADIUS / (2 * r)
            last_root: float = 0
            b: float = c
            root: float = np.arcsin(p * np.sin(2 * b) / np.sqrt(1 - 0.0067385254**2 * np.sin(b)**2))
            while abs(root - last_root) >= PRECISION_EQUATION:
                last_root = root
                b = c + last_root
                root = np.arcsin(p * np.sin(2 * b) / np.sqrt(1 - 0.0067385254**2 * np.sin(b)**2))
            B = b
            H  = D * np.cos(B) + z * np.sin(B) - EARTH_RADIUS * np.sqrt(1 - 0.0067385254**2 * np.sin(B)**2)
    return np.array([np.degrees(B), np.degrees(L), H])

"""
4.
Рассчитать 𝝆атм(𝑯);
"""
def calculate_atmosphere_density(a_0: float, a_1: float, a_2: float, a_3: float, a_4: float,
    a_5: float, a_6: float, h: float, p_0: float, k_0: float = 1, k_1: float = 0,
    k_2: float = 0, k_3: float = 0, k_4: float = 0) -> float:
    p_h: float
    p: float
    p_h = p_0 * np.exp(a_0 + a_1 * h + a_2 * h**2 + a_3 * h**3 + a_4 * h**4 + a_5 * h**5 + a_6 * h**6)
    p = p_h * k_0 * (1 + k_1 + k_2 + k_3 + k_4)
    return p * 10**9 # kg/km^3

"""
5.
Согласно формулам найти составляющие возмущающего
ускорения, обусловленного влиянием атмосферы, S, T, W;
"""
def get_components_perturbing_accelerations(h: float, c_xa: float, s_m: float, m: float, dens: float, p: float, e: float, nu: float):
    v_r: float
    v_t: float
    v: float
    S: float
    T: float
    W: float = 0
    sigma_x: float = c_xa * s_m / (2 * m) # ballistic coefficient/баллистический коэффициент
    v_r, v_t = get_speed_components(MU, p, e, nu)
    v = np.sqrt(v_r**2 + v_t**2)
    S = -sigma_x * dens * v * v_r
    T = -sigma_x * dens * v * v_t
    return S, T, W

def main():
    geocentric_coordinates: NDArrayFloat
    absolute_coordinates: NDArrayFloat
    values_for_solar_activity: NDArrayFloat = [75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 250.0]
    r_a = HEIGHT_A + EARTH_RADIUS
    r_p = HEIGHT_P + EARTH_RADIUS
    a = (r_a + r_p) / 2
    e = (r_a - r_p) / (r_a + r_p)
    r = a * (1 - e**2) / (1 + e * np.cos(AVERAGE_ANOMALY))
    nu = solve_kepler_equation(AVERAGE_ANOMALY, e)
    p = r * (1 + e * np.cos(nu))
    S_vals = []
    T_vals = []
    W_vals = []
    perturb_values = []
    absolute_coordinates = get_coordinates_in_AGERC(HEIGHT_A, HEIGHT_P, INCLINATION, LONGITUDE_ASCENDING_NODE, PERICENTER_ARGUMENT, AVERAGE_ANOMALY)
    geocentric_coordinates = convert_to_geocentric_coordinates(absolute_coordinates, INCLINATION, LONGITUDE_ASCENDING_NODE, PERICENTER_ARGUMENT)
    B, L, H = geocentric_to_geodetic(geocentric_coordinates)
    g = MU / (EARTH_RADIUS + H)**2
    print(f'Значение ускорения свободного падения: {g:.4f}')
    h_elements = 120 if H < 500 else 500
    for solution in data[h_elements]:
        density = calculate_atmosphere_density(*data[h_elements][solution], H, NIGHT_DENSITY_H_120) # Плотность атмоферы на высоте {H} км = {density}
        S, T, W = get_components_perturbing_accelerations(H, DRAG_FORCE, CROSS_SECTIONAL_AREA, SATELLITE_WEIGHT, density, p, e, nu)
        S_vals.append(S)
        T_vals.append(T)
        W_vals.append(W)
        total_perturb = np.sqrt(S**2 + T**2 + W**2)
        perturb_values.append(total_perturb)

    print(f'Компоненты возмущающего ускорения при h = {H} км')
    pprint(S_vals)
    pprint(T_vals)
    print('Полная величина возмущающего ускорения')
    pprint(perturb_values)
    plt.subplot(2, 1, 1)
    plt.plot(values_for_solar_activity, S_vals, marker='o', label='S')
    plt.plot(values_for_solar_activity, T_vals, marker='o', label='T')
    plt.plot(values_for_solar_activity, W_vals, marker='o', label='W')
    plt.title(f'Компоненты возмущающего ускорения при h = {H} км')
    plt.xlabel('Уровень солнечной активности')
    plt.ylabel('a: км/с^2')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(values_for_solar_activity, perturb_values, marker='o', label=f'Полное возмущающее ускорение')
    plt.title(f'Полная величина возмущающего ускорения при h = {H:.3f} км')
    plt.xlabel('Уровень солнечной активности')
    plt.ylabel('a: км/с^2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
