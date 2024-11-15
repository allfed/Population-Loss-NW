import geopy.distance as distance
import numpy as np
import geopandas as gpd


def calculate_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Args:
        lat1 (float): latitude of the first point
        lon1 (float): longitude of the first point
        lat2 (float): latitude of the second point
        lon2 (float): longitude of the second point

    Returns:
        dist (float): the distance in km
    """
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    dist = distance.distance(point1, point2).kilometers
    return dist


def get_fatality_rate(
    distance_from_groundzero,
    yield_kt,
    include_injuries=False,
    kill_radius_prescription="default",
    airburst=True,
):
    """
    Calculates the fatality rate given the distance from ground zero and the yield of the warhead in kt.
    This accounts for blast, fire and prompt radiation (no fallout).

    Based on Toon et al. 2007, 2008 but using different scaling laws.

    Args:
        distance_from_groundzero (float): the distance from the ground zero in km
        yield_kt (float): the yield of the warhead in kt
        include_injuries (bool): if True, includes fatalities + injuries
        kill_radius_prescription (str): The method to calculate the kill radius, one of "default",
            "Toon", or "overpressure".
    Returns:
        fatality_rate (float): the fatality rate
    """
    if include_injuries:
        sigma0 = 1.87
    else:
        sigma0 = 1.15
    scaling_factor = calculate_kill_radius_scaling_factor(
        kill_radius_prescription, yield_kt, airburst=airburst
    )
    sigma = sigma0 * scaling_factor
    return np.exp(-(distance_from_groundzero**2) / (2 * sigma**2))


def calculate_kill_radius_scaling_factor(scaling_prescription, yield_kt, airburst=True):
    """
    Calculate the scaling factor for the kill radius.

    Args:
        scaling_prescription (str): The method to calculate the kill radius.
        yield_kt (float): The yield of the warhead in kt.
        airburst (bool): True for air bursts, False for ground bursts; affects scaling factor

    Returns:
        float: The calculated scaling factor.
    """
    if not airburst:  # see Toon et al. 2007 for sqrt(2) correction
        corr = 1 / np.sqrt(2)
    else:
        corr = 1
    if scaling_prescription == "Toon":
        return corr * np.sqrt(yield_kt / 15)
    elif scaling_prescription == "default":
        return corr * (yield_kt / 18) ** 0.38287688
    elif scaling_prescription == "overpressure":
        return corr * (yield_kt / 18) ** 0.33333333
    else:
        raise ValueError(f"Unknown scaling prescription: {scaling_prescription}")


def calculate_max_radius_burn(burn_radius_prescription, yield_kt):
    """
    Calculate the maximum burn radius based on the given prescription and yield.

    Args:
        burn_radius_prescription (str): The method to calculate the burn radius.
        yield_kt (float): The yield of the warhead in kt.

    Returns:
        float: The calculated maximum burn radius.
    """
    if burn_radius_prescription == "Toon":
        # From Toon et al. 2008, linear scaling of burned area with yield, with 13 km² for Hiroshima
        return 2.03 * (yield_kt / 15) ** 0.50
    elif burn_radius_prescription == "default":
        # Most realistic model, see scripts/burn-radius-scaling.ipynb
        return 0.75094351 * yield_kt**0.38287688
    elif burn_radius_prescription == "overpressure":
        # Scales from average of Hiroshima and Nagasaki (13km² and 6.7km², 15kt and 21kt),
        # also assumes that it scales like D**(1/3)
        return 1.77 * (yield_kt / 18) ** (1 / 3)
    else:
        raise ValueError(
            f"Unknown burn radius prescription: {burn_radius_prescription}"
        )


def process_chunk(chunk, country, region_data_shape):
    """
    Process a chunk of points to determine which ones intersect with the country's area.
    Used for parallel processing of large point datasets.

    Args:
        chunk (GeoDataFrame): A chunk of points to process
        country (GeoDataFrame): The country boundary geometry
        region_data_shape (tuple): Shape of the region data array to create mask for

    Returns:
        ndarray: Boolean mask indicating which points in the chunk intersect with country
    """
    mask_region = gpd.sjoin(chunk, country, how="inner", predicate="intersects").index
    mask_region_bool = np.zeros(region_data_shape, dtype=bool)
    mask_region_bool.ravel()[mask_region] = True
    return mask_region_bool
