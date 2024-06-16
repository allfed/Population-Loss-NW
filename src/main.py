import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from folium.plugins import HeatMap
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans
from matplotlib.path import Path
from shapely.geometry import Point
import warnings

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


class LandScan:
    def __init__(self):
        """
        Load the LandScan TIF file from the data directory and replace negative values by 0
        """
        # Open the TIF file from the data directory
        tif_path = "../data/landscan-global-2022.tif"
        with rasterio.open(tif_path) as dataset:
            data_shape = dataset.shape
            data_dtype = dataset.dtypes[0]
            data = np.memmap(
                "landscan_data.memmap", dtype=data_dtype, mode="w+", shape=data_shape
            )
            # Read the first band in chunks to reduce memory usage
            for _, window in dataset.block_windows(1):
                data_block = dataset.read(window=window)
                # Replace negative values with 0 in the block
                data_block = np.where(data_block < 0, 0, data_block)
                data[
                    window.row_off : window.row_off + window.height,
                    window.col_off : window.col_off + window.width,
                ] = data_block
            print("LandScan TIF file loaded successfully.")

        assert (
            data.sum() == 7906382407
        ), "The sum of the original data should be equal to 7.9 billion"

        self.data = data
        return

    def degrade(self, degrade_factor):
        """
        Degrade the LandScan data by a factor of degrade_factor
        """
        degraded_shape = (
            self.data.shape[0] // degrade_factor,
            self.data.shape[1] // degrade_factor,
        )
        degraded_data = np.memmap(
            "degraded_landscan_data.memmap",
            dtype=self.data.dtype,
            mode="w+",
            shape=degraded_shape,
        )
        # Process the data in chunks to reduce memory usage
        chunk_size = degrade_factor
        for i in range(0, self.data.shape[0], chunk_size):
            for j in range(0, self.data.shape[1], chunk_size):
                block = self.data[i : i + chunk_size, j : j + chunk_size]
                degraded_data[i // degrade_factor, j // degrade_factor] = block.sum()

        assert (
            degraded_data.sum() == self.data.sum()
        ), "The sum of the degraded data should be equal to the sum of the original data"

        self.data = degraded_data
        return

    def plot(self):
        """
        Make a map of the LandScan data
        """
        # Create a Basemap instance
        m = Basemap(projection="cyl", resolution="l")

        # Draw coastlines, countries, and fill continents
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        m.fillcontinents(color="lightgray", lake_color="white")

        # Draw the map boundaries
        m.drawmapboundary(fill_color="white")

        # Convert the degraded data to a masked array
        masked_data = np.ma.masked_where(self.data == 0, self.data)

        # Plot the degraded data on the map
        m.pcolormesh(
            np.linspace(-180, 180, self.data.shape[1] + 1),
            np.linspace(-90, 90, self.data.shape[0] + 1)[::-1],
            masked_data,
            latlon=True,
            cmap="viridis",
            shading="flat",
        )

        # Show the plot
        plt.show()
        return


class Country:
    def __init__(self, country_name, degrade_factor=5, degrade=False):
        """
        Load the population data for the specified country and degrade it by a factor of degrade_factor

        Args:
            country_name (str): the name of the country
            degrade_factor (int): the factor by which to degrade the LandScan data
            degrade (bool): if True, degrade the LandScan data
        """
        # Load landscan data
        landscan = LandScan()
        if degrade:
            landscan.degrade(degrade_factor)

        # Get the geometry of the specified country
        country = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        country = country[country.name == country_name]

        # Get country bounds
        country_bounds = country.geometry.bounds.iloc[0]

        # Calculate the indices for the closest longitude and latitude in the landscan data
        lons = np.linspace(-180, 180, landscan.data.shape[1])
        lats = np.linspace(90, -90, landscan.data.shape[0])

        # Find indices of the closest bounds in the landscan data
        min_lon_idx = np.argmin(np.abs(lons - country_bounds.minx))
        max_lon_idx = np.argmin(np.abs(lons - country_bounds.maxx))
        min_lat_idx = np.argmin(
            np.abs(lats - country_bounds.maxy)
        )  # maxy for min index because lats are decreasing
        max_lat_idx = np.argmin(
            np.abs(lats - country_bounds.miny)
        )  # miny for max index

        # Extract the data for the region
        region_data = landscan.data[
            min_lat_idx : max_lat_idx + 1, min_lon_idx : max_lon_idx + 1
        ]

        # Create a boolean mask for the country within the extracted region
        lons_region, lats_region = np.meshgrid(
            lons[min_lon_idx : max_lon_idx + 1], lats[min_lat_idx : max_lat_idx + 1]
        )
        points_region = gpd.GeoSeries(
            gpd.points_from_xy(lons_region.ravel(), lats_region.ravel())
        )
        gdf_region = gpd.GeoDataFrame(geometry=points_region, crs="EPSG:4326")
        mask_region = gpd.sjoin(
            gdf_region, country, how="inner", predicate="intersects"
        ).index
        mask_region_bool = np.zeros(region_data.size, dtype=bool)
        mask_region_bool[mask_region] = True
        mask_region_bool = mask_region_bool.reshape(region_data.shape)

        # Apply the mask to the population data
        population_data_country = np.where(mask_region_bool, region_data, 0)

        # Get lats and lons for the extracted region
        self.lats = lats[min_lat_idx : max_lat_idx + 1]
        self.lons = lons[min_lon_idx : max_lon_idx + 1]

        self.data = population_data_country
        self.data_original = population_data_country.copy()
        self.hit = np.zeros(population_data_country.shape)
        self.exclude = np.zeros(population_data_country.shape)
        self.target_list = []
        self.fatalities = []
        self.kilotonne = []

        del landscan, lons, lats, points_region, gdf_region, mask_region
        return

    def attack_max_fatality_non_overlapping(self, arsenal):
        """
        Attack the country  by finding where to detonate a given number of warheads over the country's most populated region and without overlapping targets.

        Args:
            arsenal (list): a list of the yield of the warheads in kt
        """
        for yield_kt in arsenal:
            self.attack_next_target(yield_kt)
        return

    def attack_next_target(self, yield_kt):
        """
        Attack the next most populated region

        Args:
            yield_kt (float): the yield of the warhead in kt
        """
        # Create a mask to exclude previously hit targets
        valid_targets_mask = self.exclude == 0

        # Use the mask to filter the data and find the maximum population index
        masked_data = np.where(valid_targets_mask, self.data, np.nan)
        max_population_index = np.unravel_index(np.nanargmax(masked_data), self.data.shape)
        max_population_lat = self.lats[max_population_index[0]]
        max_population_lon = self.lons[max_population_index[1]]

        self.apply_destruction(max_population_lat, max_population_lon, yield_kt)
        self.target_list.append((max_population_lat, max_population_lon))
        self.kilotonne.append(yield_kt)

        return max_population_lat, max_population_lon

    def apply_destruction(self, lat_groundzero, lon_groundzero, yield_kt):
        """
        Destroy the country by removing the population from the specified location and the max_radius km around it

        Args:
            lat (float): the latitude of the target location
            lon (float): the longitude of the target location
            yield_kt (float): the yield of the warhead in kt
        """
        # (1) Apply destruction
        max_radius = np.sqrt(yield_kt / 15) * 3  # From Toon et al. 2008
        delta_lon = max_radius / 6371.0 / np.cos(np.radians(lat_groundzero))
        delta_lat = max_radius / 6371.0
        delta_lon = delta_lon * 180 / np.pi
        delta_lat = delta_lat * 180 / np.pi

        # Create a mask for the box that bounds the destroyed region
        lon_min = lon_groundzero - delta_lon
        lon_max = lon_groundzero + delta_lon
        lat_min = lat_groundzero - delta_lat
        lat_max = lat_groundzero + delta_lat
        lon_indices = np.where((self.lons >= lon_min) & (self.lons <= lon_max))[0]
        lat_indices = np.where((self.lats >= lat_min) & (self.lats <= lat_max))[0]

        # Apply the mask to the destroyed region
        for lat_idx in lat_indices:
            for lon_idx in lon_indices:
                lon_pixel = self.lons[lon_idx]
                lat_pixel = self.lats[lat_idx]
                if (lon_pixel - lon_groundzero) ** 2 / delta_lon**2 + (
                    lat_pixel - lat_groundzero
                ) ** 2 / delta_lat**2 <= 1:
                    population_in_pixel = self.data[lat_idx, lon_idx]
                    distance_from_groundzero = haversine_distance(
                        lat_pixel, lon_pixel, lat_groundzero, lon_groundzero
                    )
                    fatality_rate = get_fatality_rate(
                        distance_from_groundzero, yield_kt
                    )
                    self.fatalities.append(fatality_rate * population_in_pixel)
                    self.hit[lat_idx, lon_idx] = 1

        # (2) Now we apply another mask to make sure there is no overlap with future nukes
        max_radius = 2 * np.sqrt(yield_kt / 15) * 3
        delta_lon = max_radius / 6371.0 / np.cos(np.radians(lat_groundzero))
        delta_lat = max_radius / 6371.0
        delta_lon = delta_lon * 180 / np.pi
        delta_lat = delta_lat * 180 / np.pi

        # Create a mask for the box that bounds the exclusion region
        lon_min = lon_groundzero - delta_lon
        lon_max = lon_groundzero + delta_lon
        lat_min = lat_groundzero - delta_lat
        lat_max = lat_groundzero + delta_lat
        lon_indices = np.where((self.lons >= lon_min) & (self.lons <= lon_max))[0]
        lat_indices = np.where((self.lats >= lat_min) & (self.lats <= lat_max))[0]

        # Apply the mask to the exclusion region, we simply make the population
        # negative so it will never be targeted, but we still have 
        for lat_idx in lat_indices:
            for lon_idx in lon_indices:
                lon_pixel = self.lons[lon_idx]
                lat_pixel = self.lats[lat_idx]
                if (lon_pixel - lon_groundzero) ** 2 / delta_lon**2 + (
                    lat_pixel - lat_groundzero
                ) ** 2 / delta_lat**2 <= 1:
                    self.exclude[lat_idx, lon_idx] = 1

        return

    def get_total_fatalities(self):
        """
        Get the total fatalities
        """
        return int(sum(self.fatalities))

    def plot(self, show_destroyed_regions=False, show_population_density=False):
        """
        Make an interactive map

        Args:
            show_destroyed_regions (bool): if True, show the destroyed regions
            show_population_density (bool): if True, show the population density
        """

        # Create a folium map centered around the average coordinates
        avg_lat = np.mean(self.lats)
        avg_lon = np.mean(self.lons)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5, tiles="OpenStreetMap")

        # Show population density
        if show_population_density:
            # Prepare data for HeatMap (lat, lon, data_value)
            population_density = [
                [
                    float(self.lats[i]),
                    float(self.lons[j]),
                    np.log10(float(self.data_original[i, j])),
                ]
                for i in range(self.data.shape[0])
                for j in range(self.data.shape[1])
                if self.data_original[i, j] > 0
            ]
            HeatMap(
                population_density, min_opacity=0.2, radius=5, blur=4, max_zoom=1
            ).add_to(m)

        # Show destroyed regions
        if show_destroyed_regions:
            # Prepare data for destroyed regions
            destroyed_data = [
                [
                    float(self.lats[i]),
                    float(self.lons[j]),
                    1.0,
                ]  # 1.0 as a placeholder for destroyed regions
                for i in range(self.data.shape[0])
                for j in range(self.data.shape[1])
                if self.hit[i, j] == 1
            ]
            HeatMap(
                destroyed_data, gradient={0.1: "black", 1: "red"}, min_opacity=0.6
            ).add_to(m)

        # plot targets
        for i, target in enumerate(self.target_list):
            folium.Marker(
                [float(target[0]), float(target[1])],
                popup=f"Hit with {self.kilotonne[i]} kt",
                icon=folium.Icon(color="red", icon="radiation"),
            ).add_to(m)

        # Display the map
        m.save("interactive_map.html")

        return m


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two points in km."""

    R = 6371  # Earth's radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) * np.cos(
        np.radians(lat2)
    ) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def get_fatality_rate(distance_from_groundzero, yield_kt):
    """
    Calculates the fatality rate given the distance from the ground zero and the yield of the warhead in kt

    Based on Toon et al. 2007, 2008

    Args:
        distance_from_groundzero (float): the distance from the ground zero in km
        yield_kt (float): the yield of the warhead in kt

    Returns:
        fatality_rate (float): the fatality rate
    """
    sigma = 1.15 * np.sqrt(yield_kt / 15)
    return np.exp(-(distance_from_groundzero**2) / (2 * sigma**2))


##################### tests to include ##########################
# # assert that total population of China is 1.4 billion +/- 1%
# population_data_china = get_population_data_for_country("China", population_data)
# assert (
#     abs(population_data_china.sum() - 1_400_000_000) < 14_000_000
# ), "The sum of the population data for China should be equal to 1.4 billion"
