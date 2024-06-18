import cartopy.crs as ccrs
import cartopy.img_transform
import folium
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from folium.plugins import HeatMap
from mpl_toolkits.basemap import Basemap
from PIL import Image
from matplotlib.colors import ListedColormap
from pyproj import CRS
import pyproj
from scipy.ndimage import convolve
import shapely.geometry
import shapely.geometry as sgeom
import warnings


# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


class LandScan:
    def __init__(self, landscan_year=2022):
        """
        Load the LandScan TIF file from the data directory and replace negative values by 0

        Args:
            landscan_year (int): the year of the LandScan data
        """
        # Open the TIF file from the data directory
        tif_path = f"../data/landscan-global-{landscan_year}.tif"
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

        if landscan_year == 2022:
            assert (
                data.sum() == 7906382407
            ), "The sum of the original data should be equal to 7.9 billion"

        self.data = data
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
    def __init__(self, country_name, landscan_year=2022):
        """
        Load the population data for the specified country

        Args:
            country_name (str): the name of the country
            degrade_factor (int): the factor by which to degrade the LandScan data
            degrade (bool): if True, degrade the LandScan data
            landscan_year (int): the year of the LandScan data
        """
        # Load landscan data
        landscan = LandScan(landscan_year)

        # Get the geometry of the specified country
        country = gpd.read_file("../data/natural-earth/ne_10m_admin_0_countries.shp")
        country = country[country.ADMIN == country_name]

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

        self.hit = np.zeros(population_data_country.shape)
        self.exclude = np.zeros(population_data_country.shape)
        self.target_list = []
        self.fatalities = []
        self.kilotonne = []

        del landscan, lons, lats, points_region, gdf_region, mask_region
        return

    def attack_max_fatality_non_overlapping(self, arsenal, include_injuries=False):
        """
        Attack the country  by finding where to detonate a given number of warheads over the country's most populated region and without overlapping targets.

        Args:
            arsenal (list): a list of the yield of the warheads in kt
        """
        self.include_injuries = include_injuries
        if not all(x == arsenal[0] for x in arsenal):
            warnings.warn(
                "Arsenal contains different yield values. The current non-overlapping target allocation algorithm will not handle this correctly."
            )

        # Calculate the average population over neighboring cells within a specified radius
        # This will be use for target-finding only.
        # This avoids the problem of hitting a target with very high population density over
        # 1 km² but low population density around it
        #
        # Approximate region where fatalities will occur. This is approximative, but again
        # this is only used for target finging.
        #
        radius = int(1.15 * np.sqrt(arsenal[0] / 15) * 2)
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        mask = x**2 + y**2 <= radius**2
        kernel[mask] = 1
        kernel /= kernel.sum()
        self.data_averaged = convolve(self.data, kernel, mode="constant")
        self.hit = np.zeros(self.data.shape)
        self.exclude = np.zeros(self.data.shape)

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
        masked_data = np.where(valid_targets_mask, self.data_averaged, np.nan)
        max_population_index = np.unravel_index(
            np.nanargmax(masked_data), self.data.shape
        )
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
                        distance_from_groundzero, yield_kt, self.include_injuries
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
        Get the total fatalities, will include both fatalities and injuries if include_injuries is True
        """
        return int(sum(self.fatalities))

    def plot(self, show_hit_regions=False, show_population_density=False):
        """
        Make an interactive map

        Args:
            show_hit_regions (bool): if True, show the hit regions
            show_population_density (bool): if True, show the population density
        """

        # Create a folium map centered around the average coordinates
        avg_lat = np.mean(self.lats)
        avg_lon = np.mean(self.lons)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5, tiles="OpenStreetMap")

        # Show population density
        if show_population_density:
            bounds = [
                [np.min(self.lats), np.min(self.lons)],
                [np.max(self.lats), np.max(self.lons)],
            ]
            folium.raster_layers.ImageOverlay(
                image=np.log10(self.data + 1),
                bounds=bounds,
                colormap=ListedColormap(
                    ["none"] + list(plt.cm.viridis(np.linspace(0, 1, plt.cm.viridis.N)))
                ),
                opacity=0.5,
                mercator_project=True,
            ).add_to(m)

        # Show hit regions
        if show_hit_regions:
            # Reproject coordinates from PlateCarre to Mercator

            bounds = [
                [np.min(self.lats), np.min(self.lons)],
                [np.max(self.lats), np.max(self.lons)],
            ]
            folium.raster_layers.ImageOverlay(
                image=self.hit,
                bounds=bounds,
                colormap=ListedColormap(
                    ["none", "red"]
                ),  # Use a red colormap for hit regions
                opacity=0.5,
                mercator_project=True,
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


def get_fatality_rate(distance_from_groundzero, yield_kt, include_injuries=False):
    """
    Calculates the fatality rate given the distance from the ground zero and the yield of the warhead in kt

    Based on Toon et al. 2007, 2008

    Args:
        distance_from_groundzero (float): the distance from the ground zero in km
        yield_kt (float): the yield of the warhead in kt
        include_injuries (bool): if True, includes fatalities + injuries
    Returns:
        fatality_rate (float): the fatality rate
    """
    if include_injuries:
        sigma0 = 1.87
    else:
        sigma0 = 1.15
    sigma = sigma0 * np.sqrt(yield_kt / 15)
    return np.exp(-(distance_from_groundzero**2) / (2 * sigma**2))
