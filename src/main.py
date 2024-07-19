import csv
import multiprocessing as mp
import warnings

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmium
import pandas as pd
import pycountry
import pyproj
import rasterio

from branca.colormap import LinearColormap
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap
from scipy.ndimage import convolve
from shapely.geometry import box, Polygon
from skimage.measure import block_reduce


# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


class LandScan:
    def __init__(
        self,
        landscan_year=2022,
        degrade=False,
        degrade_factor=1,
        use_HD=False,
        country_HD=None,
    ):
        """
        Load the LandScan TIF file from the data directory and replace negative values by 0

        Args:
            landscan_year (int): the year of the LandScan data
            degrade (bool): if True, degrade the LandScan data
            degrade_factor (int): the factor by which to degrade the LandScan data
            use_HD (bool): if True, use the HD version of the LandScan data
            country_HD (str): the name of the country to use the HD version for
        """
        # Open the TIF file from the data directory
        tif_path = f"../data/landscan-global-{landscan_year}.tif"

        if use_HD and country_HD == "United States of America":
            tif_path = f"../data/landscan-hd/landscan-usa-{landscan_year}-conus-day.tif"
            self.min_lon = -125
            self.max_lon = -66.75
            self.min_lat = 24.25
            self.max_lat = 49.5
        elif use_HD:
            raise ValueError(f"HD LandScan for {country_HD} not yet supported")
        else:
            self.min_lon = -180
            self.max_lon = 180
            self.min_lat = -90
            self.max_lat = 90

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

        if landscan_year == 2022 and not use_HD:
            assert (
                data.sum() == 7906382407
            ), "The sum of the original data should be equal to 7.9 billion"

        if degrade:
            # Degrade the resolution of the data by summing cells using block reduce
            block_size = (degrade_factor, degrade_factor)
            self.data = block_reduce(data, block_size, np.sum)
            if landscan_year == 2022 and not use_HD:
                assert (
                    self.data.sum() == 7906382407
                ), "The sum of the original data should be equal to 7.9 billion"
        else:
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
            np.linspace(self.min_lon, self.max_lon, self.data.shape[1] + 1),
            np.linspace(self.min_lat, self.max_lat, self.data.shape[0] + 1)[::-1],
            masked_data,
            latlon=True,
            cmap="viridis",
            shading="flat",
        )

        # Show the plot
        plt.show()
        return


class Country:
    def __init__(
        self,
        country_name,
        landscan_year=2022,
        degrade=False,
        degrade_factor=1,
        use_HD=False,
        subregion=None,
        industry=True,
    ):
        """
        Load the population data for the specified country

        Args:
            country_name (str): the name of the country
            landscan_year (int): the year of the LandScan data
            degrade (bool): if True, degrade the LandScan data
            degrade_factor (int): the factor by which to degrade the LandScan data
            use_HD (bool): if True, use the HD version of the LandScan data
            subregion (list): the bounds of the subregion to extract from the LandScan data in the format
             [min_lon, max_lon, min_lat, max_lat]. This is optional and if not provided, the entire country is used.
            industry (bool): if True, then load industrial zones
        """
        # Load landscan data
        self.degrade_factor = degrade_factor
        if use_HD:
            country_HD = country_name
        else:
            country_HD = None
        landscan = LandScan(landscan_year, degrade, degrade_factor, use_HD, country_HD)

        self.approximate_resolution = 1 * self.degrade_factor  # km
        if use_HD:
            self.approximate_resolution = 0.09 * self.degrade_factor  # km

        # Get the geometry of the specified country
        country = gpd.read_file("../data/natural-earth/ne_10m_admin_0_countries.shp")
        country = country[country.ADMIN == country_name]

        # Get country bounds
        country_bounds = country.geometry.bounds.iloc[0]

        # If subregion is provided, use it
        if subregion is not None:
            country_bounds = pd.Series(
                {
                    "minx": subregion[0],
                    "maxx": subregion[1],
                    "miny": subregion[2],
                    "maxy": subregion[3],
                }
            )

        # Calculate the indices for the closest longitude and latitude in the landscan data
        lons = np.linspace(landscan.min_lon, landscan.max_lon, landscan.data.shape[1])
        lats = np.linspace(landscan.max_lat, landscan.min_lat, landscan.data.shape[0])

        # Find indices of the closest bounds in the landscan data
        min_lon_idx = np.argmin(np.abs(lons - country_bounds.minx))
        max_lon_idx = np.argmin(np.abs(lons - country_bounds.maxx))
        min_lat_idx = np.argmin(
            np.abs(lats - country_bounds.maxy)
        )  # maxy for min index because lats are decreasing
        max_lat_idx = np.argmin(
            np.abs(lats - country_bounds.miny)
        )  # miny for max index

        # For France, only include European territories
        if country_name == "France":
            min_lat_idx = np.argmin(np.abs(lats - 51))
            max_lat_idx = np.argmin(np.abs(lats - 42))
            min_lon_idx = np.argmin(np.abs(lons + 6))
            max_lon_idx = np.argmin(np.abs(lons - 10))

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

        # Split gdf_region into chunks for parallel processing
        num_processes = mp.cpu_count()
        chunk_size = len(gdf_region) // num_processes
        chunks = [
            gdf_region[i : i + chunk_size]
            for i in range(0, len(gdf_region), chunk_size)
        ]

        # Create a multiprocessing pool and process chunks in parallel
        with mp.Pool(processes=num_processes) as pool:
            mask_region_chunks = pool.starmap(
                process_chunk, [(chunk, country, region_data.shape) for chunk in chunks]
            )

        # Combine the results from all chunks using logical OR
        mask_region_bool = np.logical_or.reduce(mask_region_chunks)

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

        if industry:
            osm_file = (
                f"../data/OSM/{country_name.lower().replace(' ', '-')}-industrial.osm"
            )
            self.industry = get_industrial_areas_from_osm(osm_file)
            self.industry_equal_area = self.industry.to_crs("ESRI:54034")
            self.total_industry_area = sum(
                row.geometry.area for _, row in self.industry_equal_area.iterrows()
            )
            self.destroyed_industrial_areas = []

        del (
            landscan,
            lons,
            lats,
            points_region,
            gdf_region,
            mask_region_chunks,
            mask_region_bool,
        )
        return

    def calculate_averaged_population(self, yield_kt):
        """
        Calculate the average population over neighboring cells within a specified radius.

        This method is used for target-finding only. It avoids the problem of hitting a target
        with very high population density over 1 km² but low population density around it.

        Both radius and sigma are in units of lat/lon cells.

        Args:
            yield_kt (float): The yield of the warhead in kt.

        Returns:
            None. Sets self.data_averaged with the convolved population data.
        """
        radius = int(1.15 * np.sqrt(yield_kt / 15) * 3 / self.approximate_resolution)
        sigma = 1.15 * np.sqrt(yield_kt / 15) / self.approximate_resolution
        x = np.arange(-radius, radius + 1)
        y = np.arange(-radius, radius + 1)
        x, y = np.meshgrid(x, y)

        # Calculate the physical dimensions of pixels
        lat_center = np.mean(self.lats)
        lon_scale = np.cos(np.radians(lat_center))

        # Adjust x coordinates to account for longitude scaling
        x_scaled = x * lon_scale

        # Create the kernel using scaled coordinates
        kernel = np.exp(-(x_scaled**2 + y**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        self.data_averaged = convolve(self.data, kernel, mode="constant")

    def attack_max_fatality(
        self, arsenal, include_injuries=False, non_overlapping=True
    ):
        """
        Attack the country by finding where to detonate a given number of warheads over the country's most populated region.

        Args:
            arsenal (list): a list of the yield of the warheads in kt
            non_overlapping (bool): if True, prohibit overlapping targets as Toon et al.
        """
        self.include_injuries = include_injuries
        if not all(x == arsenal[0] for x in arsenal):
            warnings.warn(
                "Arsenal contains different yield values. The current non-overlapping target allocation algorithm will not handle this correctly."
            )
        self.hit = np.zeros(self.data.shape)
        self.exclude = np.zeros(self.data.shape)
        self.target_list = []
        self.fatalities = []
        self.kilotonne = []

        for yield_kt in arsenal:
            self.attack_next_most_populated_target(yield_kt, non_overlapping)
        return

    def attack_random_non_overlapping(self, arsenal, include_injuries=False):
        """
        Attack the country by detonating a given number of warheads at random locations without overlapping targets.

        Args:
            arsenal (list): a list of the yield of the warheads in kt
        """
        self.include_injuries = include_injuries

        self.hit = np.zeros(self.data.shape)
        self.exclude = np.zeros(self.data.shape)
        self.target_list = []
        self.fatalities = []
        self.kilotonne = []

        # Set exclude to 1 for regions outside the country's borders
        self.exclude[self.data == 0] = 1

        for yield_kt in arsenal:
            self.attack_next_random_target(yield_kt)
        return

    def attack_next_random_target(self, yield_kt):
        """
        Attack a random location in the country that hasn't been hit yet

        Args:
            yield_kt (float): the yield of the warhead in kt
        """
        # Create a mask to exclude previously hit targets
        valid_targets_mask = self.exclude == 0

        # Use the mask to filter the data and find valid target indices
        valid_target_indices = np.argwhere(valid_targets_mask)

        if len(valid_target_indices) > 0:
            # Randomly select a target index from the valid indices
            random_target_index = valid_target_indices[
                np.random.choice(len(valid_target_indices))
            ]
            random_target_lat = self.lats[random_target_index[0]]
            random_target_lon = self.lons[random_target_index[1]]

            self.apply_destruction(random_target_lat, random_target_lon, yield_kt)
            self.target_list.append((random_target_lat, random_target_lon))
            self.kilotonne.append(yield_kt)

            return random_target_lat, random_target_lon
        else:
            # No valid targets remaining
            return None, None

    def attack_next_most_populated_target(self, yield_kt, non_overlapping=True):
        """
        Attack the next most populated region

        Args:
            yield_kt (float): the yield of the warhead in kt
            non_overlapping (bool): if True, prohibit overlapping targets as Toon et al.
        """
        # Create a mask to exclude previously hit targets
        valid_targets_mask = self.exclude == 0

        # Calculate the average population over neighboring cells within a specified radius
        if not non_overlapping or (
            non_overlapping and not hasattr(self, "data_averaged")
        ):
            self.calculate_averaged_population(yield_kt)

        # Use the mask to filter the data and find the maximum population index
        masked_data = np.where(valid_targets_mask, self.data_averaged, np.nan)
        max_population_index = np.unravel_index(
            np.nanargmax(masked_data), self.data.shape
        )
        max_population_lat = self.lats[max_population_index[0]]
        max_population_lon = self.lons[max_population_index[1]]

        self.apply_destruction(
            max_population_lat, max_population_lon, yield_kt, non_overlapping
        )
        self.target_list.append((max_population_lat, max_population_lon))
        self.kilotonne.append(yield_kt)

        return max_population_lat, max_population_lon

    def attack_specific_target(self, lat, lon, yield_kt, CEP, include_injuries=False):
        """
        Attack a specific location in the country, with a circular error probable of the weapon (in meters)

        Args:
            lat (float): the latitude of the target location
            lon (float): the longitude of the target location
            yield_kt (float): the yield of the warhead in kt
            CEP (float): the circular error probable of the weapon (in meters), use 0 for 100% accuracy
        """
        self.include_injuries = include_injuries

        # Calculate distance from intended target using CEP
        distance_from_intended_target = np.random.rayleigh(CEP / 1.1774)

        # Calculate new lat/lon based on random offset
        angle = np.random.uniform(0, 2 * np.pi)
        delta_lat = distance_from_intended_target * np.cos(angle) / 111111
        delta_lon = (
            distance_from_intended_target
            * np.sin(angle)
            / (111111 * np.cos(np.radians(lat)))
        )
        actual_lat = lat + delta_lat
        actual_lon = lon + delta_lon

        self.apply_destruction(actual_lat, actual_lon, yield_kt)
        self.target_list.append((actual_lat, actual_lon))
        self.kilotonne.append(yield_kt)

    def apply_destruction(
        self, lat_groundzero, lon_groundzero, yield_kt, non_overlapping=True
    ):
        """
        Removing the population from the specified location and the max_radius_kill km around it and
        destroy infrastructure within max_radius_burn km around it

        Args:
            lat (float): the latitude of the target location
            lon (float): the longitude of the target location
            yield_kt (float): the yield of the warhead in kt
            non_overlapping (bool): if True, prohibit overlapping targets as Toon et al.
        """
        # (1) Apply population loss and infrastructure destruction
        max_radius_kill = np.sqrt(yield_kt / 15) * 3  # From Toon et al. 2008
        max_radius_burn = np.sqrt(yield_kt / 15) * 2.03  # From Toon et al. 2008, linear scaling of burned area with yield, with 13 km² for Hiroshima
        delta_lon_kill = max_radius_kill / 6371.0 / np.cos(np.radians(lat_groundzero))
        delta_lat_kill = max_radius_kill / 6371.0
        delta_lon_kill = delta_lon_kill * 180 / np.pi
        delta_lat_kill = delta_lat_kill * 180 / np.pi
        delta_lon_burn = max_radius_burn / 6371.0 / np.cos(np.radians(lat_groundzero))
        delta_lat_burn = max_radius_burn / 6371.0
        delta_lon_burn = delta_lon_burn * 180 / np.pi
        delta_lat_burn = delta_lat_burn * 180 / np.pi

        # Create a mask for the box that bounds the destroyed region
        lon_min_kill = lon_groundzero - delta_lon_kill
        lon_max_kill = lon_groundzero + delta_lon_kill
        lat_min_kill = lat_groundzero - delta_lat_kill
        lat_max_kill = lat_groundzero + delta_lat_kill
        lon_indices_kill = np.where(
            (self.lons >= lon_min_kill) & (self.lons <= lon_max_kill)
        )[0]
        lat_indices_kill = np.where(
            (self.lats >= lat_min_kill) & (self.lats <= lat_max_kill)
        )[0]

        # Calculate pixel width and height
        pixel_width = self.lons[1] - self.lons[0]
        pixel_height = self.lats[1] - self.lats[0]

        # Apply the mask to the destroyed region
        for lat_idx_kill in lat_indices_kill:
            for lon_idx_kill in lon_indices_kill:
                lon_pixel = self.lons[lon_idx_kill]
                lat_pixel = self.lats[lat_idx_kill]

                if (lon_pixel - lon_groundzero) ** 2 / delta_lon_kill**2 + (
                    lat_pixel - lat_groundzero
                ) ** 2 / delta_lat_kill**2 <= 1:
                    # kill population
                    population_in_pixel = self.data[lat_idx_kill, lon_idx_kill]
                    distance_from_groundzero = haversine_distance(
                        lat_pixel, lon_pixel, lat_groundzero, lon_groundzero
                    )
                    fatality_rate = get_fatality_rate(
                        distance_from_groundzero, yield_kt, self.include_injuries
                    )
                    self.fatalities.append(fatality_rate * population_in_pixel)
                    self.hit[lat_idx_kill, lon_idx_kill] = 1
                    self.data[lat_idx_kill, lon_idx_kill] = self.data[
                        lat_idx_kill, lon_idx_kill
                    ] * (1 - fatality_rate)

                if (lon_pixel - lon_groundzero) ** 2 / delta_lon_burn**2 + (
                    lat_pixel - lat_groundzero
                ) ** 2 / delta_lat_burn**2 <= 1:
                    # destroy infrastructure
                    self.hit[lat_idx_kill, lon_idx_kill] = 2
                    pixel_box = box(
                        lon_pixel - pixel_width / 2,
                        lat_pixel - pixel_height / 2,
                        lon_pixel + pixel_width / 2,
                        lat_pixel + pixel_height / 2,
                    )
                    overlapping = self.industry.intersects(pixel_box)
                    if overlapping.any():
                        overlapping_ids = self.industry[overlapping].index.tolist()
                        self.destroyed_industrial_areas.extend(overlapping_ids)

        if non_overlapping:
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

            # Apply the mask to the exclusion region
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

    def get_total_destroyed_industrial_area(self):
        """
        Get the total destroyed industrial area as a fraction of the country's total industrial area
        """
        self.destroyed_industrial_areas = list(set(self.destroyed_industrial_areas))
        destroyed_area = self.industry_equal_area[
            self.industry_equal_area.index.isin(self.destroyed_industrial_areas)
        ].geometry.area.sum()
        return destroyed_area / self.total_industry_area

    def plot(
        self,
        show_hit_regions=False,
        show_population_density=False,
        show_industrial_areas=False,
    ):
        """
        Make an interactive map

        Args:
            show_hit_regions (bool): if True, show the hit regions
            show_population_density (bool): if True, show the population density
            show_industrial_areas (bool): if True, show the industrial areas
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

            color_map = LinearColormap(
                colors=[(0, 0, 0, 0), (255, 0, 0, 20), (255, 0, 0, 40)],
                vmin=0,
                vmax=2,
            )

            folium.raster_layers.ImageOverlay(
                image=self.hit,
                bounds=bounds,
                colormap=color_map,
                opacity=0.5,
                mercator_project=True,
            ).add_to(m)

        # plot targets
        if not show_hit_regions:
            for i, target in enumerate(self.target_list):
                folium.Marker(
                    [float(target[0]), float(target[1])],
                    popup=f"Hit with {self.kilotonne[i]} kt",
                    icon=folium.Icon(color="red", icon="radiation"),
                ).add_to(m)

        # Plot industrial areas
        if show_industrial_areas:
            if hasattr(self, "industry"):
                for index, industrial_area in self.industry.iterrows():
                    if index in self.destroyed_industrial_areas:
                        color = "brown"
                    else:
                        color = "purple"
                    polygon_coords = [
                        (coord[1], coord[0])
                        for coord in industrial_area.geometry.exterior.coords[:]
                    ]
                    folium.Polygon(
                        locations=polygon_coords,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.3,
                        opacity=0.5,
                        popup="Industrial Area",
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


def process_chunk(chunk, country, region_data_shape):
    mask_region = gpd.sjoin(chunk, country, how="inner", predicate="intersects").index
    mask_region_bool = np.zeros(region_data_shape, dtype=bool)
    mask_region_bool.ravel()[mask_region] = True
    return mask_region_bool


def run_many_countries(
    scenario,
    degrade=False,
    degrade_factor=1,
    targeting_policy="max_fatality_non_overlapping",
    include_injuries=False,
):
    """
    Run the model for multiple countries and return the results

    Args:
        scenario (dict): a dictionary with the country names as keys and the arsenal as values
        degrade (bool): if True, degrade the LandScan data
        degrade_factor (int): the factor by which to degrade the LandScan data
        targeting_policy (str): the targeting policy to use, either "max_fatality_non_overlapping", "max_fatality", or "random_non_overlapping"
        include_injuries (bool): if True, include fatalities and injuries
    """
    results = []

    for country_name, arsenal in scenario.items():
        country = Country(
            country_name,
            landscan_year=2022,
            degrade=degrade,
            degrade_factor=degrade_factor,
        )
        if targeting_policy == "max_fatality_non_overlapping":
            country.attack_max_fatality(
                arsenal, include_injuries=include_injuries, non_overlapping=True
            )
        elif targeting_policy == "max_fatality":
            country.attack_max_fatality(
                arsenal, include_injuries=include_injuries, non_overlapping=False
            )
        elif targeting_policy == "random_non_overlapping":
            country.attack_random_non_overlapping(
                arsenal, include_injuries=include_injuries
            )
        fatalities = country.get_total_fatalities()
        print(f"{country_name}, fatalities: {fatalities}")

        industry_destroyed_pct = country.get_total_destroyed_industrial_area()
        print(f"{country_name}, industry destroyed: {1000*industry_destroyed_pct:.2f}%")

        # Get ISO3 code for the country
        try:
            iso3 = pycountry.countries.search_fuzzy(country_name)[0].alpha_3
        except LookupError:
            iso3 = "Unknown"  # Use a placeholder if the country is not found

        results.append([iso3, fatalities, industry_destroyed_pct])

    # Save results to CSV
    with open("../results/nuclear_war_fatalities.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["iso3", "population_loss", "industry_destroyed_pct"]
        )  # Write header
        writer.writerows(results)


class IndustrialAreaHandler(osmium.SimpleHandler):
    """
    Handles industrial areas in OSM data.

    Args:
        osmium.SimpleHandler: Base class for handling OSM data
    """

    def __init__(self):
        """
        Initialize IndustrialAreaHandler.
        """
        super(IndustrialAreaHandler, self).__init__()
        self.nodes = {}
        self.industrial_areas = []

    def node(self, n):
        """
        Process OSM nodes.

        Args:
            n: OSM node object
        """
        lon, lat = str(n.location).split("/")
        self.nodes[n.id] = (float(lon), float(lat))

    def way(self, w):
        """
        Process OSM ways to extract industrial areas.

        Args:
            w: OSM way object
        """
        if "landuse" in w.tags and w.tags["landuse"] == "industrial":
            try:
                coords = [self.nodes[n.ref] for n in w.nodes]
                if coords[0] != coords[-1]:  # Ensure the polygon is closed
                    coords.append(coords[0])
                if (
                    len(coords) >= 4
                ):  # A valid polygon needs at least 4 points (3 + 1 to close)
                    polygon = Polygon(coords)
                    self.industrial_areas.append((w.id, polygon))
            except Exception as e:
                print(f"Error processing way {w.id}: {e}")


def get_industrial_areas_from_osm(osm_file_path):
    """
    Extract industrial areas from a local .osm file.

    Args:
        osm_file_path (str): Path to the .osm file

    Returns:
        GeoDataFrame: GeoDataFrame of industrial areas
    """
    handler = IndustrialAreaHandler()
    handler.apply_file(osm_file_path)

    gdf = gpd.GeoDataFrame(
        handler.industrial_areas, columns=["id", "geometry"], crs="EPSG:4326"
    )
    return gdf
