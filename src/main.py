import multiprocessing as mp
import warnings

import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pycountry
import rasterio

from scipy.ndimage import convolve
from shapely.geometry import box
from skimage.measure import block_reduce

from utils import (
    calculate_kill_radius_scaling_factor,
    get_fatality_rate,
    calculate_distance_km,
    calculate_max_radius_burn,
    process_chunk,
)
from osm import get_industrial_areas_from_osm
from risop import get_OPEN_RISOP_nuclear_war_plan

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


class LandScan:
    def __init__(
        self,
        degrade=False,
        degrade_factor=1,
    ):
        """
        Load the LandScan TIF file from the data directory and replace negative values by 0

        Args:
            degrade (bool): if True, degrade the LandScan data
            degrade_factor (int): the factor by which to degrade the LandScan data
        """
        # Open the TIF file from the data directory
        tif_path = f"../data/landscan-global-2022.tif"

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

        assert (
            data.sum() == 7906382407
        ), "The sum of the original data should be equal to 7.9 billion"

        if degrade:
            # Degrade the resolution of the data by summing cells using block reduce
            block_size = (degrade_factor, degrade_factor)
            self.data = block_reduce(data, block_size, np.sum)
            assert (
                self.data.sum() == 7906382407
            ), "The sum of the original data should be equal to 7.9 billion"
        else:
            self.data = data

        return


class Country:
    def __init__(
        self,
        country_name,
        degrade=False,
        degrade_factor=1,
        subregion=None,
        industry=True,
        burn_radius_prescription="default",
        kill_radius_prescription="default",
        avoid_border_regions=False,
    ):
        """
        Load the population data for the specified country

        Args:
            country_name (str): the name of the country
            degrade (bool): if True, degrade the LandScan data
            degrade_factor (int): the factor by which to degrade the LandScan data
            subregion (list): the bounds of the subregion to extract from the LandScan data in the format
             [min_lon, max_lon, min_lat, max_lat]. This is optional and if not provided, the entire country is used.
            industry (bool): if True, then load industrial zones
            burn_radius_prescription (str): the method to calculate the burn radius ("Toon", "default", or "overpressure")
            kill_radius_prescription (str): the method to calculate the kill radius ("Toon", "default", or "overpressure")
            avoid_border_regions (bool): if True, prohibit targets in regions close to the country's border, implemented
                for the US-Mexico border.
        """
        self.country_name = country_name

        # Load landscan data
        landscan = LandScan(degrade, degrade_factor)

        self.burn_radius_prescription = burn_radius_prescription
        self.kill_radius_prescription = kill_radius_prescription
        self.avoid_border_regions = avoid_border_regions

        self.approximate_resolution = 1 * self.degrade_factor  # km

        # Get the geometry of the specified country
        country = gpd.read_file("../data/natural-earth/ne_10m_admin_0_countries.shp")
        if country_name == "Czech Republic":
            country = country[country.ADMIN == "Czechia"]
        else:
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

        # Calculate grid spacing and store as instance variables
        if len(self.lats) >= 2:
            self.delta_lat = abs(self.lats[1] - self.lats[0])
        else:
            raise ValueError("Grid is too small to calculate grid spacing")

        if len(self.lons) >= 2:
            self.delta_lon = abs(self.lons[1] - self.lons[0])
        else:
            raise ValueError("Grid is too small to calculate grid spacing")

        self.data = population_data_country.copy()
        self.population_intact = population_data_country.copy()

        # This will be used to store the hit locations
        self.hit = np.zeros(population_data_country.shape)

        # This will be used to exclude regions from attack
        self.exclude = np.zeros(population_data_country.shape)

        self.target_list = []
        self.fatalities = []
        self.kilotonne = []

        # Get ISO3 code for the country
        try:
            self.iso3 = pycountry.countries.search_fuzzy(country_name)[0].alpha_3
        except LookupError:
            self.iso3 = "Unknown"  # Use a placeholder if the country is not found

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

        # Delete unused variables to free up memory
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
            kill_radius_prescription (str): The method to calculate the kill radius. Options are:
                - "Toon": Uses the formula from Toon et al. 2008, which assumes sqrt
                - "default": Uses a scaling based on a model described in scripts/burn-radius-scaling.ipynb
                - "overpressure": Uses a scaling based on the overpressure model

        Returns:
            None. Sets self.data_averaged with the convolved population data.
        """
        scaling_factor = self.calculate_kill_radius_scaling_factor(
            self.kill_radius_prescription, yield_kt
        )
        radius = int(1.15 * scaling_factor * 3 / self.approximate_resolution)
        sigma = 1.15 * scaling_factor / self.approximate_resolution
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

        assert (
            abs(self.data_averaged.sum() - self.data.sum()) < 0.01 * self.data.sum()
        ), "The sum of the smoothed data should be within 1% of the sum of the original data"

    def attack_max_fatality(
        self,
        arsenal,
        include_injuries=False,
        airburst=True,
    ):
        """
        Attack the country by finding where to detonate a given number of warheads over the country's most populated region.

        Args:
            arsenal (list): a list of the yield of the warheads in kt
            airburst (bool): if True, use air bursts; if False, use ground bursts
        """
        arsenal = sorted(arsenal, reverse=True)
        self.include_injuries = include_injuries
        self.hit = np.zeros(self.data.shape)
        self.exclude = np.zeros(self.data.shape)

        # Exclude regions close to the country's border (optional)
        if self.avoid_border_regions:
            mask_zero = (self.data == 0).astype(int)
            kernel = np.ones((3, 3))
            convolved = convolve(mask_zero, kernel, mode="constant", cval=0)
            self.exclude[convolved > 0] = 1

        self.target_list = []
        self.fatalities = []
        self.kilotonne = []

        for yield_kt in arsenal:
            self.attack_next_most_populated_target(yield_kt, airburst=airburst)
        return

    def apply_OPEN_RISOP_nuclear_war_plan(
        self,
        include_injuries=False,
        ignore_military=False,
        ignore_dual_use=False,
        ignore_war_supporting=False,
        ignore_critical=False,
        ignore_other_civilian=False,
        icbm_only=False,
    ):
        """
        Attack all locations in the OPEN RISOP database. Only valid for the US.

        Args:
            include_injuries (bool): if True, include injuries in the fatality calculation
            ignore_military (bool): if True, ignore military targets
            ignore_dual_use (bool): if True, ignore dual-use targets
            ignore_war_supporting (bool): if True, ignore war-supporting industry targets
            ignore_critical (bool): if True, ignore critical infrastructure targets
            ignore_other_civilian (bool): if True, ignore other civilian targets
            icbm_only (bool): if True, only attack ICBM targets
        """
        if self.country_name != "United States of America":
            raise ValueError("OPEN RISOP nuclear war plan only valid for the US")

        self.hit = np.zeros(self.data.shape)
        self.target_list = []
        self.fatalities = []
        self.kilotonne = []

        targets = get_OPEN_RISOP_nuclear_war_plan(
            ignore_military=ignore_military,
            ignore_dual_use=ignore_dual_use,
            ignore_war_supporting=ignore_war_supporting,
            ignore_critical=ignore_critical,
            ignore_other_civilian=ignore_other_civilian,
            icbm_only=icbm_only,
        )
        for _, (lat, lon, hob, ykt) in targets.items():
            if hob == 0:
                airburst = False
            else:
                airburst = True
            self.attack_specific_target(
                lat, lon, ykt, include_injuries=include_injuries, airburst=airburst
            )
        return

    def attack_next_most_populated_target(self, yield_kt, airburst=True):
        """
        Attack the next most populated region.

        Args:
            yield_kt (float): the yield of the warhead in kt
            airburst (bool): if True, use air bursts; if False, use ground bursts
        """
        # Create a mask to exclude previously hit targets (if applicable)
        valid_targets_mask = self.exclude == 0

        # Calculate the average population over neighboring cells within a specified radius
        self.calculate_averaged_population(yield_kt)

        # Use the mask to filter the data and find the maximum population index
        masked_data = np.where(valid_targets_mask, self.data_averaged, np.nan)
        max_population_index = np.unravel_index(
            np.nanargmax(masked_data), self.data.shape
        )
        max_population_lat = self.lats[max_population_index[0]]
        max_population_lon = self.lons[max_population_index[1]]

        self.apply_destruction(
            max_population_lat,
            max_population_lon,
            yield_kt,
            airburst=airburst,
        )

        self.target_list.append((max_population_lat, max_population_lon))
        self.kilotonne.append(yield_kt)

        return max_population_lat, max_population_lon

    def attack_specific_target(
        self, lat, lon, yield_kt, CEP=0, include_injuries=False, airburst=True
    ):
        """
        Attack a specific location in the country, with a circular error probable of the weapon (in meters).

        Args:
            lat (float): the latitude of the target location
            lon (float): the longitude of the target location
            yield_kt (float): the yield of the warhead in kt
            CEP (float): the circular error probable of the weapon (in meters), use 0 for 100% accuracy
            airburst (bool): if True, use air bursts; if False, use ground bursts
        """
        self.include_injuries = include_injuries

        # Calculate distance from intended target using CEP
        distance_from_intended_target = np.random.rayleigh(CEP / 1.1774)

        # Calculate new lat/lon based on random offset
        angle = np.random.uniform(0, 2 * np.pi)
        delta_lat = distance_from_intended_target * np.cos(angle) / 111320
        delta_lon = (
            distance_from_intended_target
            * np.sin(angle)
            / (111320 * np.cos(np.radians(lat)))
        )
        actual_lat = lat + delta_lat
        actual_lon = lon + delta_lon

        self.apply_destruction(actual_lat, actual_lon, yield_kt, airburst=airburst)

        self.target_list.append((actual_lat, actual_lon))
        self.kilotonne.append(yield_kt)

    def apply_destruction(
        self,
        lat_groundzero,
        lon_groundzero,
        yield_kt,
        airburst=True,
    ):
        """
        Removing the population from the specified location and the max_radius_kill km around it and
        destroy infrastructure within max_radius_burn km around it.

        Args:
            lat (float): the latitude of the target location
            lon (float): the longitude of the target location
            yield_kt (float): the yield of the warhead in kt
            airburst (bool): True for air bursts, False for ground bursts; affects scaling factors
        """
        # This determines the max radius over which we apply population loss
        # The fatality rate nears 0 at this radius
        max_radius_kill = 3 * calculate_kill_radius_scaling_factor(
            self.kill_radius_prescription, yield_kt, airburst=airburst
        )

        # This determines the max radius over which we apply burn damage
        max_radius_burn = calculate_max_radius_burn(
            self.burn_radius_prescription, yield_kt
        )

        # Convert kill and burn radii to lat/lon deltas, this is only used for the mask
        # and is not directly used for the fatality/destruction calculation
        delta_lat_kill = max_radius_kill / 6371.0 * 180 / np.pi
        delta_lon_kill = delta_lat_kill / np.cos(np.radians(lat_groundzero))
        delta_lat_burn = max_radius_burn / 6371.0 * 180 / np.pi
        delta_lon_burn = delta_lat_burn / np.cos(np.radians(lat_groundzero))

        # Create a mask for the box that bounds the destroyed region
        lon_min_kill, lon_max_kill = (
            lon_groundzero - delta_lon_kill,
            lon_groundzero + delta_lon_kill,
        )
        lat_min_kill, lat_max_kill = (
            lat_groundzero - delta_lat_kill,
            lat_groundzero + delta_lat_kill,
        )
        lon_indices_kill = np.where(
            (self.lons >= lon_min_kill) & (self.lons <= lon_max_kill)
        )[0]
        lat_indices_kill = np.where(
            (self.lats >= lat_min_kill) & (self.lats <= lat_max_kill)
        )[0]

        # Calculate pixel width and height
        pixel_width = abs(self.lons[1] - self.lons[0])
        pixel_height = abs(self.lats[1] - self.lats[0])

        # Apply the mask to the destroyed region
        for lat_idx_kill in lat_indices_kill:
            for lon_idx_kill in lon_indices_kill:
                lon_pixel = self.lons[lon_idx_kill]
                lat_pixel = self.lats[lat_idx_kill]

                if (lon_pixel - lon_groundzero) ** 2 / delta_lon_kill**2 + (
                    lat_pixel - lat_groundzero
                ) ** 2 / delta_lat_kill**2 <= 1:
                    # apply fatality rate to population in pixel
                    population_in_pixel = self.data[lat_idx_kill, lon_idx_kill]
                    distance_from_groundzero = calculate_distance_km(
                        lat_pixel, lon_pixel, lat_groundzero, lon_groundzero
                    )
                    fatality_rate = get_fatality_rate(
                        distance_from_groundzero,
                        yield_kt,
                        self.include_injuries,
                        self.kill_radius_prescription,
                        airburst=airburst,
                    )
                    self.fatalities.append(fatality_rate * population_in_pixel)
                    self.hit[lat_idx_kill, lon_idx_kill] = 1
                    self.data[lat_idx_kill, lon_idx_kill] = self.data[
                        lat_idx_kill, lon_idx_kill
                    ] * (1 - fatality_rate)

                if (lon_pixel - lon_groundzero) ** 2 / delta_lon_burn**2 + (
                    lat_pixel - lat_groundzero
                ) ** 2 / delta_lat_burn**2 <= 1:
                    # apply burn
                    distance_from_groundzero = calculate_distance_km(
                        lat_pixel, lon_pixel, lat_groundzero, lon_groundzero
                    )
                    # only burn if within burn radius and not already burned
                    if (
                        distance_from_groundzero <= max_radius_burn
                        and self.hit[lat_idx_kill, lon_idx_kill] != 2
                    ):
                        self.hit[lat_idx_kill, lon_idx_kill] = 2  # mark as burned
                        # check if any industry is in the pixel
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
        return

    def get_total_destroyed_industrial_area(self, max_area_km2=10):
        """
        Get the total destroyed industrial area as a fraction of the country's total industrial area.
        Excludes polygons larger than max_area_km2.

        Args:
            max_area_km2 (float): Maximum area in square kilometers to include in calculation.
                                 Defaults to 10 km², which manual inspection suggests is a reasonable
                                 threshold for excluding quarries, proving grounds, and errors that
                                 cause some polygons to count for too much in the total.
        """
        self.destroyed_industrial_areas = list(set(self.destroyed_industrial_areas))

        # Filter out large polygons from both numerator and denominator
        # Convert km² to m² (1e6 m² = 1 km²)
        max_area_m2 = max_area_km2 * 1e6
        small_polygons = self.industry_equal_area[
            self.industry_equal_area.geometry.area <= max_area_m2
        ]

        destroyed_area = small_polygons[
            small_polygons.index.isin(self.destroyed_industrial_areas)
        ].geometry.area.sum()

        total_area = small_polygons.geometry.area.sum()

        return destroyed_area / total_area if total_area > 0 else 0

    def print_diagnostic_info(self):
        """
        Print diagnostic information
        """
        total = int(sum(self.fatalities))
        print(
            f"Total fatalities: {total} ({total/self.population_intact.sum()*100:.1f}%)"
        )
        print(
            f"Total destroyed industrial area: {100*self.get_total_destroyed_industrial_area():.1f}%"
        )

    def plot(
        self,
        show_burn_regions=False,
        show_industrial_areas=False,
        ms=2,
    ):
        """
        Make an interactive map
        Args:
            show_burn_regions (bool): if True, show the burn regions
            show_industrial_areas (bool): if True, show the industrial areas
            ms (float): the size of the markers
        """

        # Create a folium map centered around the average coordinates
        avg_lat = np.mean(self.lats)
        avg_lon = np.mean(self.lons)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5, tiles="OpenStreetMap")

        # Show hit regions
        if show_burn_regions:
            # Reproject coordinates from PlateCarre to Mercator

            bounds = [
                [np.min(self.lats), np.min(self.lons)],
                [np.max(self.lats), np.max(self.lons)],
            ]

            # Create a custom colormap
            colors = [(0, 0, 0, 0), (1, 0, 0, 1)]  # Transparent and solid red
            n_bins = 2  # We only need 2 bins: 0 and 1
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

            # Create a binary mask for burn regions
            burn_mask = (self.hit == 2).astype(float)

            # Apply colormap to data
            colored_data = cmap(burn_mask)

            bounds = [
                [np.min(self.lats), np.min(self.lons)],
                [np.max(self.lats), np.max(self.lons)],
            ]

            folium.raster_layers.ImageOverlay(
                image=colored_data,
                bounds=bounds,
                opacity=0.5,
                mercator_project=True,
            ).add_to(m)

        # plot targets
        if not show_burn_regions:
            for i, target in enumerate(self.target_list):
                folium.CircleMarker(
                    [float(target[0]), float(target[1])],
                    radius=ms * np.sqrt(self.kilotonne[i] / 100),
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.5,
                    opacity=0.5,
                    popup=f"Hit with {self.kilotonne[i]} kt",
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
