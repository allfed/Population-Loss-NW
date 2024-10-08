import csv
import json
import math
import multiprocessing as mp
import os
import re
import warnings

import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import osmium
import pandas as pd
import pycountry
import rasterio
import scipy.sparse as sparse

from branca.colormap import LinearColormap
from geopy import distance
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
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
        burn_radius_prescription="default",
        kill_radius_prescription="default",
        avoid_border_regions=False,
        fallout_grid_factor=5,
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
            burn_radius_prescription (str): the method to calculate the burn radius ("Toon", "default", or "overpressure")
            kill_radius_prescription (str): the method to calculate the kill radius ("Toon", "default", or "overpressure")
            avoid_border_regions (bool): if True, prohibit targets in regions close to the country's border, implemented
                for the US-Mexico border.
            fallout_grid_factor (int): the factor by which to refine the fallout grid
        """
        self.country_name = country_name
        self.fallout_grid_factor = fallout_grid_factor

        # Load landscan data
        self.degrade_factor = degrade_factor
        if use_HD:
            country_HD = country_name
        else:
            country_HD = None
        landscan = LandScan(landscan_year, degrade, degrade_factor, use_HD, country_HD)

        self.burn_radius_prescription = burn_radius_prescription
        self.kill_radius_prescription = kill_radius_prescription
        self.avoid_border_regions = avoid_border_regions

        self.approximate_resolution = 1 * self.degrade_factor  # km
        if use_HD:
            self.approximate_resolution = 0.09 * self.degrade_factor  # km

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

        # For France, only include European territories
        if country_name == "France":
            min_lat_idx = np.argmin(np.abs(lats - 51))
            max_lat_idx = np.argmin(np.abs(lats - 42))
            min_lon_idx = np.argmin(np.abs(lons + 6))
            max_lon_idx = np.argmin(np.abs(lons - 10))

        # For New Zealand, only include the main islands
        if country_name == "New Zealand":
            min_lat_idx = np.argmin(np.abs(lats + 34))
            max_lat_idx = np.argmin(np.abs(lats + 46))
            min_lon_idx = np.argmin(np.abs(lons - 165))
            max_lon_idx = np.argmin(np.abs(lons - 178))

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
            self.delta_lat = 0.0

        if len(self.lons) >= 2:
            self.delta_lon = abs(self.lons[1] - self.lons[0])
        else:
            self.delta_lon = 0.0

        self.data = population_data_country.copy()
        self.population_intact = population_data_country.copy()

        # This will be used to store the hit locations
        self.hit = np.zeros(population_data_country.shape)

        # This will be used to exclude regions from attack
        self.exclude = np.zeros(population_data_country.shape)

        # Initialize the finer fallout grid as a sparse matrix
        fallout_shape = (
            self.data.shape[0] * self.fallout_grid_factor,
            self.data.shape[1] * self.fallout_grid_factor,
        )

        # Used to store the radiation dose accumulated over some time (default is 4 days)
        self.fallout = sparse.lil_matrix(fallout_shape)

        # Used to store the radiation dose rate a long time after the initial detonation (1 year is the default)
        self.fallout_longterm = sparse.lil_matrix(fallout_shape)

        # Create finer latitude and longitude arrays for the fallout grid
        self.fallout_lats = np.linspace(self.lats[0], self.lats[-1], fallout_shape[0])
        self.fallout_lons = np.linspace(self.lons[0], self.lons[-1], fallout_shape[1])

        self.target_list = []
        self.fatalities = []
        self.kilotonne = []
        self.soot_Tg = 0

        # Get ISO3 code for the country
        try:
            self.iso3 = pycountry.countries.search_fuzzy(country_name)[0].alpha_3
        except LookupError:
            self.iso3 = "Unknown"  # Use a placeholder if the country is not found

        # Initialize an empty list to store DataFrames for each CSV file
        dfs = []

        # Iterate through CSV files in the custom locations directory
        custom_locations_dir = "../data/custom-locations"
        for filename in os.listdir(custom_locations_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(custom_locations_dir, filename)

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Select rows where iso3 matches self.iso3
                df_filtered = df[df["iso3"] == self.iso3][
                    ["name", "latitude", "longitude"]
                ]

                # Append the filtered DataFrame to the list
                dfs.append(df_filtered)

        # Concatenate all DataFrames in the list
        self.custom_locations = pd.concat(dfs, ignore_index=True)
        self.custom_locations["status"] = "intact"

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
            kill_radius_prescription (str): The method to calculate the kill radius. Options are:
                - "Toon": Uses the formula from Toon et al. 2008, which assumes sqrt
                - "default": Uses a scaling based on a model described in burn-radius-scaling.ipynb
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

    def attack_max_fatality(
        self,
        arsenal,
        include_injuries=False,
        non_overlapping=True,
        airburst=True,
    ):
        """
        Attack the country by finding where to detonate a given number of warheads over the country's most populated region.
        Uses air bursts where we assume there is no radiation fallout.

        Args:
            arsenal (list): a list of the yield of the warheads in kt
            non_overlapping (bool): if True, prohibit overlapping targets as Toon et al.
            airburst (bool): if True, use air bursts where we assume there is no radiation fallout
        """
        arsenal = sorted(arsenal, reverse=True)
        self.include_injuries = include_injuries
        if not all(x == arsenal[0] for x in arsenal) and non_overlapping:
            warnings.warn(
                "Arsenal contains different yield values. The current non-overlapping target allocation algorithm will not handle this correctly."
            )
        self.hit = np.zeros(self.data.shape)
        self.exclude = np.zeros(self.data.shape)

        if self.avoid_border_regions:
            mask_zero = (self.data == 0).astype(int)
            kernel = np.ones((3, 3))
            convolved = convolve(mask_zero, kernel, mode="constant", cval=0)
            self.exclude[convolved > 0] = 1

        self.target_list = []
        self.fatalities = []
        self.kilotonne = []

        for yield_kt in arsenal:
            self.attack_next_most_populated_target(
                yield_kt, non_overlapping, airburst=airburst
            )
        return

    def attack_random_non_overlapping(self, arsenal, include_injuries=False):
        """
        Attack the country by detonating a given number of warheads at random locations without overlapping targets.
        Uses air bursts where we assume there is no radiation fallout.

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

    def attack_next_random_target(self, yield_kt, airburst=True):
        """
        Attack a random location in the country that hasn't been hit yet.
        Uses air bursts where we assume there is no radiation fallout.

        Args:
            yield_kt (float): the yield of the warhead in kt
            airburst (bool): if True, use air bursts where we assume there is no radiation fallout
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

            self.apply_destruction(
                random_target_lat, random_target_lon, yield_kt, airburst=airburst
            )
            if not airburst:
                self.add_fallout(random_target_lat, random_target_lon, yield_kt)
            self.target_list.append((random_target_lat, random_target_lon))
            self.kilotonne.append(yield_kt)

            return random_target_lat, random_target_lon
        else:
            # No valid targets remaining
            return None, None

    def attack_next_most_populated_target(
        self, yield_kt, non_overlapping=True, airburst=True
    ):
        """
        Attack the next most populated region.
        Uses air bursts where we assume there is no radiation fallout.

        Args:
            yield_kt (float): the yield of the warhead in kt
            non_overlapping (bool): if True, prohibit overlapping targets as Toon et al.
            airburst (bool): if True, use air bursts where we assume there is no radiation fallout
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
            max_population_lat,
            max_population_lon,
            yield_kt,
            non_overlapping,
            airburst=airburst,
        )

        if not airburst:
            self.add_fallout(max_population_lat, max_population_lon, yield_kt)

        self.target_list.append((max_population_lat, max_population_lon))
        self.kilotonne.append(yield_kt)

        return max_population_lat, max_population_lon

    def apply_1956_US_nuclear_war_plan(self, yield_kt, include_injuries=False):
        """
        Attack all locations within the country's borders using the 1956 US nuclear war plan

        Args:
            yield_kt (float): the yield in kt for all warheads used
            include_injuries (bool): if True, include injuries in the fatality calculation
        """
        self.hit = np.zeros(self.data.shape)
        self.target_list = []
        self.fatalities = []
        self.kilotonne = []

        targets = get_1956_US_nuclear_war_plan(self.country_name)
        for city, (lat, lon) in targets.items():
            self.attack_specific_target(
                lat, lon, yield_kt, include_injuries=include_injuries
            )
        return

    def apply_OPEN_RISOP_nuclear_war_plan(
        self,
        include_injuries=False,
        ignore_military=False,
        ignore_dual_use=False,
        ignore_war_supporting=False,
        ignore_critical=False,
        ignore_other_civilian=False,
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
        )
        for city, (lat, lon, hob, ykt) in targets.items():
            if hob == 0:
                airburst = False
            else:
                airburst = True
            self.attack_specific_target(
                lat, lon, ykt, include_injuries=include_injuries, airburst=airburst
            )
        return
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
            airburst (bool): if True, use air bursts where we assume there is no radiation fallout
                if False, use ground bursts and include radiation fallout
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

        if not airburst:
            self.add_fallout(actual_lat, actual_lon, yield_kt)

        self.target_list.append((actual_lat, actual_lon))
        self.kilotonne.append(yield_kt)

    def add_fallout(
        self,
        lat_groundzero,
        lon_groundzero,
        yield_kt,
        threshold_rads=10,
        evaluate_longterm_dose_rate=True,
    ):
        """
        Add fallout radiation dose to each pixel based on the ground zero location and weapon yield.
        Dynamically determines the area of significant radiation. Wind is assumed to blow eastward
        in the northern hemisphere, and westward in the southern hemisphere.

        Args:
            lat_groundzero (float): Latitude of the ground zero.
            lon_groundzero (float): Longitude of the ground zero.
            yield_kt (float): Yield of the weapon in kilotons.
            threshold_rads (float): Radiation threshold in rads to determine significant area,
                calculations are only performed in that region for efficiency. This should be small
                enough so that even with many warheads, the dose will not exceed the lethal dose.
            evaluate_longterm_dose_rate (bool): if True, evaluate the longterm dose rate, useful to
                evaluate the exclusion area after the war
        """
        # Determine wind direction based on hemisphere
        hemisphere = "north" if lat_groundzero >= 0 else "south"
        wind_direction_deg = 90 if hemisphere == "north" else 270  # East or West

        # Function to find distance where radiation falls below threshold
        def find_threshold_distance(direction):
            distance = 1  # Start with 1 km
            while True:
                if direction == "downwind":
                    dose = calculate_total_dose(distance, 0.1, yield_kt)
                else:  # perpendicular
                    dose = calculate_total_dose(0.1, distance, yield_kt)

                if dose < threshold_rads:
                    return distance
                distance *= 1.5  # Increase distance by 50% each iteration

        # Find threshold distances
        downwind_distance = find_threshold_distance("downwind")
        perpendicular_distance = find_threshold_distance("perpendicular")

        # Calculate lat/lon bounds
        km_per_degree_lat = 111.32  # Approximate
        km_per_degree_lon = 111.32 * np.cos(np.radians(lat_groundzero))

        lat_range = perpendicular_distance / km_per_degree_lat
        lon_range = downwind_distance / km_per_degree_lon

        lat_min = max(lat_groundzero - lat_range, min(self.fallout_lats))
        lat_max = min(lat_groundzero + lat_range, max(self.fallout_lats))

        if wind_direction_deg == 90:  # East wind
            lon_min = lon_groundzero
            lon_max = min(lon_groundzero + lon_range, max(self.fallout_lons))
        else:  # West wind
            lon_min = max(lon_groundzero - lon_range, min(self.fallout_lons))
            lon_max = lon_groundzero

        # Adjust longitudes to handle meridian crossing
        if lon_max > 180:
            lon_max -= 360
        if lon_min < -180:
            lon_min += 360

        # Get indices in the finer grid
        lat_indices = np.where(
            (self.fallout_lats >= lat_min) & (self.fallout_lats <= lat_max)
        )[0]
        lon_indices = np.where(
            (self.fallout_lons >= lon_min) & (self.fallout_lons <= lon_max)
        )[0]

        # Create meshgrid of latitude and longitude indices
        lat_mesh, lon_mesh = np.meshgrid(
            self.fallout_lats[lat_indices],
            self.fallout_lons[lon_indices],
            indexing="ij",
        )

        # Compute distances and bearings using vectorized operations
        distances, bearings = calculate_distance_km_vectorized(
            lat_groundzero, lon_groundzero, lat_mesh, lon_mesh
        )

        angle_diff_rad = np.radians(bearings - wind_direction_deg)
        downwind_distances = distances * np.cos(angle_diff_rad)
        perpendicular_distances = distances * np.sin(angle_diff_rad)

        # Patch to avoid "zero" bugs
        downwind_distances[abs(downwind_distances) < 0.1] = 0.1
        perpendicular_distances[
            (perpendicular_distances > 0) & (perpendicular_distances < 0.1)
        ] = 0.1
        perpendicular_distances[
            (perpendicular_distances < 0) & (perpendicular_distances > -0.1)
        ] = -0.1

        # Create a mask where downwind_distance > 0
        mask = downwind_distances > 0

        # Compute doses where downwind_distance is positive
        doses = np.zeros_like(distances)
        doses[mask] = calculate_total_dose(
            downwind_distances[mask], perpendicular_distances[mask], yield_kt
        )

        # Add the calculated doses to the fallout sparse matrix
        nonzero_indices = np.where(doses > 0)
        i_indices = lat_indices[nonzero_indices[0]]
        j_indices = lon_indices[nonzero_indices[1]]
        self.fallout[i_indices, j_indices] += doses[nonzero_indices]

        if evaluate_longterm_dose_rate:
            longterm_dose_rate = np.zeros_like(distances)
            longterm_dose_rate[mask] = calculate_longterm_dose_rate(
                downwind_distances[mask], perpendicular_distances[mask], yield_kt
            )
            nonzero_indices = np.where(longterm_dose_rate > 0)
            i_indices = lat_indices[nonzero_indices[0]]
            j_indices = lon_indices[nonzero_indices[1]]
            self.fallout_longterm[i_indices, j_indices] += longterm_dose_rate[
                nonzero_indices
            ]

        return

    @staticmethod
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
            # Most realistic model, see burn-radius-scaling.ipynb
            return 0.75094351 * yield_kt**0.38287688
        elif burn_radius_prescription == "overpressure":
            # Scales from average of Hiroshima and Nagasaki (13km² and 6.7km², 15kt and 21kt),
            # also assumes that it scales like D**(1/3)
            return 1.77 * (yield_kt / 18) ** (1 / 3)
        else:
            raise ValueError(
                f"Unknown burn radius prescription: {burn_radius_prescription}"
            )

    @staticmethod
    def calculate_kill_radius_scaling_factor(
        scaling_prescription, yield_kt, airburst=True
    ):
        """
        Calculate the scaling factor for the kill radius. This is only
        really used to enforce the non-overlapping targets condition, the actual fatalities are
        calculated with get_fatality_rate.

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

    def apply_destruction(
        self,
        lat_groundzero,
        lon_groundzero,
        yield_kt,
        non_overlapping=True,
        airburst=True,
    ):
        """
        Removing the population from the specified location and the max_radius_kill km around it and
        destroy infrastructure within max_radius_burn km around it.
        Uses air bursts where we assume there is no radiation fallout.

        Args:
            lat (float): the latitude of the target location
            lon (float): the longitude of the target location
            yield_kt (float): the yield of the warhead in kt
            non_overlapping (bool): if True, prohibit overlapping targets as Toon et al.
            airburst (bool): True for air bursts, False for ground bursts; affects scaling factors
        """
        # (1) Apply population loss and infrastructure destruction
        max_radius_kill = 3 * self.calculate_kill_radius_scaling_factor(
            self.kill_radius_prescription, yield_kt, airburst=airburst
        )
        max_radius_burn = self.calculate_max_radius_burn(
            self.burn_radius_prescription, yield_kt
        )

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
                    # kill population
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
                    # destroy infrastructure
                    distance_from_groundzero = calculate_distance_km(
                        lat_pixel, lon_pixel, lat_groundzero, lon_groundzero
                    )
                    if distance_from_groundzero <= max_radius_burn:
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

                        # Calculate soot emissions
                        population_in_pixel = self.population_intact[
                            lat_idx_kill, lon_idx_kill
                        ]
                        area_in_pixel = (
                            pixel_width
                            * pixel_height
                            * 111.32
                            * 111.32
                            * np.cos(np.radians(lat_pixel))
                        )
                        soot_emissions_in_pixel = (
                            area_in_pixel * 1.3e5 + population_in_pixel * 1.8e2
                        )  # kg soot per pixel, from Toon et al. 2008

                        self.soot_Tg += soot_emissions_in_pixel * 1e-9

                    # Check for destroyed custom locations
                    for _, row in self.custom_locations.iterrows():
                        custom_location_distance = calculate_distance_km(
                            lat_groundzero, lon_groundzero, row.latitude, row.longitude
                        )
                        if custom_location_distance <= max_radius_burn:
                            # Change the status of the custom location to "destroyed"
                            self.custom_locations.loc[
                                (self.custom_locations["latitude"] == row.latitude)
                                & (self.custom_locations["longitude"] == row.longitude),
                                "status",
                            ] = "destroyed"

        if non_overlapping:
            # (2) Now we apply another mask to make sure there is no overlap with future nukes
            max_radius = max_radius_kill * 2
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

    def calculate_contaminated_area(self, threshold=0.002):
        """
        Calculate the total land area where the long-term fallout exceeds a given threshold.

        Args:
            threshold (float): The threshold value for long-term fallout, in rads/hr. Default is 0.002.
                The time at which this is evaluated depends on the settings for calculate_longterm_dose_rate,
                but is 1 year after the attack by default.

        Returns:
            float: The total contaminated area in square kilometers.
        """
        # Convert fallout_longterm to dense array for easier processing
        fallout_longterm_dense = self.fallout_longterm.toarray()

        # Create a mask of areas exceeding the threshold
        contaminated_mask = fallout_longterm_dense > threshold

        # Calculate the area of each cell
        # We'll use the haversine formula to calculate the area of each cell
        contaminated_area = 0
        for i in range(contaminated_mask.shape[0]):
            for j in range(contaminated_mask.shape[1]):
                if contaminated_mask[i, j]:
                    # Calculate the corners of the cell
                    lat1, lat2 = self.fallout_lats[i], self.fallout_lats[i + 1]
                    lon1, lon2 = self.fallout_lons[j], self.fallout_lons[j + 1]

                    # Calculate the area of the cell
                    cell_area = calculate_cell_area(lat1, lon1, lat2, lon2)
                    contaminated_area += cell_area

        return contaminated_area

    def get_total_fatalities(self):
        """
        Get the total fatalities, including both immediate fatalities from blast/fire/prompt radiation
        and near-term fatalities from radiation fallout. Will include both fatalities and injuries
        if include_injuries is True.
        """
        immediate_fatalities = int(sum(self.fatalities))

        # Convert fallback fallout to COO format for easier iteration
        fallout_coo = self.fallout.tocoo()

        radiation_fatalities = 0
        for i, j, radiation_dose in zip(
            fallout_coo.row, fallout_coo.col, fallout_coo.data
        ):
            fatality_rate = get_fatality_rate_fallout(radiation_dose)
            population = self.data[
                i // self.fallout_grid_factor, j // self.fallout_grid_factor
            ]
            radiation_fatalities += fatality_rate * (
                population / self.fallout_grid_factor**2
            )

        radiation_fatalities = int(radiation_fatalities)

        total_fatalities = immediate_fatalities + radiation_fatalities

        return total_fatalities, immediate_fatalities, radiation_fatalities

    def get_total_destroyed_industrial_area(self):
        """
        Get the total destroyed industrial area as a fraction of the country's total industrial area
        """
        self.destroyed_industrial_areas = list(set(self.destroyed_industrial_areas))
        destroyed_area = self.industry_equal_area[
            self.industry_equal_area.index.isin(self.destroyed_industrial_areas)
        ].geometry.area.sum()
        return destroyed_area / self.total_industry_area

    def get_number_destroyed_custom_locations(self):
        """
        Get the number of destroyed custom locations with uncertainty
        """
        number_destroyed = len(
            self.custom_locations[self.custom_locations["status"] == "destroyed"]
        )
        total_locations = len(self.custom_locations)
        fraction_destroyed = number_destroyed / total_locations
        uncertainty = np.sqrt(
            fraction_destroyed * (1 - fraction_destroyed) / total_locations
        )
        percentage = fraction_destroyed * 100
        uncertainty_percentage = uncertainty * 100
        return f"{number_destroyed} custom locations destroyed out of {total_locations} ({percentage:.1f}% ± {uncertainty_percentage:.1f}%)"

    def print_diagnostic_info(self):
        """
        Print diagnostic information
        """
        total, immediate, radiation = self.get_total_fatalities()
        print(
            f"Total fatalities: {total} ({total/self.population_intact.sum()*100:.1f}%), of which {radiation/total*100:.1f}% are from radiation fallout"
        )
        print(
            f"Total destroyed industrial area: {100*self.get_total_destroyed_industrial_area():.1f}%"
        )
        print(f"Soot emissions: {self.soot_Tg:.1f} Tg")

    def plot(
        self,
        show_burn_regions=False,
        show_population_density=False,
        show_industrial_areas=False,
        show_custom_locations=False,
        show_fallout=False,
        show_fallout_longterm=False,
        ms=2,
    ):
        """
        Make an interactive map
        Args:
            show_burn_regions (bool): if True, show the burn regions
            show_population_density (bool): if True, show the population density
            show_industrial_areas (bool): if True, show the industrial areas
            show_custom_locations (bool): if True, show the custom locations from data/custom-locations/*.csv
            show_fallout (bool): if True, show the total dose of radiation in each pixel after 4 days (default value)
            show_fallout_longterm (bool): if True, show the dose rate of radiation 1 year after the detonation
            ms (float): the size of the markers
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

        if show_custom_locations:
            for i, row in enumerate(self.custom_locations.itertuples()):
                if row.status == "destroyed":
                    color = "red"
                else:
                    color = "green"
                folium.Marker(
                    [row.latitude, row.longitude],
                    popup=row.name,
                    icon=folium.Icon(color=color, icon="info-sign"),
                ).add_to(m)

        # Show fallout
        if show_fallout:
            bounds = [
                [np.min(self.lats), np.min(self.lons)],
                [np.max(self.lats), np.max(self.lons)],
            ]

            fallout_dense = self.fallout.toarray()

            # Create a custom colormap for fallout
            fallout_cmap = LinearColormap(
                colors=["green", "yellow", "orange", "red"],
                vmin=0,
                vmax=5000,
            )
            folium.raster_layers.ImageOverlay(
                image=fallout_dense,
                bounds=bounds,
                colormap=fallout_cmap,
                opacity=0.5,
                mercator_project=True,
            ).add_to(m)

            # Add a colorbar legend
            fallout_cmap.add_to(m)
            fallout_cmap.caption = "Radiation fallout, rads"

        if show_fallout_longterm:
            bounds = [
                [np.min(self.lats), np.min(self.lons)],
                [np.max(self.lats), np.max(self.lons)],
            ]

            fallout_longterm_dense = self.fallout_longterm.toarray()

            # Create a custom colormap for fallout
            fallout_longterm_cmap = LinearColormap(
                colors=["green", "yellow", "orange", "red"],
                vmin=0,
                vmax=0.002,
            )
            folium.raster_layers.ImageOverlay(
                image=fallout_longterm_dense,
                bounds=bounds,
                colormap=fallout_longterm_cmap,
                opacity=0.5,
                mercator_project=True,
            ).add_to(m)

            fallout_longterm_cmap.add_to(m)
            fallout_longterm_cmap.caption = "Radiation dose rate, rads/hr"

        # Display the map
        m.save("interactive_map.html")

        return m


def calculate_distance_km(lat1, lon1, lat2, lon2, bearing=False):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Args:
        lat1 (float): latitude of the first point
        lon1 (float): longitude of the first point
        lat2 (float): latitude of the second point
        lon2 (float): longitude of the second point
        bearing (bool): if True, also return the bearing in degrees

    Returns:
        dist (float): the distance in km
        bearing (float): the bearing in degrees (only if bearing is True)
    """
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)

    # Calculate the distance
    dist = distance.distance(point1, point2).kilometers

    if bearing:
        # Calculate the initial bearing
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        diff_lon_rad = math.radians(lon2 - lon1)

        x = math.sin(diff_lon_rad) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
            lat2_rad
        ) * math.cos(diff_lon_rad)

        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        bearing = (initial_bearing + 360) % 360

        return dist, bearing
    else:
        return dist


def get_fatality_rate(
    distance_from_groundzero,
    yield_kt,
    include_injuries=False,
    kill_radius_prescription="default",
    airburst=True,
):
    """
    Calculates the fatality rate given the distance from the ground zero and the yield of the warhead in kt.
    This accounts for blast, fire and prompt radiation. For fallout, see get_fatality_rate_fallout.

    Based on Toon et al. 2007, 2008 but using average yield of Hiroshima and Nagasaki (15kt and 21kt)
    for the scaling

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
    scaling_factor = Country.calculate_kill_radius_scaling_factor(
        kill_radius_prescription, yield_kt, airburst=airburst
    )
    sigma = sigma0 * scaling_factor
    return np.exp(-(distance_from_groundzero**2) / (2 * sigma**2))


def get_fatality_rate_fallout(radiation_dose_rads):
    """
    Calculates the fatality rate given the radiation dose in rads. Assumes
    an LD50 of 350 rads. Assumes that 50% of the population has a protection
    factor of 3 and 50% of the population has a protection factor of 10.

    Args:
        radiation_dose_rads (float): the radiation dose in rads

    Returns:
        fatality_rate (float): the fatality rate
    """
    # Step 4 - convert the radiation dose into a fatality rate
    alpha0 = lambda D: max(
        0, min(D / 300.0 - 2.0 / 3.0, 1)
    )  # Ensure alpha0 is between 0 and 1

    # Calculate the fatality rate considering protection factors
    alpha = 0.5 * alpha0(radiation_dose_rads / 3.0) + 0.5 * alpha0(
        radiation_dose_rads / 10.0
    )
    return alpha


def calculate_wind_adjustment_factor(windspeed_kmh):
    """
    Calculates the wind adjustment factor for nuclear fallout spread. Note that this
    is an verage over the vertical extent of the atmosphere relevant for fallout transport,
    not just the ground level wind speed

    Args:
        windspeed_kmh (float): Wind speed in km/h.

    Returns:
        float: Wind adjustment factor F.

    Warns:
        UserWarning: If the wind speed is outside the reliable range for this model.
    """
    v_mph = windspeed_kmh * 0.621  # Convert km/h to mph

    if windspeed_kmh < 13 or windspeed_kmh > 72:
        warnings.warn(
            "Wind speed is outside the reliable range for this model (13-72 km/h).",
            UserWarning,
        )

    if windspeed_kmh <= 24:
        F = 1 + (v_mph - 15) / 30
    else:
        F = 1 + (v_mph - 15) / 60

    return F


def calculate_reference_dose_contours(
    yield_kilotons, windspeed_kmh=24, fission_fraction=0.5
):
    """
    Calculate reference dose rate contours for a given yield, wind speed, and fission fraction.

    Args:
        yield_kilotons (float): Yield of the weapon in kilotons
        windspeed_kmh (float): Wind speed in km/h (default 24)
        fission_fraction (float): Fraction of energy from fission (default 0.5)

    Returns:
        dict: Dictionary of reference dose rates and their corresponding contour dimensions in km
    """
    # Cache for storing previously calculated contours
    if not hasattr(calculate_reference_dose_contours, "cache"):
        calculate_reference_dose_contours.cache = {}

    # Check if result is already in cache
    cache_key = (yield_kilotons, windspeed_kmh, fission_fraction)
    if cache_key in calculate_reference_dose_contours.cache:
        return calculate_reference_dose_contours.cache[cache_key]

    W = yield_kilotons
    F = calculate_wind_adjustment_factor(windspeed_kmh)

    reference_dose_rates = [3000, 1000, 300, 100, 30, 10, 3, 1]
    contours = {}

    miles_to_km = 1.60934

    for dose_rate in reference_dose_rates:
        if dose_rate == 3000:
            downwind = 0.95 * W**0.45 * F * miles_to_km
            max_width = 0.0076 * W**0.86 * miles_to_km
        elif dose_rate == 1000:
            downwind = 1.8 * W**0.45 * F * miles_to_km
            max_width = 0.036 * W**0.76 * miles_to_km
        elif dose_rate == 300:
            downwind = 4.5 * W**0.45 * F * miles_to_km
            max_width = 0.13 * W**0.66 * miles_to_km
        elif dose_rate == 100:
            downwind = 8.9 * W**0.45 * F * miles_to_km
            max_width = 0.38 * W**0.56 * miles_to_km
        elif dose_rate == 30:
            downwind = 16 * W**0.45 * F * miles_to_km
            max_width = 0.76 * W**0.56 * miles_to_km
        elif dose_rate == 10:
            downwind = 24 * W**0.45 * F * miles_to_km
            max_width = 1.4 * W**0.53 * miles_to_km
        elif dose_rate == 3:
            downwind = 30 * W**0.45 * F * miles_to_km
            max_width = 2.2 * W**0.50 * miles_to_km
        elif dose_rate == 1:
            downwind = 40 * W**0.45 * F * miles_to_km
            max_width = 3.3 * W**0.48 * miles_to_km

        # Adjust dose rate by fission fraction
        adjusted_dose_rate = dose_rate * fission_fraction

        contours[adjusted_dose_rate] = {
            "downwind_distance": downwind,
            "max_width": max_width,
        }

    # Store result in cache
    calculate_reference_dose_contours.cache[cache_key] = contours

    return contours


def calculate_total_dose(
    downwind_distance,
    perpendicular_distance,
    yield_kt,
    windspeed=24,
    tb=96,
    fission_fraction=0.5,
):
    """
    Calculate the total radiation dose at given points based on elliptical contours.

    Args:
        downwind_distance (ndarray): Distance downwind from ground zero in km.
        perpendicular_distance (ndarray): Perpendicular distance from the downwind axis in km.
        yield_kt (float): Yield of the weapon in kilotons.
        windspeed (float): Wind speed in km/h (default 24).
        tb (float): Total time of dose integration in hours after the detonation (default 96).
        fission_fraction (float): Fraction of energy from fission (default 0.5).

    Returns:
        ndarray: Total dose at the given points in rads.
    """
    # Make sure downwind_distance and perpendicular_distance are numpy arrays
    if not isinstance(downwind_distance, np.ndarray):
        downwind_distance = np.array([downwind_distance])
    if not isinstance(perpendicular_distance, np.ndarray):
        perpendicular_distance = np.array([perpendicular_distance])

    # Calculate time since detonation when fallout starts to be deposited
    ta = downwind_distance / windspeed  # hours

    # Calculate reference dose contours
    contours = calculate_reference_dose_contours(yield_kt, windspeed, fission_fraction)

    # Initialize dose rates array
    dose_rates = np.zeros_like(downwind_distance, dtype=float)

    # Sort the dose rates from highest to lowest to prioritize higher doses
    sorted_dose_rates = sorted(contours.keys(), reverse=True)

    for dose_rate in sorted_dose_rates:
        contour = contours[dose_rate]
        downwind_total = contour["downwind_distance"]
        max_width = contour["max_width"]

        # Calculate semi-major and semi-minor axes
        a = downwind_total / 2.0  # Semi-major axis (km)
        b = max_width / 2.0  # Semi-minor axis (km)

        # Center the ellipse at 0.5 * downwind_distance along the wind axis
        # Shift downwind_distance by half the major axis to center the ellipse correctly
        x_shifted = downwind_distance - (0.5 * downwind_total)

        # Compute the elliptical condition
        ellipse_condition = (x_shifted / a) ** 2 + (
            perpendicular_distance / b
        ) ** 2 <= 1

        # Assign dose rate where the condition is met and dose_rates is still zero
        dose_rates = np.where(
            ellipse_condition & (dose_rates == 0), dose_rate, dose_rates
        )

    # Calculate total dose
    with np.errstate(divide="ignore", invalid="ignore"):
        D = 5 * dose_rates * (ta**-0.2 - tb**-0.2)
        D[ta >= tb] = 0  # No dose if fallout arrives after tb
        D = np.maximum(D, 0)  # Ensure non-negative doses

    return D


def calculate_longterm_dose_rate(
    downwind_distance,
    perpendicular_distance,
    yield_kt,
    windspeed=24,
    tb=365 * 24,
    fission_fraction=0.5,
):
    """
    Calculate the radiation dose rate at a given point after a long time tb
    Used to evaluate the exclusion zone around a nuclear weapon detonation

    Args:
        downwind_distance (ndarray): Distance downwind from ground zero in km.
        perpendicular_distance (ndarray): Perpendicular distance from the downwind axis in km.
        yield_kt (float): Yield of the weapon in kilotons.
        windspeed (float): Wind speed in km/h (default 24).
        tb (float): Number of hours after detonation when the dose rate is calculated (default is 1 year)
        fission_fraction (float): Fraction of energy from fission (default 0.5).

    Returns:
        ndarray: Radiation dose rate in rads/hr
    """
    # Make sure downwind_distance and perpendicular_distance are numpy arrays
    if not isinstance(downwind_distance, np.ndarray):
        downwind_distance = np.array([downwind_distance])
    if not isinstance(perpendicular_distance, np.ndarray):
        perpendicular_distance = np.array([perpendicular_distance])

    # Calculate reference dose contours
    contours = calculate_reference_dose_contours(yield_kt, windspeed, fission_fraction)

    # Initialize dose rates array
    dose_rates = np.zeros_like(downwind_distance, dtype=float)

    # Sort the dose rates from highest to lowest to prioritize higher doses
    sorted_dose_rates = sorted(contours.keys(), reverse=True)

    for dose_rate in sorted_dose_rates:
        contour = contours[dose_rate]
        downwind_total = contour["downwind_distance"]
        max_width = contour["max_width"]

        # Calculate semi-major and semi-minor axes
        a = downwind_total / 2.0  # Semi-major axis (km)
        b = max_width / 2.0  # Semi-minor axis (km)

        # Center the ellipse at 0.5 * downwind_distance along the wind axis
        # Shift downwind_distance by half the major axis to center the ellipse correctly
        x_shifted = downwind_distance - (0.5 * downwind_total)

        # Compute the elliptical condition
        ellipse_condition = (x_shifted / a) ** 2 + (
            perpendicular_distance / b
        ) ** 2 <= 1

        # Assign dose rate where the condition is met and dose_rates is still zero
        dose_rates = np.where(
            ellipse_condition & (dose_rates == 0), dose_rate, dose_rates
        )

    # Calculate total dose
    with np.errstate(divide="ignore", invalid="ignore"):
        longterm_dose_rate = dose_rates * tb**-1.2

    return longterm_dose_rate


def calculate_cell_area(lat1, lon1, lat2, lon2):
    """
    Calculate the approximate area of a rectangular cell defined by two latitude and longitude pairs.

    Args:
        lat1 (float): Latitude of the first corner (degrees).
        lon1 (float): Longitude of the first corner (degrees).
        lat2 (float): Latitude of the opposite corner (degrees).
        lon2 (float): Longitude of the opposite corner (degrees).

    Returns:
        float: The area of the cell in square kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers

    # Convert degrees to radians
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad, lon2_rad = np.radians([lat2, lon2])

    # Calculate the absolute difference in longitude
    delta_lambda = abs(lon2_rad - lon1_rad)

    # Calculate the absolute difference in the sine of latitudes
    delta_phi = abs(np.sin(lat2_rad) - np.sin(lat1_rad))

    # Calculate the area using the spherical excess formula
    area = R**2 * delta_lambda * delta_phi

    return area


def calculate_distance_km_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance and initial bearing between a single point and arrays of points.

    Args:
        lat1 (float): Latitude of the first point (scalar).
        lon1 (float): Longitude of the first point (scalar).
        lat2 (ndarray): Latitudes of the second points (array).
        lon2 (ndarray): Longitudes of the second points (array).

    Returns:
        distances (ndarray): Distances in km between (lat1, lon1) and each point in (lat2, lon2).
        bearings (ndarray): Initial bearings in degrees from (lat1, lon1) to each point in (lat2, lon2).
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = 6371.0 * c  # Earth radius in km

    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
        lat2_rad
    ) * np.cos(dlon)
    initial_bearings_rad = np.arctan2(y, x)
    initial_bearings_deg = (np.degrees(initial_bearings_rad) + 360) % 360

    return distances, initial_bearings_deg


def process_chunk(chunk, country, region_data_shape):
    mask_region = gpd.sjoin(chunk, country, how="inner", predicate="intersects").index
    mask_region_bool = np.zeros(region_data_shape, dtype=bool)
    mask_region_bool.ravel()[mask_region] = True
    return mask_region_bool


def run_many_countries(
    scenario,
    degrade=False,
    degrade_factor=1,
    targeting_policy="max_fatality",
    include_injuries=False,
    kill_radius_prescription="default",
    burn_radius_prescription="default",
):
    """
    Run the model for multiple countries and write results to CSV

    Args:
        scenario (dict): a dictionary with the country names as keys and the arsenal as values
        degrade (bool): if True, degrade the LandScan data
        degrade_factor (int): the factor by which to degrade the LandScan data
        targeting_policy (str): the targeting policy to use, either "max_fatality_non_overlapping", "max_fatality", or "random_non_overlapping"
        include_injuries (bool): if True, include fatalities and injuries
        kill_radius_prescription (str): the method to calculate the kill radius, one of "default", "Toon", or "overpressure"
        burn_radius_prescription (str): the method to calculate the burn radius, one of "default", "Toon", or "overpressure"
    """

    # Open CSV file for writing results
    with open("../results/scenario_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "iso3",
                "population_loss",
                "population_loss_pct",
                "industry_destroyed_pct",
                "soot_emissions_Tg",
                "degrade_factor",
                "targeting_policy",
                "include_injuries",
                "kill_radius_prescription",
                "burn_radius_prescription",
            ]
        )  # Write header

        for country_name, arsenal in scenario.items():
            country = Country(
                country_name,
                landscan_year=2022,
                degrade=degrade,
                degrade_factor=degrade_factor,
                kill_radius_prescription=kill_radius_prescription,
                burn_radius_prescription=burn_radius_prescription,
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
            fatalities = country.get_total_fatalities()[0]
            population_loss_pct = 100 * fatalities / country.population_intact.sum()
            print(
                f"{country_name}, fatalities: {fatalities} ({population_loss_pct:.1f}%)"
            )

            industry_destroyed_pct = country.get_total_destroyed_industrial_area()
            print(
                f"{country_name}, industry destroyed: {100*industry_destroyed_pct:.2f}%"
            )

            print(f"{country_name}, soot emissions: {country.soot_Tg:.1f} Tg")

            # Write results for this country immediately to CSV
            writer.writerow(
                [
                    country.iso3,
                    fatalities,
                    population_loss_pct,
                    industry_destroyed_pct,
                    country.soot_Tg,
                    degrade_factor,
                    targeting_policy,
                    include_injuries,
                    kill_radius_prescription,
                    burn_radius_prescription,
                ]
            )


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


def apply_emp_damage(
    ground_zeros,
    radius_kms,
    industry,
):
    """
    Calculate the EMP damage to industrial areas within radii of multiple detonation locations

    Args:
        ground_zeros (list): list of tuples containing (latitude, longitude) of the detonation locations
        radius_kms (list): the radii in km within which to assess EMP damage for each detonation
        industry (GeoDataFrame): the industrial areas in the affected country

    Returns:
        disabled_industrial_areas_idx (list): the indices of the disabled industrial areas
    """
    # Initialize list to store disabled industrial area IDs
    disabled_industrial_areas_idx = []

    # Iterate through each industrial area
    for idx, row in industry.iterrows():
        # Check if the industrial area is within any EMP radius
        for (lat_gz, lon_gz), radius_km in zip(ground_zeros, radius_kms):
            distance = calculate_distance_km(
                lat_gz,
                lon_gz,
                row.geometry.centroid.y,
                row.geometry.centroid.x,
            )

            if distance <= radius_km:
                disabled_industrial_areas_idx.append(idx)
                break  # No need to check other EMPs if already disabled

    return disabled_industrial_areas_idx


def generate_circle_points(center_lat, center_lon, radius_km, num_points=100):
    points = []
    for i in range(num_points):
        angle = math.radians(i * (360 / num_points))
        point = distance.distance(kilometers=radius_km).destination(
            (center_lat, center_lon), bearing=math.degrees(angle)
        )
        lat, lon = point.latitude, point.longitude
        # if lon < 0:
        #     lon += 360
        points.append((lat, lon))
    return points


def plot_emp_damage(
    ground_zeros,
    radii_km,
    industry,
    disabled_industrial_areas_idx,
    show_industry=True,
):
    """
    Create an interactive map showing the EMP damage areas and affected industrial zones.

    Args:
        ground_zeros (list): list of tuples containing (latitude, longitude) of the detonation locations
        radii_km (list): list of radii in km within which to assess EMP damage for each detonation
        industry (GeoDataFrame): the industrial areas in the affected country
        disabled_industrial_areas_idx (list): the indices of the disabled industrial areas
        show_industry (bool): if True, show the industrial areas
    Returns:
        folium.Map: An interactive map object
    """
    # Create a folium map centered around the first ground zero
    m = folium.Map(location=ground_zeros[0], zoom_start=8)

    # Add markers and circles for each ground zero
    for (lat_groundzero, lon_groundzero), radius_km in zip(ground_zeros, radii_km):
        # Add a marker for the ground zero
        folium.Marker(
            [lat_groundzero, lon_groundzero],
            popup="Ground Zero",
            icon=folium.Icon(color="red", icon="x"),
        ).add_to(m)

        circle_points = generate_circle_points(
            lat_groundzero, lon_groundzero, radius_km
        )
        folium.Polygon(
            locations=circle_points,
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.1,
            popup=f"EMP Radius: {radius_km} km",
        ).add_to(m)

    # Plot industrial areas
    if show_industry:
        for idx, row in industry.iterrows():
            color = "black" if idx in disabled_industrial_areas_idx else "purple"
            folium.Polygon(
                locations=[(y, x) for x, y in row.geometry.exterior.coords],
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.3,
                popup=(
                    "Disabled Industrial Area"
                    if idx in disabled_industrial_areas_idx
                    else "Industrial Area"
                ),
            ).add_to(m)

    return m


def get_1956_US_nuclear_war_plan(country):
    """
    Parse the US1956_target_definitions.js file to build a dict of nuclear targets
    for a specific country.

    Args:
        country (str): The name of the country to filter targets for.

    Returns:
        dict: A dictionary of targets with city names as keys and tuples of
              (latitude, longitude) as values.
    """
    # Read the contents of the JS file
    with open("../data/target-lists/US1956_target_definitions.js", "r") as file:
        js_content = file.read()

    # Extract the JSON-like string from the JS file
    json_str = re.search(r"var targets=(\{.*?\});", js_content, re.DOTALL)
    if not json_str:
        raise ValueError("Could not find targets data in the JS file")

    # Parse the JSON-like string
    targets_data = json.loads(json_str.group(1))

    # Filter and format the targets for the specified country
    country_targets = {}
    for feature in targets_data["features"]:
        if feature["properties"]["Country"] == country:
            city = feature["properties"]["City"]
            lon, lat = feature["geometry"]["coordinates"]
            country_targets[city] = (lat, lon)

    if len(country_targets) == 0:
        raise ValueError(f"No targets found for {country}")

    return country_targets


def get_OPEN_RISOP_nuclear_war_plan(
    ignore_military=False,
    ignore_dual_use=False,
    ignore_war_supporting=False,
    ignore_critical=False,
    ignore_other_civilian=False,
    icbm_only=False,
):
    """
    Get the nuclear war plan from the OPEN RISOP database

    Args:
        ignore_military (bool): If True, ignore military targets
        ignore_dual_use (bool): If True, ignore dual-use targets
        ignore_war_supporting (bool): If True, ignore war-supporting industry targets
        ignore_critical (bool): If True, ignore critical infrastructure targets
        ignore_other_civilian (bool): If True, ignore other civilian targets
        icbm_only (bool): If True, only include ICBM targets
    Returns:
        dict: A dictionary of targets with names as keys and tuples of
                (latitude, longitude, hob, yield, category) as values.
    """
    # Read the Excel file
    df = pd.read_excel(
        "../data/target-lists/OPEN-RISOP 1.00 MIXED COUNTERFORCE+COUNTERVALUE ATTACK.xlsx"
    )

    # Read the CSV file
    df2 = pd.read_csv(
        "../data/target-lists/OPEN-RISOP 1.00 TARGET DATABASE.csv"
    )

    # Merge the dataframes
    # Function to find the closest match based on latitude and longitude
    def find_closest_match(row, df2):
        diff = abs(df2["LATITUDE"] - row["Latitude"]) + abs(
            df2["LONGITUDE"] - row["Longitude"]
        )
        return df2.iloc[diff.idxmin()]

    # Apply the function to each row in df
    merged_df = df.apply(
        lambda row: pd.concat([row, find_closest_match(row, df2)]), axis=1
    )

    # Reset the index of the merged dataframe
    merged_df = merged_df.reset_index(drop=True)

    # Ensure all columns from df are present
    for col in df.columns:
        if col not in merged_df.columns:
            merged_df[col] = df[col]

    merged_df = merged_df.drop_duplicates(subset=["Latitude", "Longitude"])

    # Define category lists
    military = [
        "ICBM SILOS",
        "AIRFIELDS, MILITARY",
        "MILITARY BASES",
        "ICBM LCCS",
        "SPECIAL COMMS",
        "INTELLIGENCE FACILITIES",
        "NUCLEAR WEAPONS STORAGE",
        "SENIOR MILITARY LEADERSHIP",
        "RADAR SYSTEMS",
        "ANG ISR-SPACE SQ",
        "AIR DEFENSE CONTROL SITES",
        "SUBMARINE BASES",
        "MILITARY FORCES, AIRCRAFT",
    ]
    dual_use = [
        "AIRFIELDS, JOINT USE",
        "SPACE SYSTEMS",
        "FFRDC FACILITIES",
        "SPACE LAUNCH COMPLEXES",
        "STATE AREA HQ",
    ]
    war_supporting_industries = [
        "DEFENSE INDUSTRIAL BASE",
        "OIL REFINERIES",
        "STEEL PRODUCTION",
        "CHEMICAL MANUFACTURING",
        "SULFURIC ACID PRODUCTION",
        "ALUMINUM SMELTERS",
        "SPENT FUEL INSTALLATIONS",
        "NUCLEAR FUEL & COMPONENTS",
        "AMMONIA PRODUCTION",
    ]
    critical_infrastructure = [
        "AIRFIELD, CIVILIAN",
        "DAMS",
        "RAILROAD YARDS",
        "4ESS SWITCH",
        "CABLE LANDING STATION",
        "CRITICAL TELECOM",
        "LOCKS",
        "ARTCC",
        "PORT FACILITIES",
        "STRATEGIC PETROLEUM RESERVES",
        "THERMAL POWER PLANTS",
        "POL STORAGE",
        "NG COMPRESSOR STATIONS",
        "INTERMODAL FACILITIES",
        "RAILROAD SHOPS",
    ]
    other_civilian = [
        "CITY HALL",
        "STATE EOC",
        "STATE CAPITOL",
        "COLLEGES AND UNIVERSITIES",
        "FEMA SITE",
        "ADVANCED MEDICAL CENTERS",
        "FEDERAL RESERVE BRANCH",
        "74101 GOVERNMENT BRANCH DEPARTMENT HQS",
        "FEMA SPECIAL COMMS",
        "FEDERAL RESERVE RO",
        "FEDERAL RESERVE BANK",
        "GOVERNMENT CONTROL CENTERS",
        "POTENTIAL COG/COOP SITES",
        "FEDERAL RESERVE BRANCH",
        "FEMA SPECIAL COMMS",
    ]

    # Create a dictionary with the required structure
    targets = {}
    total_yield = 0
    for _, row in merged_df.iterrows():
        name = row["Name"]
        lat = row["Latitude"]
        lon = row["Longitude"]
        hob = row["HOB (m)"]
        ykt = row["Yield (kt)"]
        subclass = row["SUBCLASS"]

        # Assign category based on SUBCLASS
        if subclass in military:
            category = "Military"
        elif subclass in dual_use:
            category = "Dual-use"
        elif subclass in war_supporting_industries:
            category = "War-supporting"
        elif subclass in critical_infrastructure:
            category = "Critical"
        elif subclass in other_civilian:
            category = "Other Civilian"
        else:
            category = "Unknown"

        # Check if the category should be ignored
        if (
            (ignore_military and category == "Military")
            or (ignore_dual_use and category == "Dual-use")
            or (ignore_war_supporting and category == "War-supporting")
            or (ignore_critical and category == "Critical")
            or (ignore_other_civilian and category == "Other Civilian")
        ):
            continue

        if icbm_only and subclass!="ICBM SILOS":
            continue

        targets[name] = (lat, lon, hob, ykt)
        total_yield += ykt

    print(f"Total yield: {total_yield} kt")
    return targets


def build_scaling_curve(
    country_name,
    yield_kt,
    numbers_of_weapons,
    non_overlapping=True,
    degrade_factor=1,
):
    """
    Build a scaling curve for the given country by simulating nuclear attacks with varying numbers of weapons.

    Args:
        country_name (str): The name of the country to be attacked.
        yield_kt (float): The yield of each warhead in kilotons.
        numbers_of_weapons (list): A list of integers representing the number of weapons to be used in each simulation.
        non_overlapping (bool): If True, the weapons will be detonated in non-overlapping fashion.
        degrade_factor (int): The factor by which to degrade the LandScan data.

    Returns:
        None. The results are saved to a CSV file in the ../results/ directory.
    """
    output_file = (
        f"../results/{country_name.lower().replace(' ', '_')}_scaling_results.csv"
    )
    degrade = degrade_factor > 1

    # Create or load existing CSV file
    if os.path.exists(output_file):
        results_df = pd.read_csv(output_file)
    else:
        results_df = pd.DataFrame(
            columns=[
                "country",
                "number_of_weapons",
                "yield_kt",
                "fatalities",
                "industry_destroyed_pct",
                "soot_emissions",
                "non_overlapping",
            ]
        )

    for number_of_weapons in numbers_of_weapons:
        print(f"Number of weapons: {number_of_weapons}")
        arsenal = number_of_weapons * [yield_kt]
        country = Country(
            country_name,
            landscan_year=2022,
            degrade=degrade,
            degrade_factor=degrade_factor,
        )
        country.attack_max_fatality(
            arsenal, include_injuries=False, non_overlapping=non_overlapping
        )
        country.print_diagnostic_info()

        fatalities = country.get_total_fatalities()[0]
        industry_destroyed_pct = country.get_total_destroyed_industrial_area()
        soot_emissions = country.soot_Tg
        destroyed_custom_locations = country.get_number_destroyed_custom_locations()

        new_row = pd.DataFrame(
            {
                "country": [country_name],
                "number_of_weapons": [number_of_weapons],
                "yield_kt": [yield_kt],
                "fatalities": [fatalities],
                "industry_destroyed_pct": [industry_destroyed_pct],
                "soot_emissions": [soot_emissions],
                "non_overlapping": [non_overlapping],
            }
        )

        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Save the updated results after each iteration
        results_df.to_csv(output_file, index=False)


def plot_scaling_results(yield_kt=100):
    """
    Plot the scaling results of all ../results/*_scaling_results.csv files.
    One curve per country (that is, per file).

    Args:
        yield_kt (float): The yield of each warhead in kilotons. Default is 100.

    Returns:
        None. The plots are displayed.
    """
    results_dir = "../results"
    files = [f for f in os.listdir(results_dir) if f.endswith("_scaling_results.csv")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for i, file in enumerate(files):
        file_path = os.path.join(results_dir, file)
        df = pd.read_csv(file_path)
        df_filtered = df[df["yield_kt"] == yield_kt]

        country_name = df_filtered["country"].iloc[0]
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]

        for non_overlapping in [True, False]:
            df_filtered_case = df_filtered[
                df_filtered["non_overlapping"] == non_overlapping
            ]
            if not df_filtered_case.empty:
                label = f"{country_name} - {'Non-overlapping' if non_overlapping else 'Overlapping'}"
                linestyle = "-" if non_overlapping else "--"
                ax1.plot(
                    df_filtered_case["number_of_weapons"],
                    df_filtered_case["fatalities"],
                    label=label,
                    linestyle=linestyle,
                    color=color,
                )
                ax2.plot(
                    df_filtered_case["number_of_weapons"],
                    df_filtered_case["industry_destroyed_pct"] * 100,
                    label=label,
                    linestyle=linestyle,
                    color=color,
                )

    ax1.set_ylabel("Fatalities")
    ax1.legend()

    ax2.set_xlabel(f"Number of {yield_kt}-kt weapons")
    ax2.set_ylabel("% industry destroyed")

    plt.tight_layout()
    plt.show()


def plot_static_target_map(target_list, yields, region=None):
    """
    Plot a static map of the target list.

    Args:
        target_list (list): A list of tuples, where each tuple contains the latitude and longitude of a target.
        yields (list): A list of the yields of the warheads.
        region (str): Optional. The region to focus on. Options: 'Europe', 'South Asia', 'China', 'CONUS', 'Russia'.

    Returns:
        None. The plot is displayed.
    """
    # Define region boundaries
    regions = {
        "Europe": {"llcrnrlat": 35, "urcrnrlat": 70, "llcrnrlon": -10, "urcrnrlon": 40},
        "South Asia": {
            "llcrnrlat": 5,
            "urcrnrlat": 38,
            "llcrnrlon": 60,
            "urcrnrlon": 93,
        },
        "China": {"llcrnrlat": 15, "urcrnrlat": 55, "llcrnrlon": 70, "urcrnrlon": 140},
        "CONUS": {
            "llcrnrlat": 25,
            "urcrnrlat": 50,
            "llcrnrlon": -125,
            "urcrnrlon": -65,
        },
        "Russia": {"llcrnrlat": 40, "urcrnrlat": 75, "llcrnrlon": 20, "urcrnrlon": 180},
    }

    # Create a new figure
    plt.figure(figsize=(15, 10))

    # Create a Basemap instance
    if region and region in regions:
        m = Basemap(projection="mill", **regions[region], resolution="i")
    else:
        m = Basemap(
            projection="mill",
            llcrnrlat=-60,
            urcrnrlat=90,
            llcrnrlon=-180,
            urcrnrlon=180,
            resolution="i",
        )

    # Draw map features
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.fillcontinents(color="lightgrey", lake_color="white")
    m.drawstates()

    # Plot targets
    lats, lons = zip(*target_list)
    x, y = m(lons, lats)

    # Scale marker sizes based on yield
    min_size, max_size = 20, 200  # Adjust these values for desired marker size range
    sizes = [min_size + (max_size - min_size) * np.sqrt(y / 500) for y in yields]

    # Plot targets with size based on yield
    scatter = m.scatter(
        x,
        y,
        s=sizes,
        c="red",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    m.drawmapboundary(fill_color="white", linewidth=0.4, color="black")
    plt.savefig("../images/newmap.png", bbox_inches="tight")
    plt.show()
