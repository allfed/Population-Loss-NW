import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from glob import glob


def get_latest_food_supply(country):
    """
    Get the total number of calories produced for a given country from all agriculture CSV files.

    Args:
        country (str): Name of the country

    Returns:
        dict: A dictionary with crop names as keys and their total calories as values
    """
    food_supply = {}
    csv_files = glob("../data/agriculture/*.csv")

    for file in csv_files:
        df = pd.read_csv(file)
        crop = os.path.splitext(os.path.basename(file))[0]
        country_data = df[df.Country == country].sort_values("Year", ascending=False)

        if not country_data.empty:
            for _, row in country_data.iterrows():
                year = row["Year"]
                production = row["Production (t)"]
                calories_supply = row["Food supply (kcal per capita per day)"]
                grams_supply = row["Food supply (g per capita per day)"]
                try:
                    caloric_density = calories_supply / grams_supply
                except ZeroDivisionError:
                    caloric_density = np.nan
                production_calories = production * caloric_density * 1e6
                if pd.notna(production_calories):
                    food_supply[crop] = production_calories
                    break
                elif (
                    year < 2010
                ):  # probably that this country is not producing that crop
                    break

    return food_supply


def calculate_crop_fractions(food_supply):
    """
    Calculate the fraction of kcals provided by each crop.

    Args:
        food_supply (dict): Dictionary with crop names as keys and their total calories as values

    Returns:
        dict: A dictionary with crop names as keys and their fractions as values
    """
    total_supply = sum(food_supply.values())
    return {crop: supply / total_supply for crop, supply in food_supply.items()}


def calculate_yield_loss(country, crop, input, pct, debug=False):
    """
    Calculate the yield loss for a given country and crop.

    Args:
        country (str): Name of the country
        crop (str): Name of the crop
        input (str): Name of the input (fertilizer, phosphorus, potassium, nitrogen)
        pct (float): Percentage loss of input
        debug (bool): If True, produce a debug map

    Returns:
        float: Percentage yield loss
    """
    if pct > 75:
        raise ValueError("Percentage loss cannot be greater than 75%")

    if pct == 0:
        return 0

    if country == "United States":
        country = "United States of America"
    elif country == "Cape Verde":
        country = "Cabo Verde"
    elif country == "Congo":
        country = "Republic of the Congo"
    elif country == "Democratic Republic of Congo":
        country = "Democratic Republic of the Congo"
    elif country == "Eswatini":
        country = "eSwatini"
    elif country == "Cote d'Ivoire":
        country = "Ivory Coast"
    elif country == "Serbia":
        country = "Republic of Serbia"
    elif country == "Tanzania":
        country = "United Republic of Tanzania"

    world = gpd.read_file("../data/natural-earth/ne_10m_admin_0_countries.shp")
    country_shape = world[world.ADMIN == country].geometry.values[0]

    # Determine the two closest percentages for interpolation
    valid_pcts = [0, 25, 50, 75]
    lower_pct = max([p for p in valid_pcts if p <= pct])
    upper_pct = min([p for p in valid_pcts if p >= pct])

    def read_shock_file(file_pct):
        if file_pct == 0:
            return 0
        shock_file = f"../data/ahvo/{crop}_{input}_shock{file_pct}.tif"
        baseline_file = f"../data/ahvo/{crop}_modelled_baseline_yield.tif"
        with rasterio.open(shock_file) as src, rasterio.open(
            baseline_file
        ) as baseline_src:
            shock_yield, transform = mask(src, [country_shape], crop=True)
            baseline_yield, _ = mask(baseline_src, [country_shape], crop=True)

            # Get the latitudes of each pixel
            rows, _ = np.indices(shock_yield.shape[1:])
            lats = np.degrees(np.arctan2(rows - transform[5], transform[4]))
            lats = np.expand_dims(lats, axis=0)

            # Mask out areas where baseline yield is zero or NaN
            valid_mask = (baseline_yield != 0) & (~np.isnan(baseline_yield))
            shock_yield = shock_yield[valid_mask]
            baseline_yield = baseline_yield[valid_mask]
            lats = lats[valid_mask]

            # If there is no baseline yield, return 0
            if np.sum(baseline_yield) == 0:
                return 0

            # Calculate weights based on the cosine of the latitude, to account for the area of each pixel
            weights = np.abs(np.cos(np.radians(lats)))

            # Calculate weighted average, including pixel area and latitude in the weight
            weighted_shock = np.sum(shock_yield * baseline_yield * weights) / np.sum(
                baseline_yield * weights
            )
            return weighted_shock

    lower_shock = read_shock_file(lower_pct)
    if lower_pct != upper_pct:
        upper_shock = read_shock_file(upper_pct)
        # Linear interpolation
        shock_avg = lower_shock + (upper_shock - lower_shock) * (pct - lower_pct) / (
            upper_pct - lower_pct
        )
    else:
        shock_avg = lower_shock

    if shock_avg > 0 or np.isnan(shock_avg):
        shock_avg = 0

    if debug:
        fig, ax = plt.subplots(figsize=(15, 10))
        world.plot(ax=ax, color="lightgrey", edgecolor="black")

        bounds = country_shape.bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

        # Use the lower percentage for visualization
        shock_file = f"../data/ahvo/{crop}_{input}_shock{lower_pct}.tif"
        with rasterio.open(shock_file) as src:
            shock_yield, _ = mask(src, [country_shape], crop=True)

        # Convert the 2D array to 1D arrays of coordinates and values
        lon, lat = np.meshgrid(
            np.linspace(bounds[0], bounds[2], shock_yield.shape[2]),
            np.linspace(bounds[3], bounds[1], shock_yield.shape[1]),
        )
        lon, lat = lon.flatten(), lat.flatten()
        values = shock_yield[0].flatten()

        # Remove NaN values
        mask_values = ~np.isnan(values)
        lon, lat, values = lon[mask_values], lat[mask_values], values[mask_values]

        # Create scatter plot
        scatter = ax.scatter(
            lon, lat, c=values, cmap="RdYlBu_r", vmin=-100, vmax=0, s=1, alpha=0.5
        )

        plt.colorbar(scatter, label="Yield loss (%)")
        ax.set_title(
            f"Yield Loss for {crop.capitalize()} in {country} (Interpolated: {pct}%)"
        )
        plt.show()

    return shock_avg


def calculate_agriculture_loss(country, N_loss, P_loss, K_loss, pesticide_loss):
    """
    Calculate the calories-weighted average yield loss for a country across all crops.

    Args:
        country (str): Name of the country
        N_loss (float): Percentage loss of nitrogen
        P_loss (float): Percentage loss of phosphorus
        K_loss (float): Percentage loss of potassium
        pesticide_loss (float): Percentage loss of pesticide

    Returns:
        float: Weighted average yield loss percentage
    """

    food_supply = get_latest_food_supply(country)
    crop_fractions = calculate_crop_fractions(food_supply)

    total_weighted_loss = 0
    for crop, fraction in crop_fractions.items():
        N_induced_loss = calculate_yield_loss(country, crop, "nitrogen", N_loss)
        P_induced_loss = calculate_yield_loss(country, crop, "phosphorus", P_loss)
        K_induced_loss = calculate_yield_loss(country, crop, "potassium", K_loss)
        pesticide_induced_loss = calculate_yield_loss(
            country, crop, "pesticide", pesticide_loss
        )

        # Combine losses multiplicatively
        yield_loss = 1 - (1 + N_induced_loss / 100) * (1 + P_induced_loss / 100) * (
            1 + K_induced_loss / 100
        ) * (1 + pesticide_induced_loss / 100)
        yield_loss *= -100  # Convert back to percentage

        total_weighted_loss += yield_loss * fraction

    print(f"{country} loses {-total_weighted_loss:.1f}% of its agriculture production")
    return total_weighted_loss


def process_country(
    country, N_loss, P_loss, K_loss, pesticide_loss, existing_countries, results
):
    """
    Process a single country and calculate its agriculture loss. Helper function for multiprocessing
    in calculate_agriculture_loss_for_all_countries.
    """
    if (
        country not in existing_countries
        and "FAO" not in country
        and "World" not in country
        and "countries" not in country
        and "former" not in country
        and "USSR" not in country
        and "(" not in country
        and country != "Africa"
        and country != "Africa (FAO)"
        and "Asia" not in country
        and "Europe" not in country
        and "Oceania" not in country
        and "South America" not in country
        and "North America" not in country
        and "Czechoslovakia" not in country
        and "Yugoslavia" not in country
        and "Serbia and Montenegro" not in country
        and "Polynesia" not in country
        and "Bahamas" not in country
        and "Hong Kong" not in country
        and "Macao" not in country
        and "Melanesia" not in country
        and "Netherlands Antilles" not in country
        and "Sao Tome and Principe" not in country
    ):
        ans = calculate_agriculture_loss(
            country, N_loss, P_loss, K_loss, pesticide_loss
        )
        results.append({"country": country, "yield_loss_pct": ans})

        # Append the current result to CSV as a backup
        results_df = pd.DataFrame([{"country": country, "yield_loss_pct": ans}])
        results_df.to_csv(
            "../results/yield_loss_results.csv",
            mode="a",
            header=not os.path.exists("../results/yield_loss_results.csv"),
            index=False,
        )


def calculate_agriculture_loss_for_all_countries(
    N_loss, P_loss, K_loss, pesticide_loss
):
    """
    Calculate the agriculture loss for all countries and save the results to a CSV file.

    Args:
        N_loss (float): Percentage loss of nitrogen
        P_loss (float): Percentage loss of phosphorus
        K_loss (float): Percentage loss of potassium
        pesticide_loss (float): Percentage loss of pesticide

    Returns:
        None
    """
    # Build a list of all country names
    df = pd.read_csv("../data/agriculture/wheat.csv")
    results = []

    # Read already calculated results
    if os.path.exists("../results/yield_loss_results.csv"):
        existing_results_df = pd.read_csv("../results/yield_loss_results.csv")
        existing_countries = set(existing_results_df["country"])
    else:
        existing_countries = set()

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.starmap(
            process_country,
            [
                (
                    country,
                    N_loss,
                    P_loss,
                    K_loss,
                    pesticide_loss,
                    existing_countries,
                    results,
                )
                for country in df.Country.unique()
            ],
        )

    print("Results saved to ../results/yield_loss_results.csv")


def plot_yield_loss_world_map(
    input_file="../results/yield_loss_results.csv", scenario_name="unspecified"
):
    """
    Makes a map of the yield loss for each country based on the input file,
    generated by calculate_agriculture_loss_for_all_countries.
    """
    # Read the yield loss results
    yield_loss_df = pd.read_csv(input_file)
    yield_loss_df["country"] = yield_loss_df["country"].replace(
        {
            "United States": "United States of America",
            "Cape Verde": "Cabo Verde",
            "Congo": "Republic of the Congo",
            "Democratic Republic of Congo": "Democratic Republic of the Congo",
            "Eswatini": "eSwatini",
            "Cote d'Ivoire": "Ivory Coast",
            "Serbia": "Republic of Serbia",
            "Tanzania": "United Republic of Tanzania",
        }
    )

    # Multiply yield loss by -1 to plot positive values
    yield_loss_df["yield_loss_pct"] = yield_loss_df["yield_loss_pct"] * -1

    # Read the world map shapefile
    world = gpd.read_file("../data/natural-earth/ne_10m_admin_0_countries.shp")

    # Merge the yield loss data with the world map
    world = world.merge(yield_loss_df, left_on="ADMIN", right_on="country", how="left")
    world = world.to_crs("+proj=wintri")

    # Define the color map and normalization
    default_color = "lightgrey"
    cmap = plt.cm.get_cmap("plasma")
    colors = cmap(np.linspace(0, 1, 11))
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(0, 50, 5)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the world map with yield loss data
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sm = world.plot(
        column="yield_loss_pct",
        cmap=cmap,
        norm=norm,
        legend=True,
        legend_kwds={
            "label": "Yield loss (%)",
            "orientation": "vertical",
            "shrink": 0.3,
        },
        ax=ax,
        missing_kwds={"color": default_color},
    )
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xlim(-15000000, 19000000)
    ax.set_ylim(-6500000, 9500000)

    if scenario_name != "unspecified":
        plt.title(f"Yield loss due to input shock ({scenario_name})")
    else:
        plt.title("Yield loss due to input shock")

    plt.tight_layout()
    plt.savefig("../results/yield_loss_world_map.pdf", bbox_inches="tight")


def add_iso3_to_yield_loss_results(input_file="../results/yield_loss_results.csv"):
    """
    Add the ISO3 code to the yield loss results.
    """
    # Read the yield loss results
    yield_loss_df = pd.read_csv(input_file)

    # Read the world map shapefile to get ISO3 codes
    world = gpd.read_file("../data/natural-earth/ne_10m_admin_0_countries.shp")

    # Create a dictionary mapping country names to ISO3 codes
    country_to_iso3 = dict(zip(world["ADMIN"], world["ADM0_A3"]))

    # Create a temporary column with standardized country names for mapping
    yield_loss_df["temp_country"] = yield_loss_df["country"].replace({
        "United States": "United States of America",
        "Cape Verde": "Cabo Verde",
        "Congo": "Republic of the Congo",
        "Democratic Republic of Congo": "Democratic Republic of the Congo",
        "Eswatini": "eSwatini",
        "Cote d'Ivoire": "Ivory Coast",
        "Serbia": "Republic of Serbia",
        "Tanzania": "United Republic of Tanzania"
    })

    # Add ISO3 column to yield_loss_df using the temporary standardized names
    yield_loss_df["iso3"] = yield_loss_df["temp_country"].map(country_to_iso3)

    # Remove the temporary column
    yield_loss_df = yield_loss_df.drop(columns=["temp_country"])

    # Reorder columns to make iso3 the first column
    columns = yield_loss_df.columns.tolist()
    columns = ["iso3"] + [col for col in columns if col != "iso3"]
    yield_loss_df = yield_loss_df[columns]

    # Save the updated dataframe
    yield_loss_df.to_csv(input_file, index=False)

    print(f"Added ISO3 codes to {input_file}")
