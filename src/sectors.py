import pandas as pd


def calculate_sector_losses(total_industry_loss):
    """
    Calculate sector losses for a given total industry loss per country

    Args:
        total_industry_loss (dict): Total industry loss per country (%)

    Returns:
        dict: Sector losses for the whole world (%)
    """
    sector_losses = {}
    sector_losses = calculate_fertilizer_loss(total_industry_loss, sector_losses)
    sector_losses = calculate_pesticide_loss(total_industry_loss, sector_losses)

    # Print a nicely formatted table of sector losses
    print("\nLosses per sector for the whole world:")
    print("-" * 40)
    print(f"{'Sector':<20} {'Loss (%)':<10}")
    print("-" * 40)
    for sector, loss in sector_losses.items():
        sector_name = sector.replace("_loss_percent", "").replace("_", " ").capitalize()
        print(f"{sector_name:<20} {loss:<10.2f}")
    print("-" * 40)

    return sector_losses


def calculate_fertilizer_loss(total_industry_loss, sector_losses):
    """
    Calculate fertilizer loss for a given total industry loss per country

    Args:
        total_industry_loss (dict): Total industry loss per country (%)
        sector_losses (dict): Sector losses for the whole world (%)

    Returns:
        dict: Sector losses for the whole world (%)
    """

    fertilizers = load_fertilizer_production_by_nutrient_type_npk()

    # Initialize total losses
    total_losses = {"N_loss": 0, "P_loss": 0, "K_loss": 0}

    # Calculate losses for each country and sum them up
    for country, loss_percentage in total_industry_loss.items():
        if country in fertilizers["Code"].values:
            country_data = fertilizers[fertilizers["Code"] == country].iloc[0]
            total_losses["N_loss"] += country_data["N_percent"] * loss_percentage / 100
            total_losses["P_loss"] += country_data["P_percent"] * loss_percentage / 100
            total_losses["K_loss"] += country_data["K_percent"] * loss_percentage / 100
        else:
            print(f"Country {country} not found in fertilizer production data")

    sector_losses["N_fertilizer_loss_percent"] = total_losses["N_loss"]
    sector_losses["P_fertilizer_loss_percent"] = total_losses["P_loss"]
    sector_losses["K_fertilizer_loss_percent"] = total_losses["K_loss"]

    return sector_losses


def calculate_pesticide_loss(total_industry_loss, sector_losses):
    """
    Calculate pesticide loss for a given total industry loss per country

    Args:
        total_industry_loss (dict): Total industry loss per country (%)
        sector_losses (dict): Sector losses for the whole world (%)

    Returns:
        dict: Updated sector losses for the whole world (%)
    """
    pesticide_production = calculate_pesticide_production()

    total_loss = 0

    for country, loss_percentage in total_industry_loss.items():
        country_data = pesticide_production[pesticide_production["iso3"] == country]
        if not country_data.empty:
            country_production = country_data["percentage_of_world_production"].iloc[0]
            country_loss = country_production * loss_percentage / 100
            total_loss += country_loss
        else:
            print(f"Country {country} not found in pesticide production data")

    sector_losses["pesticide_loss_percent"] = total_loss

    return sector_losses


def load_fertilizer_production_by_nutrient_type_npk():
    """
    Load fertilizer production by nutrient type NPK from OWID
    """
    df = pd.read_csv(
        "../data/industry-sectors/fertilizer-production-by-nutrient-type-npk.csv"
    )
    # Drop continents and keep only countries and world total
    df = df[df["Entity"].isin(["World"]) | (df["Code"].notna())]

    # Select year 2021
    df = df[df["Year"] == 2021]

    # Rename columns
    df = df.rename(
        columns={
            "Nutrient potash K2O (total) | 00003104 || Production | 005510 || Tonnes": "K_tonnes",
            "Nutrient phosphate P2O5 (total) | 00003103 || Production | 005510 || Tonnes": "P_tonnes",
            "Nutrient nitrogen N (total) | 00003102 || Production | 005510 || Tonnes": "N_tonnes",
        }
    )

    # Reset index after filtering
    df = df.reset_index(drop=True)

    # Calculate world totals
    world_totals = df[df["Entity"] == "World"][
        ["K_tonnes", "P_tonnes", "N_tonnes"]
    ].iloc[0]

    # Calculate sum of individual countries
    country_sums = df[df["Entity"] != "World"][
        ["K_tonnes", "P_tonnes", "N_tonnes"]
    ].sum()

    # Check that world totals match sum of individual countries
    for nutrient in ["K_tonnes", "P_tonnes", "N_tonnes"]:
        assert (
            abs(world_totals[f"{nutrient}"] - country_sums[f"{nutrient}"])
            / world_totals[f"{nutrient}"]
            < 0.001
        ), f"World totals do not match sum of individual countries for {nutrient} world total: {world_totals[f'{nutrient}']} country sum: {country_sums[f'{nutrient}']}"

    # Drop year and entity columns
    df = df.drop(columns=["Year", "Entity"])

    # Calculate percentages of world total for N, P, K
    world_total = df[df["Code"] == "OWID_WRL"].iloc[0]
    for nutrient in ["N_tonnes", "P_tonnes", "K_tonnes"]:
        df[f"{nutrient[0]}_percent"] = df[nutrient] / world_total[nutrient] * 100

    # Drop the original tonnage columns
    df = df.drop(columns=["N_tonnes", "P_tonnes", "K_tonnes"])

    # Remove the world row
    df = df[df["Code"] != "OWID_WRL"]

    # Check that the sum of percentages is 100% of all nutrients
    assert (
        abs(df["N_percent"].sum() - 100) < 0.001
    ), "Sum of percentages is not 100% of all nutrients"
    assert (
        abs(df["P_percent"].sum() - 100) < 0.001
    ), "Sum of percentages is not 100% of all nutrients"
    assert (
        abs(df["K_percent"].sum() - 100) < 0.001
    ), "Sum of percentages is not 100% of all nutrients"

    return df


def get_bilateral_pesticide_trade():
    """
    Loads and processes bilateral pesticide trade data from BACI dataset.

    Returns:
    pd.DataFrame: A DataFrame containing bilateral pesticide trade data with columns:
        - year: Year of trade
        - exporter: Exporting country (ISO3 code)
        - importer: Importing country (ISO3 code)
        - pesticide_total_value: Total value of pesticide trade in 1000 USD
        - pesticide_total_tonnes: Estimated tonnes of active ingredient traded
    """
    df = pd.read_csv("../data/industry-sectors/BACI/BACI_HS17_Y2022_V202401b.csv")

    # Rename columns based on the information from the BACI readme file
    df = df.drop(columns=["q"])
    df = df.rename(
        columns={
            "t": "year",
            "i": "exporter",
            "j": "importer",
            "k": "product",
            "v": "value",
        }
    )

    # Filter for pesticide products
    pesticide_codes = [
        "380852",
        "380859",
        "380861",
        "380862",
        "380869",
        "380891",
        "380892",
        "380893",
    ]
    df = df[df["product"].astype(str).isin(pesticide_codes)]

    # Load country code mapping
    country_codes = pd.read_csv(
        "../data/industry-sectors/BACI/country_codes_V202401b.csv"
    )
    country_code_map = dict(
        zip(country_codes["country_code"], country_codes["country_iso3"])
    )

    # Replace country codes with ISO3 codes
    df["exporter"] = df["exporter"].map(country_code_map)
    df["importer"] = df["importer"].map(country_code_map)

    # Group by exporter and importer, summing the value and quantity for all pesticide products
    df_summed = (
        df.groupby(["year", "exporter", "importer"]).agg({"value": "sum"}).reset_index()
    )

    # Rename the columns to reflect that these are now totals
    df_summed = df_summed.rename(columns={"value": "pesticide_total_value"})

    # Estimate tonnes of active ingredient traded
    # See https://docs.google.com/document/d/1GTqVgXl5T-gEt58ArWINwn4_qrew73lvMBdBlvxhtBs/edit?usp=drive_link
    # for the $1M = 40 tonnes conversion
    df_summed["pesticide_total_tonnes"] = df_summed["pesticide_total_value"] / 1000 * 40

    return df_summed


def load_pesticide_use():
    """
    Load pesticide use data for the year 2021 from the 'pesticide-use-tonnes.csv' file.

    Returns:
    pandas.DataFrame: A dataframe containing pesticide use data with columns:
        - iso3: ISO 3166-1 alpha-3 country code
        - use: Pesticide use in tonnes for the year 2021

    The function filters for data from 2021, drops rows with missing country codes,
    and renames columns for clarity.
    """
    import pandas as pd

    # Load the CSV file
    df = pd.read_csv("../data/industry-sectors/pesticide-use-tonnes.csv")

    # Filter for the year 2021
    df = df[df["Year"] == 2021]

    # Drop the 'Entity' column and rename 'Code' to 'iso3'
    df = df.drop(columns=["Entity"])
    df = df.rename(columns={"Code": "iso3"})

    # Drop rows where iso3 is empty
    df = df.dropna(subset=["iso3"])

    # Rename the last column to 'use'
    df = df.rename(columns={df.columns[-1]: "use"})

    # Select only the 'iso3' and 'use' columns
    df = df[["iso3", "use"]]
    df = df[df["iso3"] != "OWID_WRL"]

    return df


def calculate_pesticide_production():
    """
    Calculate pesticide production for each country based on trade and use data.

    This function combines data from bilateral pesticide trade and pesticide use
    to calculate the production of pesticides for each country.

    Returns:
    pandas.DataFrame: A dataframe containing pesticide production data with columns:
        - iso3: ISO 3166-1 alpha-3 country code
        - production: Estimated pesticide production in tonnes
        - percentage_of_world_production: Percentage of world production

    The production is calculated as: use + export - import
    """
    # Get bilateral pesticide trade data
    df = get_bilateral_pesticide_trade()

    # Get pesticide use data
    df2 = load_pesticide_use()

    # Calculate exports and imports for each country
    exports = (
        df.groupby("exporter")["pesticide_total_tonnes"]
        .sum()
        .reset_index(name="exports")
    )
    imports = (
        df.groupby("importer")["pesticide_total_tonnes"]
        .sum()
        .reset_index(name="imports")
    )

    # Merge exports and imports
    net_trade = exports.merge(
        imports, left_on="exporter", right_on="importer", how="outer"
    )

    # Fill NaN values with 0 for countries that only import or only export
    net_trade["exports"] = net_trade["exports"].fillna(0)
    net_trade["imports"] = net_trade["imports"].fillna(0)

    # Clean up the dataframe
    net_trade = net_trade.drop(columns=["importer"])
    net_trade = net_trade.rename(columns={"exporter": "iso3"})

    # Merge net_trade with pesticide use data
    production = net_trade.merge(df2, on="iso3", how="outer")

    # Calculate production (use + export - import)
    production["production"] = (
        production["use"].fillna(0) + production["exports"] - production["imports"]
    )

    # Select only iso3 and production columns
    production = production[["iso3", "production"]]

    # Sort by production in descending order and reset index
    production = production.sort_values("production", ascending=False).reset_index(
        drop=True
    )

    # Calculate the percentage of world production
    world_production = production["production"].sum()
    production["percentage_of_world_production"] = (
        100 * production["production"] / world_production
    )

    # Replace NaN values with 0
    production["percentage_of_world_production"] = production[
        "percentage_of_world_production"
    ].fillna(0)

    # There seems to be issues with Indonesia in the OWID data so we will remove it
    production = production[production["iso3"] != "IDN"]

    return production
