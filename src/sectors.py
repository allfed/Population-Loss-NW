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

    # Print a nicely formatted table of sector losses
    print("\nLosses per sector for the whole world:")
    print("-" * 40)
    print(f"{'Sector':<20} {'Loss (%)':<10}")
    print("-" * 40)
    for sector, loss in sector_losses.items():
        sector_name = sector.replace('_loss_percent', '').replace('_', ' ').capitalize()
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
    total_losses = {'N_loss': 0, 'P_loss': 0, 'K_loss': 0}

    # Calculate losses for each country and sum them up
    for country, loss_percentage in total_industry_loss.items():
        if country in fertilizers['Code'].values:
            country_data = fertilizers[fertilizers['Code'] == country].iloc[0]
            total_losses['N_loss'] += country_data['N_percent'] * loss_percentage / 100
            total_losses['P_loss'] += country_data['P_percent'] * loss_percentage / 100
            total_losses['K_loss'] += country_data['K_percent'] * loss_percentage / 100
        else:
            print(f"Country {country} not found in fertilizer production data")

    sector_losses['N_fertilizer_loss_percent'] = total_losses['N_loss']
    sector_losses['P_fertilizer_loss_percent'] = total_losses['P_loss']
    sector_losses['K_fertilizer_loss_percent'] = total_losses['K_loss']

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
