import pandas as pd

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
    df2 = pd.read_csv("../data/target-lists/OPEN-RISOP 1.00 TARGET DATABASE.csv")

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

        # Basic data validation
        assert -90 <= lat <= 90, f"Invalid latitude for {name}: {lat}"
        assert -180 <= lon <= 180, f"Invalid longitude for {name}: {lon}"
        assert hob >= 0, f"Invalid height of burst for {name}: {hob}"
        assert ykt > 0, f"Invalid yield for {name}: {ykt}"

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

        if icbm_only and subclass != "ICBM SILOS":
            continue

        targets[name] = (lat, lon, hob, ykt)
        total_yield += ykt

    print(f"Total yield: {total_yield} kt")
    return targets
