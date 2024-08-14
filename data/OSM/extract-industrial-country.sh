#!/bin/bash

echo "Welcome to the Industrial Landuse Data Extractor!"

# Declare an associative array to map countries to regions
declare -A country_region=(
    ["albania"]="europe"
    ["algeria"]="africa"
    ["argentina"]="south-america"
    ["australia"]="australia-oceania"
    ["austria"]="europe"
    ["bangladesh"]="asia"
    ["bahamas"]="central-america"
    ["belarus"]="europe"
    ["belgium"]="europe"
    ["bolivia"]="south-america"
    ["brazil"]="south-america"
    ["bulgaria"]="europe"
    ["canada"]="north-america"
    ["chile"]="south-america"
    ["china"]="asia"
    ["colombia"]="south-america"
    ["costa rica"]="central-america"
    ["croatia"]="europe"
    ["cuba"]="central-america"
    ["cyprus"]="europe"
    ["czech republic"]="europe"
    ["denmark"]="europe"
    ["ecuador"]="south-america"
    ["egypt"]="africa"
    ["estonia"]="europe"
    ["finland"]="europe"
    ["france"]="europe"
    ["germany"]="europe"
    ["greece"]="europe"
    ["guatemala"]="central-america"
    ["honduras"]="central-america"
    ["hungary"]="europe"
    ["iceland"]="europe"
    ["india"]="asia"
    ["indonesia"]="asia"
    ["iran"]="asia"
    ["iraq"]="asia"
    ["ireland"]="europe"
    ["israel"]="asia"
    ["italy"]="europe"
    ["japan"]="asia"
    ["jordan"]="asia"
    ["kazakhstan"]="asia"
    ["kenya"]="africa"
    ["kosovo"]="europe"
    ["latvia"]="europe"
    ["lebanon"]="asia"
    ["liechtenstein"]="europe"
    ["lithuania"]="europe"
    ["luxembourg"]="europe"
    ["malaysia"]="asia"
    ["malta"]="europe"
    ["mexico"]="north-america"
    ["moldova"]="europe"
    ["mongolia"]="asia"
    ["montenegro"]="europe"
    ["morocco"]="africa"
    ["netherlands"]="europe"
    ["new zealand"]="australia-oceania"
    ["nicaragua"]="central-america"
    ["north macedonia"]="europe"
    ["norway"]="europe"
    ["pakistan"]="asia"
    ["panama"]="central-america"
    ["paraguay"]="south-america"
    ["peru"]="south-america"
    ["philippines"]="asia"
    ["poland"]="europe"
    ["portugal"]="europe"
    ["romania"]="europe"
    ["russia"]="europe" 
    ["saudi arabia"]="asia"
    ["serbia"]="europe"
    ["singapore"]="asia"
    ["slovakia"]="europe"
    ["slovenia"]="europe"
    ["south africa"]="africa"
    ["south korea"]="asia"
    ["spain"]="europe"
    ["sri lanka"]="asia"
    ["sweden"]="europe"
    ["switzerland"]="europe"
    ["syria"]="asia"
    ["taiwan"]="asia"
    ["thailand"]="asia"
    ["tunisia"]="africa"
    ["turkey"]="europe"      
    ["ukraine"]="europe"
    ["united arab emirates"]="asia"
    ["united kingdom"]="europe"
    ["us"]="north-america"
    ["uruguay"]="south-america"
    ["venezuela"]="south-america"
    ["vietnam"]="asia"
)

# Get country name from the user
read -p "Enter the country name (e.g., France, Germany, US): " country

# Ensure the country name is lowercase to match Geofabrik's conventions and present in the array 
country=$(echo "$country" | tr '[:upper:]' '[:lower:]')

# Determine region based on country name
echo "Determining region for $country..."
if [[ -z "${country_region[$country]}" ]]; then
    echo "Error: Invalid country name or region not found. Please check your spelling or try another country."
    exit 1
else
    region="${country_region[$country]}"
    echo "Region identified: $region"
fi

# Replace spaces in country name with dashes to match Geofabrik's conventions
country=$(echo "$country" | tr ' ' '-')


# Construct the download URL
download_url="https://download.geofabrik.de/${region}/${country}-latest.osm.pbf"

# Check if the URL exists (using wget's spider mode)
echo "Checking data availability..."
if wget --spider "$download_url" 2>/dev/null; then
    echo "Data found! Downloading..."
    wget "$download_url"

    # Check if the download was successful
    if [[ $? -ne 0 ]]; then
        echo "Error: Download failed. Please check your internet connection and try again."
        exit 1
    fi
    echo "Download complete!"
else
    # If the URL is invalid, direct the user to the Geofabrik homepage
    echo "Error: Data not found for this country and region. Please check the Geofabrik website for available data: https://download.geofabrik.de/"
    exit 1
fi

# Convert to .osm format (this can take some time for larger countries)
echo "Converting to .osm format..."
osmconvert "${country}-latest.osm.pbf" -o="${country}-latest.osm"
echo "Conversion complete."

# Filter to keep only industrial landuse data
echo "Filtering for industrial landuse data..."
osmfilter "${country}-latest.osm" --keep="landuse=industrial" -o="${country}-industrial.osm"
echo "Filtering complete."

# Remove unnecessary files (comment out if you want to keep them)
echo "Cleaning up temporary files..."
rm "${country}-latest.osm" "${country}-latest.osm.pbf"

echo "Processing complete! Industrial landuse data for $country saved in ${country}-industrial.osm"
