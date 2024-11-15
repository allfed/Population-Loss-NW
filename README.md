# Industrial infrastructure loss due to nuclear war
Estimate loss of industry in the direct aftermath of a nuclear war. 

## Installation
1. Clone the repo on your local machine.
2. Create the Conda environment using `conda env create -f environment.yml`.
3. Activate the new environment using `conda activate Industry-Loss-NW`.

## Methodology
The methodology is fully described in the paper. Here we provide a brief overview.

We assess the direct destruction of industrial infrastructure using OpenStreetMap data. Industrial areas that overlap with the burn radius of a weapon are considered destroyed. The burn radius is calculated using the methodology described in `scripts/burn-radius-scaling.ipynb`.

Using [LandScan](https://landscan.ornl.gov/) data for population, we also estimate the number of fatalities in the immediate aftermath of a nuclear war by integrating over the distribution of distances from ground zero. Here we use a methodology similar to [Toon et al. 2007](https://acp.copernicus.org/articles/7/1973/2007/acp-7-1973-2007.pdf) and [Toon et al. 2008](https://pubs.aip.org/physicstoday/article/61/12/37/393240/Environmental-consequences-of-nuclear-warA) to estimate fatality rates.

## Data sources
* [LandScan](https://landscan.ornl.gov/) for population data
* [OSM](https://download.geofabrik.de/) for industrial data

## Codebase orientation
1. Simply use `scripts/scenarios.ipynb` to run the scenarios presented in the paper. It contains the code to calculate the number of fatalities and destruction of industrial infrastructure in the Pakistan, India, and US nuclear attack scenarios detailed in the paper.

2. The source code is in `src`.

3. `scripts/burn-radius-scaling.ipynb` details how the burn radius scaling law was derived.