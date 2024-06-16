# Population-Loss-NW
Estimate fatalities in the direct aftermath of a nuclear war


## Methodology
Here we use the methodology of [Toon et al. 2007](https://acp.copernicus.org/articles/7/1973/2007/acp-7-1973-2007.pdf) and [Toon et al. 2008](https://pubs.aip.org/physicstoday/article/61/12/37/393240/Environmental-consequences-of-nuclear-warA) to estimate the number of fatalities in the aftermath of a nuclear war.

Hiroshima and Nagasaki data give a normal distribution around ground zero for the fatality rate, $\alpha(R) = e^{-\frac{R^2}{2 \sigma^2}}$, where $R$ is the distance from ground zero and $\sigma=1.15$ km for a 15 kt airburst. Following Toon et al. 2008, the width of this distribution is assumed to scale as $\sqrt{\frac{Y}{15\,{\rm kt}}}$, where $Y$ is the yield of the nuclear weapon. This is so that the area with a given $\alpha(R)$ contours scales linearly with $Y$. Note that this excludes fatalities related to radioactive fallout, which depends on a number of hard to predict factors (sheltering, evacuation, weather, etc.).

Using [LandScan](https://landscan.ornl.gov/) data for population, we can estimate the number of fatalities in the immediate aftermath of a nuclear war by integrating over the distribution of distances from ground zero.

Currently, targets are selected by finding for a given country where to detonate a given number of warheads over the country's most populated region and without overlapping targets.

## To do
* Write codebase orientation
* Perform verification by comparing to Toon results
* Can't mix kt types and optimal no overlap, but optimal no overlap works if just one kt type is used
* Reinstate the degrade resolution function, which is better to find the best targets than the 1km resolution that can give weird results

## Codebase orientation