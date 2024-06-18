# Population-Loss-NW
Estimate fatalities in the direct aftermath of a nuclear war


## Methodology
Here we use the methodology of [Toon et al. 2007](https://acp.copernicus.org/articles/7/1973/2007/acp-7-1973-2007.pdf) and [Toon et al. 2008](https://pubs.aip.org/physicstoday/article/61/12/37/393240/Environmental-consequences-of-nuclear-warA) to estimate the number of fatalities in the aftermath of a nuclear war.

In Hiroshima and Nagasaki data a normal distribution around ground zero was measured for the fatality rate, $\alpha(R) = e^{-\frac{R^2}{2 \sigma^2}}$, where $R$ is the distance from ground zero and $\sigma=1.15$ km for a 15 kt airburst. Following Toon et al. 2008, the width of this distribution is assumed to scale as $\sqrt{\frac{Y}{15\,{\rm kt}}}$, where $Y$ is the yield of the nuclear weapon. This is so that the area with a given $\alpha(R)$ contours scales linearly with $Y$. Note that this excludes fatalities related to radioactive fallout, which depends on a number of hard to predict factors (sheltering, evacuation, weather, etc.).

Using [LandScan](https://landscan.ornl.gov/) data for population, we can estimate the number of fatalities in the immediate aftermath of a nuclear war by integrating over the distribution of distances from ground zero.

Currently, targets are selected by finding for a given country where to detonate a given number of warheads over the country's most populated region and without overlapping targets (following the Toon et al. methodology). For example, here are the results for 50 100-kt non-overlapping strikes on Germany. The areas colored in red are regions where fatalities are expected from the blasts and ensuing fires. 14 million fatalities are expected in this scenario.

![200 100-kt striked on Germany](images/germany-50-100kt-example.png) 

## Limitations
* Nuclear fallout is not considered.
* The current non-overlapping target allocation algorithm will not handle correctly a case where the nuclear arsenal hitting a country is made of warheads with different yields.
* The code requires quite a bit of RAM if the target country is large. If this is an issue, you can use the `degrade` option to degrade the resolution of the LandScan data. The original resolution is 30 arc-seconds, so the individual pixels are somewhat smaller than 1 km² for the regions most susceptible to nuclear war.

## Codebase orientation
Simply use `scripts/master.ipynb` to calculate the number of fatalities in a nuclear war given an attack with a given number of warheads against a given country. All the code is in `src/main.py`. `results` contains the number of fatalities for different scenarios.

## Verification
To verify that the implementation is correct, we can compare to the [results](https://pubs.aip.org/view-large/figure/45882429/37_1_f1.jpg) of Toon et al. Below is a comparison between the number of casualties (in millions) in different scenarios. Note that this includes fatalities and injuries to facilitate the comparison with the results of Toon et al. Everything seems to work ok. Some numbers are significantly higher, but this can be attributed to population increase over the years (India in particular).


| Scenario | Toon et al. | This code |
|----------|----------|----------|
| Pakistan, 50x 15kt  | 18   |  22  |
| Pakistan, 200x 100kt  | 50   |  66  |
| UK, 50x 15kt | 6 | 6 |
| UK, 200x 100kt | 28 | 29 |
| Germany, 200x 100kt | 28 | 26 |
| India, 50x 15kt | 26 | 34 |
| India, 200x 100kt | 116 | 172 |
| Japan, 50x 15kt | 13 | 11 |
| Japan, 200x 100kt | 59 | 50 |
| US, 50x 15kt | 8 | 8 |
| US, 1000x 100kt | 104 | 108 |
| Russia, 50x 15kt | 12 | 8 |
| Russia, 1100x 100kt | 76 | 66 |
| China, 50x 15kt | 32 | 24 |
| China, 1100x 100kt | 287 | 314 |
| France, 50x 15kt | 7 | 6 |
| France, 200x 100kt | 23 | 20 |


## Scenarios considered
See the `results` directory for the results of the scenarios considered.

### `Toon2008_SORT`
This scenario is based on [Toon et al. 2008](https://pubs.aip.org/physicstoday/article/61/12/37/393240/Environmental-consequences-of-nuclear-war). In this scenario,  we assume that Russia targets 1000 weapons on the US and 200 warheads each on France, Germany, India, Japan, Pakistan, and the UK. We assume the US targets 1100 weapons each on China and Russia. Targets are selected by finding for a given country where to detonate a given number of warheads over the country's most populated region and without overlapping targets. In this scenario, total fatalities reach 570 million.
