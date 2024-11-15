import osmium
import geopandas as gpd
from shapely.geometry import Polygon


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