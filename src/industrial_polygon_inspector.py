#!/usr/bin/env python3
"""
Industrial Polygon Inspector

This script allows you to inspect industrial polygons from OpenStreetMap data 
for the United States, Pakistan, and India. It randomly samples N polygons 
from each country and provides a single interactive map with high-quality 
satellite imagery for visual assessment.

Usage:
    python industrial_polygon_inspector.py

Features:
- Load OSM industrial polygon data using existing functions
- Random sampling of polygons by country
- Single interactive map with multiple satellite imagery options
- Google Maps links for detailed inspection
- Summary statistics and export capabilities
"""

import sys
import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import webbrowser
from pathlib import Path

# Import from main.py
from main import get_industrial_areas_from_osm, calculate_distance_km

class IndustrialPolygonInspector:
    """Main class for inspecting industrial polygons"""
    
    def __init__(self, n_samples=3, random_seed=42):
        """
        Initialize the inspector
        
        Args:
            n_samples (int): Number of polygons to sample per country
            random_seed (int): Random seed for reproducible results
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # File paths for OSM data
        self.osm_files = {
            'USA': '../data/OSM/united-states-of-america-industrial.osm',
            'Pakistan': '../data/OSM/pakistan-industrial.osm',
            'India': '../data/OSM/india-industrial.osm'
        }
        
        # Storage for loaded data
        self.country_data = {}
        self.sampled_polygons = {}
        
        print(f"🏭 Industrial Polygon Inspector initialized")
        print(f"📊 Sampling {n_samples} polygons per country")
        print(f"🎲 Random seed: {random_seed}")
    
    def load_country_data(self):
        """Load industrial polygon data from all countries"""
        print("\n🔄 Loading industrial polygon data...")
        
        for country, file_path in self.osm_files.items():
            print(f"\n=== Processing {country} ===")
            
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
                
            try:
                # Use the existing function from main.py
                gdf = get_industrial_areas_from_osm(file_path)
                
                if len(gdf) > 0:
                    # Calculate areas
                    gdf = gdf.to_crs("EPSG:3857")  # Project to meters
                    gdf['area_sqm'] = gdf.geometry.area
                    gdf['area_hectares'] = gdf['area_sqm'] / 10000
                    gdf = gdf.to_crs("EPSG:4326")  # Back to lat/lon
                    
                    original_count = len(gdf)
                    
                    # Filter out very large polygons (greater than 10 km² = 1000 hectares)
                    # This excludes quarries, mining areas, and potential data errors
                    gdf_filtered = gdf[gdf['area_hectares'] <= 1000.0].copy()
                    large_excluded = original_count - len(gdf_filtered)
                    
                    # Add metadata
                    gdf_filtered['country'] = country
                    gdf_filtered['centroid_lat'] = gdf_filtered.geometry.centroid.y
                    gdf_filtered['centroid_lon'] = gdf_filtered.geometry.centroid.x
                    
                    print(f"✅ Loaded {len(gdf_filtered)} industrial polygons")
                    print(f"📏 Area range: {gdf_filtered['area_hectares'].min():.2f} - {gdf_filtered['area_hectares'].max():.2f} hectares")
                    if large_excluded > 0:
                        print(f"🔍 Excluded {large_excluded} polygons > 1000 hectares (10 km²)")
                    
                    self.country_data[country] = gdf_filtered
                else:
                    print(f"❌ No valid polygons found for {country}")
                    
            except Exception as e:
                print(f"❌ Error processing {country}: {e}")
        
        print(f"\n✅ Successfully loaded data for {len(self.country_data)} countries")
    
    def sample_polygons(self):
        """Sample random polygons from each country using area-weighted sampling"""
        print(f"\n🎲 Sampling {self.n_samples} polygons per country (weighted by area)...")
        
        for country, gdf in self.country_data.items():
            if len(gdf) >= self.n_samples:
                # Use area as weights for sampling (larger polygons more likely to be selected)
                weights = gdf['area_hectares'].values
                sample = gdf.sample(n=self.n_samples, weights=weights, random_state=self.random_seed)
                self.sampled_polygons[country] = sample
                print(f"📍 {country}: Sampled {len(sample)} polygons (area-weighted)")
                
                # Show the range of areas in the sample
                min_area = sample['area_hectares'].min()
                max_area = sample['area_hectares'].max()
                total_area = sample['area_hectares'].sum()
                print(f"   📏 Sample areas: {min_area:.1f} - {max_area:.1f} ha (total: {total_area:.1f} ha)")
            else:
                self.sampled_polygons[country] = gdf
                print(f"📍 {country}: Only {len(gdf)} polygons available (less than {self.n_samples})")
        
        # Display summary statistics
        print("\n📊 Area-Weighted Sample Summary:")
        for country, sample in self.sampled_polygons.items():
            total_country_area = self.country_data[country]['area_hectares'].sum()
            sample_area = sample['area_hectares'].sum()
            coverage_pct = (sample_area / total_country_area) * 100
            
            print(f"\n{country}:")
            print(f"  Count: {len(sample)} polygons")
            print(f"  Area range: {sample['area_hectares'].min():.2f} - {sample['area_hectares'].max():.2f} hectares")
            print(f"  Mean area: {sample['area_hectares'].mean():.2f} hectares")
            print(f"  Total sample area: {sample_area:.1f} ha ({coverage_pct:.1f}% of country's industrial area)")
    
    def create_google_maps_url(self, lat, lon, zoom=18):
        """Create Google Maps URL for given coordinates"""
        return f"https://www.google.com/maps/@{lat},{lon},{zoom}z"
    
    def create_comprehensive_map(self):
        """Create a single comprehensive map with all sampled polygons"""
        print(f"\n🗺️  Creating comprehensive inspection map...")
        
        # Calculate center point for map
        all_lats = []
        all_lons = []
        
        for sample_gdf in self.sampled_polygons.values():
            for _, row in sample_gdf.iterrows():
                all_lats.append(row['centroid_lat'])
                all_lons.append(row['centroid_lon'])
        
        if not all_lats:
            print("❌ No polygons to display")
            return None
        
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        # Create main map with Google satellite as default
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,  # Higher zoom to see polygons better
            tiles=None
        )
        
        # Add Google Satellite as the primary/default layer
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Maps Satellite',
            name='Google Satellite (Default)',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add Google Hybrid (satellite + labels)
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google Maps Hybrid',
            name='Google Hybrid',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add OpenStreetMap for reference only
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap (Reference)').add_to(m)
        
        # Color and style mapping for countries with more prominent styling
        country_styles = {
            'USA': {'color': 'red', 'fillColor': 'red', 'icon_color': 'red'},
            'Pakistan': {'color': 'lime', 'fillColor': 'lime', 'icon_color': 'green'},
            'India': {'color': 'yellow', 'fillColor': 'yellow', 'icon_color': 'orange'}
        }
        
        # Add polygons to map with enhanced visibility
        polygon_count = 0
        for country, sample_gdf in self.sampled_polygons.items():
            style = country_styles.get(country, {'color': 'blue', 'fillColor': 'blue', 'icon_color': 'blue'})
            
            for idx, (_, row) in enumerate(sample_gdf.iterrows()):
                polygon_count += 1
                google_url = self.create_google_maps_url(row['centroid_lat'], row['centroid_lon'])
                
                # Create detailed popup content with styling
                popup_content = f"""
                <div style="width: 350px; font-family: Arial, sans-serif;">
                    <h3 style="color: {style['color']}; margin-top: 0px; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                        🏭 {country} - Polygon #{idx+1}
                    </h3>
                    <table style="width: 100%; font-size: 14px; margin: 10px 0;">
                        <tr><td><b>ID:</b></td><td>{row['id']}</td></tr>
                        <tr><td><b>Area:</b></td><td>{row['area_hectares']:.2f} hectares</td></tr>
                        <tr><td><b>Area (m²):</b></td><td>{row['area_sqm']:.0f} m²</td></tr>
                        <tr><td><b>Coordinates:</b></td><td>{row['centroid_lat']:.6f}, {row['centroid_lon']:.6f}</td></tr>
                    </table>
                    <div style="text-align: center; margin: 15px 0;">
                        <a href="{google_url}" target="_blank" 
                           style="background-color: #4285f4; color: white; 
                                  padding: 10px 20px; text-decoration: none; 
                                  border-radius: 5px; display: inline-block; font-weight: bold;">
                            🌍 Open in Google Maps
                        </a>
                    </div>
                    <div style="margin-top: 15px; padding: 10px; background-color: #e8f4fd; border-radius: 5px; border-left: 4px solid #4285f4;">
                        <small><b>💡 How to inspect:</b><br>
                        1. Make sure you're on "Google Satellite" layer<br>
                        2. Zoom in to see the polygon clearly<br>
                        3. Click "Open in Google Maps" for detailed view<br>
                        4. Compare the outlined area with satellite imagery</small>
                    </div>
                </div>
                """
                
                # Add polygon with much more prominent styling
                if hasattr(row['geometry'], 'exterior'):
                    coords = [[point[1], point[0]] for point in row['geometry'].exterior.coords]
                    
                    folium.Polygon(
                        locations=coords,
                        color=style['color'],
                        weight=4,  # Thicker outline
                        fillColor=style['fillColor'],
                        fillOpacity=0.6,  # More opaque
                        opacity=1.0,  # Fully opaque outline
                        popup=folium.Popup(popup_content, max_width=380),
                        tooltip=f"🏭 {country} - Polygon #{idx+1} ({row['area_hectares']:.1f} ha) - CLICK FOR DETAILS"
                    ).add_to(m)
                
                # Add large, prominent marker at centroid
                folium.CircleMarker(
                    [row['centroid_lat'], row['centroid_lon']],
                    radius=12,  # Larger radius
                    popup=folium.Popup(popup_content, max_width=380),
                    tooltip=f"🏭 {country} - #{idx+1}",
                    color='white',
                    weight=3,
                    fillColor=style['color'],
                    fillOpacity=1.0
                ).add_to(m)
                
                # Add a pulsing marker for extra visibility
                folium.plugins.BeautifyIcon(
                    icon='industry',
                    iconShape='circle',
                    borderColor=style['color'],
                    borderWidth=3,
                    textColor='white',
                    backgroundColor=style['color']
                ).add_to(folium.Marker(
                    [row['centroid_lat'], row['centroid_lon']],
                    popup=folium.Popup(popup_content, max_width=380)
                ).add_to(m))
        
        # Add custom legend with better visibility info
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 280px; 
                    background-color: white; border: 3px solid #333; z-index:9999; 
                    font-size:14px; padding: 15px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.3);">
        <h3 style="margin-top: 0px; color: #333;">🏭 Industrial Polygons Inspector</h3>
        <div style="border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-bottom: 10px;">
            <p><span style="color:red; font-size: 18px;">●</span> <b>USA</b> ({len(self.sampled_polygons.get("USA", []))} polygons)</p>
            <p><span style="color:lime; font-size: 18px;">●</span> <b>Pakistan</b> ({len(self.sampled_polygons.get("Pakistan", []))} polygons)</p>
            <p><span style="color:gold; font-size: 18px;">●</span> <b>India</b> ({len(self.sampled_polygons.get("India", []))} polygons)</p>
        </div>
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <b>🎯 Quick Start:</b><br>
            1. <b>Zoom in</b> to see polygons clearly<br>
            2. <b>Click colored areas</b> for details<br>
            3. Use <b>Google Maps links</b> for comparison
        </div>
        <div style="background-color: #fff3cd; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;">
            <small><b>⚠️ Can't see polygons?</b><br>
            Zoom in more! Some polygons are very small and only visible at higher zoom levels.</small>
        </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control in a prominent position
        folium.LayerControl(position='topleft', collapsed=False).add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen(position='topleft').add_to(m)
        
        # Add a mini map for better navigation
        minimap = plugins.MiniMap(toggle_display=True, position='bottomleft')
        m.add_child(minimap)
        
        # Add zoom to bounds of all polygons button
        if polygon_count > 0:
            # Calculate bounds of all polygons
            all_coords = []
            for sample_gdf in self.sampled_polygons.values():
                for _, row in sample_gdf.iterrows():
                    all_coords.append([row['centroid_lat'], row['centroid_lon']])
            
            if all_coords:
                # Add fit bounds functionality
                bounds = [[min(coord[0] for coord in all_coords), min(coord[1] for coord in all_coords)],
                         [max(coord[0] for coord in all_coords), max(coord[1] for coord in all_coords)]]
                m.fit_bounds(bounds, padding=[20, 20])
        
        # Add search functionality
        plugins.Search(
            layer=folium.FeatureGroup().add_to(m),
            search_label='name',
            placeholder='Search polygons...',
            collapsed=False,
            position='topleft'
        ).add_to(m)
        
        # Save comprehensive map
        map_filename = "industrial_polygons_inspector.html"
        m.save(map_filename)
        print(f"💾 Comprehensive map saved: {map_filename}")
        print(f"🎯 Total polygons on map: {polygon_count}")
        print(f"🌍 Using Google Maps satellite imagery as default")
        print(f"🔍 Tip: Zoom in to see small polygons clearly!")
        
        return m, map_filename
    
    def display_polygon_info(self, polygon_row, country_name, index):
        """Display detailed information about a polygon"""
        centroid_lat = polygon_row['centroid_lat']
        centroid_lon = polygon_row['centroid_lon']
        google_maps_url = self.create_google_maps_url(centroid_lat, centroid_lon)
        
        print(f"\n📍 {country_name} - Polygon {index + 1}")
        print(f"   ID: {polygon_row['id']} | Area: {polygon_row['area_hectares']:.2f} ha")
        print(f"   Coords: {centroid_lat:.6f}, {centroid_lon:.6f}")
        print(f"   🗺️  {google_maps_url}")
    
    def create_summary_report(self):
        """Create a summary report with all polygon information"""
        print(f"\n📋 Creating summary report...")
        
        # Create summary data
        summary_data = []
        
        for country, sample_gdf in self.sampled_polygons.items():
            for idx, (_, row) in enumerate(sample_gdf.iterrows()):
                google_maps_url = self.create_google_maps_url(row['centroid_lat'], row['centroid_lon'])
                
                summary_data.append({
                    'Country': country,
                    'Polygon_Number': idx + 1,
                    'Polygon_ID': row['id'],
                    'Area_Hectares': row['area_hectares'],
                    'Area_SqM': row['area_sqm'],
                    'Centroid_Latitude': row['centroid_lat'],
                    'Centroid_Longitude': row['centroid_lon'],
                    'Google_Maps_URL': google_maps_url
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_filename = 'industrial_polygons_summary.csv'
        summary_df.to_csv(summary_filename, index=False)
        print(f"💾 Summary saved: {summary_filename}")
        
        # Display compact summary
        print(f"\n📊 SUMMARY:")
        for country in summary_df['Country'].unique():
            country_data = summary_df[summary_df['Country'] == country]
            print(f"   {country}: {len(country_data)} polygons, "
                  f"avg area: {country_data['Area_Hectares'].mean():.1f} ha")
        
        return summary_df, summary_filename
    
    def run_inspection(self):
        """Run the complete inspection process"""
        print(f"\n🚀 Starting Industrial Polygon Inspection")
        print(f"{'='*60}")
        
        # Load data
        self.load_country_data()
        
        if not self.country_data:
            print("❌ No data loaded. Exiting.")
            return
        
        # Sample polygons (now area-weighted)
        self.sample_polygons()
        
        if not self.sampled_polygons:
            print("❌ No polygons sampled. Exiting.")
            return
        
        # Display basic info for each polygon
        print(f"\n📋 Sampled Polygons:")
        for country, sample_gdf in self.sampled_polygons.items():
            print(f"\n--- {country} ---")
            for idx, (_, polygon_row) in enumerate(sample_gdf.iterrows()):
                self.display_polygon_info(polygon_row, country, idx)
        
        # Create comprehensive map
        map_obj, map_filename = self.create_comprehensive_map()
        
        # Create summary report
        summary_df, summary_filename = self.create_summary_report()
        
        # Final summary
        print(f"\n🎉 INSPECTION COMPLETE!")
        print(f"{'='*60}")
        print(f"📊 Total polygons inspected: {len(summary_df)}")
        print(f"🗂️  Files created:")
        print(f"   📄 {map_filename} (main inspection map)")
        print(f"   📄 {summary_filename} (polygon data)")
        
        print(f"\n💡 How to use:")
        print(f"   1. Open {map_filename} in your browser")
        print(f"   2. Switch to satellite view (Google/Esri layers)")
        print(f"   3. Click on colored polygons to see details")
        print(f"   4. Use Google Maps links for detailed comparison")
        print(f"   5. Larger polygons were more likely to be selected (area-weighted sampling)")
        
        return {
            'summary_df': summary_df,
            'map_file': map_filename,
            'summary_file': summary_filename
        }


def main():
    """Main function to run the industrial polygon inspector"""
    print("🏭 Industrial Polygon Inspector")
    print("===============================")
    
    # Configuration
    N_SAMPLES = 50  # Adjust this to sample more or fewer polygons
    RANDOM_SEED = 42
    
    # Create inspector and run
    inspector = IndustrialPolygonInspector(
        n_samples=N_SAMPLES,
        random_seed=RANDOM_SEED
    )
    
    try:
        results = inspector.run_inspection()
        
        # Open the map in browser
        if 'map_file' in results:
            print(f"\n🌐 Opening inspection map in browser...")
            webbrowser.open(results['map_file'])
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Inspection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during inspection: {e}")
        print(f"💡 Check that the OSM data files exist in the data/OSM/ directory")


if __name__ == "__main__":
    main() 