# app.py

# --- Section 1: Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Point, LineString
from datetime import datetime, timedelta
import random
import math
import os
import joblib 
from sklearn.metrics import accuracy_score
# You might have other imports like folium, etc. Keep them.

# --- Section 2: Configuration and Caching ---

# Define your place of interest.
place_name = "Deoria, Uttar Pradesh, India"

# Define road filters (matching main_project.py)
custom_filter_highway = {
    "highway": [
        "motorway", "trunk", "primary", "secondary"
    ]
}
custom_filter_service = {
    "highway": [
        "service", "unclassified"
    ]
}

# Define data directories
base_data_dir = "data"
map_data_output_dir = os.path.join(base_data_dir, "map_data")
trajectory_output_dir = os.path.join(base_data_dir, "trajectory_data")
feature_output_dir = os.path.join(base_data_dir, "features")
model_output_dir = os.path.join(base_data_dir, "model")
model_path = os.path.join(model_output_dir, "road_type_classifier_model.joblib")

# Ensure directories exist
os.makedirs(map_data_output_dir, exist_ok=True)
os.makedirs(trajectory_output_dir, exist_ok=True)
os.makedirs(feature_output_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)

@st.cache_resource
def load_map_data(place_name): # Ensure only 1 argument here
    st.info(f"Loading map data for {place_name}... (This happens once per app run)")
    try:
        # REMOVED 'custom_filters' argument here
        G_all = ox.graph_from_place(place_name, network_type="all")

        # Get edges as GeoDataFrame
        edges = ox.graph_to_gdfs(G_all, nodes=False, edges=True)

        # DEBUG PRINT: Keep this, it's helpful
        if 'highway' in edges.columns:
            unique_highway_tags = edges['highway'].explode().unique()
            st.write("Debug: Unique 'highway' tags found in loaded map data:")
            st.write(unique_highway_tags)
            print(f"Debug: Unique 'highway' tags found: {unique_highway_tags}")
        else:
            st.write("Debug: 'highway' column not found in loaded map edges.")
            print("Debug: 'highway' column not found in loaded map edges.")

        # Broadened these types to capture more data from OSM
        highway_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
                         'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
                         'road', 'unclassified', 'residential'] # Added more general types
        gdf_highways = edges[edges['highway'].apply(lambda x: any(ht in str(x) for ht in highway_types))]
        gdf_highways_geo = gdf_highways.to_crs(epsg=4326)

        service_road_types = ['service', 'residential', 'unclassified', 'track', 'path', 'pedestrian', 'cycleway'] # Added more general types
        gdf_service_roads = edges[edges['highway'].apply(lambda x: any(srt in str(x) for srt in service_road_types))]
        gdf_service_roads_geo = gdf_service_roads.to_crs(epsg=4326)

        st.success("Map data loaded successfully!")
        st.session_state.G_all = G_all # THIS LINE MUST BE PRESENT
        return gdf_highways_geo, gdf_service_roads_geo
    except Exception as e:
        st.error(f"Error loading map data: {e}. Please check the place name or your internet connection.")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()

# --- app.py (around line 75) ---

@st.cache_resource
def load_ml_model(model_file_path):
    """Loads the pre-trained ML model with caching."""
    st.info("Loading ML model... (This happens once per app run)")
    try:
        model = joblib.load(model_file_path)
        st.success("Model loaded successfully!")
        # ADD THESE LINES STARTING HERE
        if hasattr(model, 'feature_names_in_'):
            print(f"--- DEBUG: Model expects features in this order: {list(model.feature_names_in_)} ---")
        else:
            print("--- DEBUG: Model does not have 'feature_names_in_' attribute. Check scikit-learn version or model type. ---")
        # ADD THESE LINES ENDING HERE
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_file_path}. Please run main_project.py to train and save the model.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# --- Section 3: Helper Functions for Simulation ---

# Helper function: generate_noisy_point
def generate_noisy_point(lat, lon, noise_meters=10):
    """Adds random noise to lat/lon coordinates in meters."""
    # Approximate meters per degree lat/lon
    # Latitude: ~111,320 meters per degree
    # Longitude: ~111,320 * cos(latitude) meters per degree
    lat_noise = (random.uniform(-1, 1) * noise_meters) / 111320
    lon_noise = (random.uniform(-1, 1) * noise_meters) / (111320 * np.cos(np.radians(lat)))
    return lat + lat_noise, lon + lon_noise

# --- app.py (your create_simulated_trajectory function) ---

# --- app.py (Approximate Line 100-150 - Define create_simulated_trajectory) ---

# --- Replace your existing create_simulated_trajectory function with this entire block ---
# --- REPLACE YOUR ENTIRE create_simulated_trajectory FUNCTION WITH THIS BLOCK ---
def create_simulated_trajectory(num_points, noise_meters, road_label, graph):
    """
    Generates a simulated GPS trajectory along a randomly selected road segment
    from the given graph, ensuring it's of the specified road_label type.
    """
    if not graph.nodes:
        st.error("OSMnx graph is empty. Cannot simulate trajectory.")
        return pd.DataFrame()

    # Define target highway types based on the requested road_label AND your unique tags
    # Unique tags were: ['trunk', 'unclassified', 'tertiary', 'residential', 'secondary', 'service', 'track', 'path', 'trunk_link', 'primary']

    selected_target_types = []
    if road_label == "Highway":
        selected_target_types = ['trunk', 'primary', 'secondary', 'tertiary', 'trunk_link']
        # 'unclassified' is common, include for more coverage if appropriate
        if 'unclassified' not in selected_target_types:
             selected_target_types.append('unclassified')
        if 'secondary' not in selected_target_types:
             selected_target_types.append('secondary')
    elif road_label == "Service Road":
        selected_target_types = ['residential', 'service', 'track', 'path']
        # 'unclassified' can also be a service road type
        if 'unclassified' not in selected_target_types:
            selected_target_types.append('unclassified')
        if 'tertiary' not in selected_target_types: # tertiary can sometimes act as service roads
            selected_target_types.append('tertiary')
    else: # Fallback for any other label or if no specific type is requested
        selected_target_types = ['trunk', 'unclassified', 'tertiary', 'residential', 'secondary', 'service', 'track', 'path', 'trunk_link', 'primary']

    selected_target_types = list(set(selected_target_types)) # Remove any duplicates

    selected_segment_geometry = None
    MAX_RETRIES = 500 # Increased retries further for robust testing

    for _ in range(MAX_RETRIES):
        start_node = random.choice(list(graph.nodes))

        possible_edges = []
        for u, v, key, data in graph.edges(start_node, keys=True, data=True):
            if 'highway' in data:
                highway_tags = data['highway']
                if isinstance(highway_tags, list):
                    if any(h_type in selected_target_types for h_type in highway_tags):
                        possible_edges.append((u, v, key))
                elif isinstance(highway_tags, str):
                    if highway_tags in selected_target_types:
                        possible_edges.append((u, v, key))

        if possible_edges:
            chosen_u, chosen_v, chosen_key = random.choice(possible_edges)
            selected_segment_data = graph.edges[chosen_u, chosen_v, chosen_key]

            selected_segment_geometry = selected_segment_data.get('geometry')
            if selected_segment_geometry is None:
                selected_segment_geometry = LineString([
                    (graph.nodes[chosen_u]['x'], graph.nodes[chosen_u]['y']),
                    (graph.nodes[chosen_v]['x'], graph.nodes[chosen_v]['y'])
                ])

            if selected_segment_geometry.length > 0:
                break # Found a suitable segment

    if selected_segment_geometry is None or selected_segment_geometry.length == 0:
        st.warning(f"Failed to find a suitable '{road_label}' segment after {MAX_RETRIES} retries. Map data might be too sparse or tags too specific.")
        return pd.DataFrame()

    points_on_segment = []
    if num_points < 2: num_points = 2
    for i in range(num_points):
        fraction = i / (num_points - 1)
        point = selected_segment_geometry.interpolate(fraction, normalized=True)
        points_on_segment.append((point.y, point.x))

    trajectory_data = []
    start_time = datetime.now()

    for i, (lat, lon) in enumerate(points_on_segment):
        current_noise = noise_meters if road_label == "Highway" else noise_meters * 1.5
        noisy_lat, noisy_lon = generate_noisy_point(lat, lon, noise_meters=current_noise)
        timestamp = start_time + timedelta(seconds=i * random.uniform(2, 5))

        if road_label == "Highway": speed_kph = random.uniform(60, 100)
        elif road_label == "Service Road": speed_kph = random.uniform(20, 40)
        else: speed_kph = random.uniform(30, 70)

        heading = 0
        if i < len(points_on_segment) - 1:
            next_lat, next_lon = points_on_segment[i+1]
            delta_lon = next_lon - lon
            y = np.sin(np.radians(delta_lon)) * np.cos(np.radians(next_lat))
            x = np.cos(np.radians(lat)) * np.sin(np.radians(next_lat)) - \
                np.sin(np.radians(lat)) * np.cos(np.radians(next_lat)) * np.cos(np.radians(delta_lon))
            heading = np.degrees(np.arctan2(y, x))
            heading = (heading + 360) % 360
        elif trajectory_data:
            heading = trajectory_data[-1]['heading']

        trajectory_data.append({
            'timestamp': timestamp,
            'latitude': noisy_lat,
            'longitude': noisy_lon,
            'speed_kph': speed_kph,
            'heading': heading,
            'road_type_label': road_label
        })
    return pd.DataFrame(trajectory_data)

# --- Section 4: Feature Engineering Function ---

def apply_feature_engineering(raw_trajectory_df, gdf_highways_geo, gdf_service_roads_geo):
    """
    Applies feature engineering to a raw GPS trajectory DataFrame.
    Includes kinematic and geometric features.
    """
    print("--- DEBUG FE: Starting apply_feature_engineering ---")

    # Ensure timestamp is datetime type
    if not pd.api.types.is_datetime64_any_dtype(raw_trajectory_df['timestamp']):
        raw_trajectory_df['timestamp'] = pd.to_datetime(raw_trajectory_df['timestamp'])

    # Sort by timestamp within each trajectory_id to ensure correct diff calculations
    raw_trajectory_df = raw_trajectory_df.sort_values(by=['trajectory_id', 'timestamp']).reset_index(drop=True)

    # Kinematic Features
    raw_trajectory_df['time_diff_seconds'] = raw_trajectory_df.groupby('trajectory_id')['timestamp'].diff().dt.total_seconds().fillna(0)
    raw_trajectory_df['speed_mps'] = raw_trajectory_df['speed_kph'] * 1000 / 3600
    
    # Handle division by zero for acceleration where time_diff_seconds is 0
    raw_trajectory_df['acceleration_mps2'] = raw_trajectory_df.groupby('trajectory_id')['speed_mps'].diff().fillna(0) / raw_trajectory_df['time_diff_seconds']
    # Fix for FutureWarning:
    raw_trajectory_df['acceleration_mps2'] = raw_trajectory_df['acceleration_mps2'].replace([np.inf, -np.inf], 0).fillna(0)

    raw_trajectory_df['heading_diff'] = raw_trajectory_df.groupby('trajectory_id')['heading'].diff().fillna(0)
    raw_trajectory_df['heading_diff'] = raw_trajectory_df['heading_diff'].apply(lambda x: (x + 180) % 360 - 180)
    
    # Handle division by zero for angular_velocity_deg_per_sec
    raw_trajectory_df['angular_velocity_deg_per_sec'] = raw_trajectory_df['heading_diff'] / raw_trajectory_df['time_diff_seconds']
    # Fix for FutureWarning:
    raw_trajectory_df['angular_velocity_deg_per_sec'] = raw_trajectory_df['angular_velocity_deg_per_sec'].replace([np.inf, -np.inf], 0).fillna(0)
    print("--- DEBUG FE: Kinematic features calculated ---")


    # Convert to GeoDataFrame for spatial operations (initial CRS is geographic)
    gdf_points = gpd.GeoDataFrame(
        raw_trajectory_df,
        geometry=gpd.points_from_xy(raw_trajectory_df.longitude, raw_trajectory_df.latitude),
        crs="EPSG:4326"
    )
    print("--- DEBUG FE: Trajectory converted to GeoDataFrame ---")

    # --- IMPORTANT FIX: Reproject to a PROJECED CRS (meters) for accurate distance calculations ---
    # We will work with projected CRSs for sjoin_nearest and distance calculations
    # EPSG:3857 (Web Mercator) is a good general-purpose choice.
    print("--- DEBUG FE: Reprojecting GeoDataFrames to EPSG:3857 (meters) ---")
    gdf_points_proj = gdf_points.to_crs(epsg=3857)
    
    # Ensure the highway and service road GDFs are also reprojected
    # Use the local variable names here (gdf_highways_geo, gdf_service_roads_geo)
    gdf_highways_proj = gdf_highways_geo.to_crs(epsg=3857)
    gdf_service_roads_proj = gdf_service_roads_geo.to_crs(epsg=3857)


    # 3. Distance to Nearest Highway
    print("--- DEBUG FE: Starting highway distance calculation ---")
    max_distance_meters = 250 # Max distance in METERS (e.g., within 250m)

    # Only proceed if there are valid highway segments
    if not gdf_highways_proj.empty and not gdf_points_proj.empty:
        sjoined_highway = gpd.sjoin_nearest(
            gdf_points_proj, # Use projected points
            gdf_highways_proj, # Use projected highways
            how="left",
            max_distance=max_distance_meters, # Now correctly interpreted as meters
            distance_col="dist_to_nearest_highway_temp" # Distance column created by sjoin_nearest
        )
        print("--- DEBUG FE: sjoin_nearest (highway) completed ---")
        
        # Map the minimum distance back to the original gdf_points (indexed correctly)
        # Use .groupby() and .min() to get the nearest distance for each original point
        min_distances_highway = sjoined_highway.groupby(sjoined_highway.index)['dist_to_nearest_highway_temp'].min()
        gdf_points['dist_to_nearest_highway'] = gdf_points.index.map(min_distances_highway)
    else:
        print("--- DEBUG FE: Skipping highway distance calculation: gdf_highways_proj or gdf_points_proj is empty.")
        gdf_points['dist_to_nearest_highway'] = np.nan # Initialize with NaN if no matches

    # Fix for FutureWarning: Use direct assignment instead of inplace=True
    gdf_points['dist_to_nearest_highway'] = gdf_points['dist_to_nearest_highway'].fillna(9999) # Fill NaN with a large number
    print("--- DEBUG FE: Highway distance calculation complete ---")


    # 4. Distance to Nearest Service Road
    print("--- DEBUG FE: Starting service road distance calculation ---")
    if not gdf_service_roads_proj.empty and not gdf_points_proj.empty:
        sjoined_service = gpd.sjoin_nearest(
            gdf_points_proj, # Use projected points
            gdf_service_roads_proj, # Use projected service roads
            how="left",
            max_distance=max_distance_meters, # Now correctly interpreted as meters
            distance_col="dist_to_nearest_service_road_temp"
        )
        print("--- DEBUG FE: sjoin_nearest (service road) completed ---")

        min_distances_service = sjoined_service.groupby(sjoined_service.index)['dist_to_nearest_service_road_temp'].min()
        gdf_points['dist_to_nearest_service_road'] = gdf_points.index.map(min_distances_service)
    else:
        print("--- DEBUG FE: Skipping service road distance calculation: gdf_service_roads_proj or gdf_points_proj is empty.")
        gdf_points['dist_to_nearest_service_road'] = np.nan # Initialize with NaN if no matches

    # Fix for FutureWarning: Use direct assignment instead of inplace=True
    gdf_points['dist_to_nearest_service_road'] = gdf_points['dist_to_nearest_service_road'].fillna(9999) # Fill NaN with a large number
    print("--- DEBUG FE: Service road distance calculation complete ---")


    # 5. Ratio of Distances (for classification context)
    print("--- DEBUG FE: Ratio feature calculated ---")
    # Add small epsilon to prevent division by zero, then calculate ratio
    gdf_points['dist_ratio_service_highway'] = (
        gdf_points['dist_to_nearest_service_road'] /
        (gdf_points['dist_to_nearest_highway'] + 1e-6) # Add small epsilon
    )
    # Fix for FutureWarning: Use direct assignment instead of inplace=True
    gdf_points['dist_ratio_service_highway'] = gdf_points['dist_ratio_service_highway'].replace([np.inf, -np.inf], 1e6)


    # Select final features (ensure 'road_type_label' is present for simulated data)
    final_features_df = gdf_points[[
        'trajectory_id', 'timestamp', 'latitude', 'longitude', 'speed_kph', 'heading',
        'acceleration_mps2', 'angular_velocity_deg_per_sec',
        'dist_to_nearest_highway', 'dist_to_nearest_service_road', 'dist_ratio_service_highway',
        'road_type_label' # Keep ground truth for comparison if available
    ]].copy() # .copy() to avoid SettingWithCopyWarning later

    print("--- DEBUG FE: Returning from apply_feature_engineering ---")
    return final_features_df

# --- Section 5: Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Road Type Classifier & Map-Matching")

st.title("🛣️ Road Type Classifier & Map-Matching Demo")
st.write("Upload a CSV with GPS trajectory data or use simulated data to classify road types (Highway/Service Road) using a trained ML model.")

# Load map data and model globally (cached)
# --- app.py (around line 332) ---
gdf_highways_geo, gdf_service_roads_geo = load_map_data(place_name)
model = load_ml_model(model_path)

# Initialize session state for trajectory data
if 'raw_trajectory_df' not in st.session_state:
    st.session_state.raw_trajectory_df = None
if 'processed_trajectory_gdf' not in st.session_state:
    st.session_state.processed_trajectory_gdf = None

# --- Sidebar for Data Input ---
st.sidebar.header("Input GPS Trajectory Data")

data_source = st.sidebar.radio("Choose Data Source:", ("Simulate New Trajectory", "Upload CSV"))

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = ['timestamp', 'latitude', 'longitude', 'speed_kph', 'heading']
            if all(col in df.columns for col in required_cols):
                st.session_state.raw_trajectory_df = df.copy()
                st.sidebar.success("CSV uploaded successfully!")
                st.sidebar.dataframe(st.session_state.raw_trajectory_df.head())
                st.session_state.processed_trajectory_gdf = None # Reset processed data on new upload
            else:
                st.sidebar.error(f"Missing required columns. Please ensure CSV has: {', '.join(required_cols)}")
                st.session_state.raw_trajectory_df = None
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            st.session_state.raw_trajectory_df = None

# --- Your app.py (inside your data_source logic, for "Simulate New Trajectory") ---

elif data_source == "Simulate New Trajectory":
    st.sidebar.subheader("Simulate Trajectory Parameters")
    road_type_choice = st.sidebar.selectbox("Simulate on:", ("Highway", "Service Road"))
    num_points = st.sidebar.slider("Number of simulated points:", min_value=50, max_value=500, value=200)
    noise_meters = st.sidebar.slider("GPS Noise (meters):", min_value=1, max_value=50, value=10)

    if st.sidebar.button("Generate Simulated Trajectory"):
        # Check if the graph is loaded before attempting simulation
        if 'G_all' in st.session_state and st.session_state.G_all:
            # The 'create_simulated_trajectory' function now uses st.session_state.G_all directly.
            # So, 'selected_gdf' is no longer needed.
            # Also, 'label' was converting to lowercase, but the function expects "Highway" or "Service Road".

            sim_df = create_simulated_trajectory(num_points, noise_meters, road_type_choice, st.session_state.G_all)

            if not sim_df.empty:
                sim_df['trajectory_id'] = f"simulated_traj_{datetime.now().strftime('%H%M%S')}"
                st.session_state.raw_trajectory_df = sim_df
                st.session_state.processed_trajectory_gdf = None # Reset processed data on new simulation
                st.sidebar.success(f"Simulated trajectory generated on a {road_type_choice}!") # Use road_type_choice for message
                st.sidebar.dataframe(st.session_state.raw_trajectory_df.head())
            else:
                # This warning means the simulation function tried multiple times but couldn't find a segment
                st.sidebar.warning(f"Could not generate simulated trajectory for '{road_type_choice}'. Map data might be too sparse or road types not commonly tagged.")
        else:
            # This handles the case where the map graph itself hasn't been loaded yet
            st.sidebar.error("Map data (OSMnx graph) not loaded. Please ensure place name is entered and map data loads successfully.")


# --- Main Content Area for Map-Matching and Visualization ---
st.header("Map-Matched Trajectory Visualization")

if st.session_state.raw_trajectory_df is not None and not st.session_state.raw_trajectory_df.empty and model is not None:
    if st.button("Perform Map-Matching"):
        print("--- Terminal Debug: Button clicked, entering map-matching block ---") # DEBUG PRINT
        st.info("Map-Matching: Button clicked. Starting processing...")


    with st.spinner("Processing trajectory and making predictions..."): 
        
        # NOTE: st.spinner is commented out for debugging purposes
        # If the prints below work, uncomment st.spinner for a better UX
        # with st.spinner("Processing trajectory and making predictions..."):
        
        print("--- Terminal Debug: About to call apply_feature_engineering ---") # DEBUG PRINT
        # Pass gdf_highways_geo and gdf_service_roads_geo directly
        st.session_state.processed_trajectory_gdf = apply_feature_engineering(
            st.session_state.raw_trajectory_df, gdf_highways_geo, gdf_service_roads_geo
        )
        print("--- Terminal Debug: Returned from apply_feature_engineering ---") # DEBUG PRINT

        # Check if feature engineering returned a valid DataFrame
        if st.session_state.processed_trajectory_gdf is None or st.session_state.processed_trajectory_gdf.empty:
            st.error("Feature engineering failed or returned empty data. Cannot proceed with prediction.")
            print("--- Terminal Debug: Feature engineering result is empty/None ---") # DEBUG PRINT
        else:
            st.info("Map-Matching: Feature engineering completed. Proceeding to prediction.")
            print("--- Terminal Debug: Starting prediction ---") # DEBUG PRINT
            
            features_for_prediction = [
                'speed_kph',
                'heading',
                'acceleration_mps2',
                'angular_velocity_deg_per_sec',
                'dist_to_nearest_highway',
                'dist_to_nearest_service_road',
                'dist_ratio_service_highway'
                # 'heading' is also a feature but can be more complex to use directly with RF sometimes
                # Removed 'heading' from prediction features for initial simplicity if needed, but it's okay to keep generally.
                # If your model was trained with 'heading', make sure to include it here.
            ]

            # Ensure all required features are present
            missing_features = [f for f in features_for_prediction if f not in st.session_state.processed_trajectory_gdf.columns]
            if missing_features:
                st.error(f"Missing features for prediction: {', '.join(missing_features)}. Please check feature engineering.")
                print(f"--- Terminal Debug: Missing features for prediction: {missing_features} ---") # DEBUG PRINT
            else:
                # Make prediction
                X_predict = st.session_state.processed_trajectory_gdf[features_for_prediction]
                
                # Check for NaNs/Infs in X_predict before prediction
                if X_predict.isnull().values.any() or np.isinf(X_predict.values).any():
                    st.warning("Warning: NaN or Inf values found in prediction features. Model might produce unreliable results.")
                    print("--- Terminal Debug: NaN/Inf values in X_predict ---") # DEBUG PRINT
                    # You might want to fill these or drop rows, depending on your strategy
                    X_predict = X_predict.fillna(0).replace([np.inf, -np.inf], 0) # Simple handling for demo

                try:
                    predicted_road_types = model.predict(X_predict)
                    st.session_state.processed_trajectory_gdf['predicted_road_type'] = predicted_road_types
                    st.success("Prediction completed!")
                    print("--- Terminal Debug: Prediction completed ---") # DEBUG PRINT

                    # --- Visualization ---
                    st.subheader("Results Overview")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Points Processed", len(st.session_state.processed_trajectory_gdf))
                    with col2:
                        if 'road_type_label' in st.session_state.processed_trajectory_gdf.columns:
                            # If simulated data with ground truth
                            accuracy = accuracy_score(st.session_state.processed_trajectory_gdf['road_type_label'], st.session_state.processed_trajectory_gdf['predicted_road_type'])
                            st.metric("Prediction Accuracy (Simulated Data)", f"{accuracy:.2%}")
                            st.write("Ground Truth Counts:")
                            st.dataframe(st.session_state.processed_trajectory_gdf['road_type_label'].value_counts())
                        st.write("Predicted Counts:")
                        st.dataframe(st.session_state.processed_trajectory_gdf['predicted_road_type'].value_counts())
                    
                    st.subheader("Predicted Trajectory on Map")

                    # Create plot
                    fig, ax = plt.subplots(figsize=(12, 12))

                    # Plot highways and service roads from your map data
                    gdf_highways_geo.plot(ax=ax, color='lightgray', linewidth=2, label='Highways (Map)')
                    gdf_service_roads_geo.plot(ax=ax, color='darkgray', linewidth=1, label='Service Roads (Map)')

                    # Plot the predicted trajectory
                    # Color points based on predicted_road_type
                    colors = {'highway': 'blue', 'service_road': 'red'}
                    for road_type, color in colors.items():
                        subset = st.session_state.processed_trajectory_gdf[st.session_state.processed_trajectory_gdf['predicted_road_type'] == road_type]
                        if not subset.empty:
                            ax.plot(subset['longitude'], subset['latitude'], marker='o', linestyle='-', markersize=5,
                                    color=color, alpha=0.7, label=f'Predicted {road_type.capitalize()}')
                    
                    # --- app.py (around line 439, just before ax.set_title(...) ) ---

                    # ... (previous code before ax.set_title) ...

                    # Add the simulated raw trajectory (before map-matching) for context
                    # Use a distinct color, e.g., purple or yellow
                    if st.session_state.raw_trajectory_df is not None and not st.session_state.raw_trajectory_df.empty:
                        ax.plot(st.session_state.raw_trajectory_df['longitude'],
                                st.session_state.raw_trajectory_df['latitude'],
                                marker='x', linestyle=':', markersize=4,
                                color='purple', alpha=0.8, label='Raw Simulated Trajectory')
                        print("--- Terminal Debug: Raw simulated trajectory plotted ---") # NEW DEBUG PRINT


                    ax.set_title("Predicted Road Types for Trajectory")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    
                    # Adjust legend placement if it's overlapping
                    # ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1)) # Example adjustment if needed
                    ax.legend(loc='lower left', bbox_to_anchor=(1, 0)) # Keep previous, but observe if it overlaps
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    print("--- Terminal Debug: Map visualization rendered ---")

                    # --- ADD THESE NEW DEBUG DATAFRAMES BELOW THE MAP ---
                    st.subheader("Debug: Feature Engineered Data Head")
                    st.dataframe(st.session_state.processed_trajectory_gdf.head())

                    st.subheader("Debug: Distance Feature Statistics")
                    if 'dist_to_nearest_highway' in st.session_state.processed_trajectory_gdf.columns:
                        st.write("Highway Distance Stats:")
                        st.dataframe(st.session_state.processed_trajectory_gdf['dist_to_nearest_highway'].describe())
                    if 'dist_to_nearest_service_road' in st.session_state.processed_trajectory_gdf.columns:
                        st.write("Service Road Distance Stats:")
                        st.dataframe(st.session_state.processed_trajectory_gdf['dist_to_nearest_service_road'].describe())
                    # --- END OF NEW DEBUG DATAFRAMES ---

                except Exception as e:
                    st.error(f"Error during prediction or visualization: {e}")
                    print(f"--- Terminal Debug: Error during prediction/visualization: {e} ---")
                    import traceback
                    print(traceback.format_exc())

                    ax.set_title("Predicted Road Types for Trajectory")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
                    plt.tight_layout()
                    st.pyplot(fig)
                    print("--- Terminal Debug: Map visualization rendered ---") # DEBUG PRINT

                except Exception as e:
                    st.error(f"Error during prediction or visualization: {e}")
                    print(f"--- Terminal Debug: Error during prediction/visualization: {e} ---") # DEBUG PRINT
                    import traceback
                    print(traceback.format_exc()) # Print full traceback to terminal

else:
    st.info("Upload a CSV or generate a simulated trajectory from the sidebar to visualize map-matching.")

st.markdown("""
---
**Note:**
* **Place Name:** The map data loaded is based on the place name defined in the script (`Deoria, Uttar Pradesh, India`). Ensure your uploaded/simulated data falls within this area.
* **CSV Format:** Ensure your uploaded CSV has `timestamp`, `latitude`, `longitude`, `speed_kph`, `heading` columns.
* **Performance:** Loading map data and the model happens only once. Feature engineering and prediction run when you click "Perform Map-Matching".
""")
