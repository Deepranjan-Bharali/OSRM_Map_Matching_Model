FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y git cmake g++ libboost-all-dev lua5.3 \
    liblua5.3-dev libtbb-dev libstxxl-dev libxml2-dev \
    libbz2-dev libzip-dev libosmpbf-dev libprotobuf-dev \
    protobuf-compiler wget curl python3-pip python3

# Install OSRM-backend
RUN git clone https://github.com/Project-OSRM/osrm-backend.git && \
    cd osrm-backend && mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build .

# Download OSM data (sample region)
RUN mkdir -p /data
WORKDIR /data
RUN wget http://download.geofabrik.de/europe/monaco-latest.osm.pbf

# Preprocess map data
RUN /osrm-backend/build/osrm-extract -p /osrm-backend/profiles/car.lua monaco-latest.osm.pbf && \
    /osrm-backend/build/osrm-partition monaco-latest.osrm && \
    /osrm-backend/build/osrm-customize monaco-latest.osrm

# Expose HTTP port
EXPOSE 5000

# Start OSRM server
CMD ["/osrm-backend/build/osrm-routed", "--algorithm", "mld", "/data/monaco-latest.osrm"]
