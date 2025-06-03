import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from rgb_voxel.voxel_indexer import RGBVoxelIndexer

st.set_page_config(page_title="RGB Voxel Explorer", layout="centered")
st.title("RGB Voxel Explorer")

# Load voxel indexer
indexer = RGBVoxelIndexer('classified_dress_colors.csv')

# Sliders for user to pick RGB query
r = st.slider("Red", 0, 255, 128)
g = st.slider("Green", 0, 255, 128)
b = st.slider("Blue", 0, 255, 128)

# Get cube index
voxel_index = indexer._get_voxel(r, g, b)
st.markdown(f"🧊 **Current Voxel Cube Index:** `{voxel_index}`")

# Average color in full-width
voxel_colors = indexer.voxel_map.get(voxel_index, [])
st.markdown("**Average Color in Cube:**")
if voxel_colors:
    avg_color = tuple(int(np.mean(c)) for c in zip(*voxel_colors))
    st.markdown(f"**RGB:** `{avg_color}`")
    avg_rgb_hex = f'rgb({avg_color[0]}, {avg_color[1]}, {avg_color[2]})'
    st.markdown(
        f'<div style="width:100px; height:40px; background-color:{avg_rgb_hex}; border:1px solid #000;"></div>',
        unsafe_allow_html=True
    )
else:
    st.markdown("⚠️ No colors in this cube.")

# Side-by-side voxel neighborhood preview
st.markdown("**Voxel Neighborhood Preview**")
preview_col1, preview_col2 = st.columns(2)

# First half of the preview grid
with preview_col1:
    grid_html_1 = "<div style='display:grid; grid-template-columns: repeat(5, 40px); gap:4px;'>"
    for x in range(5):
        for y in range(5):
            for z in range(0, 3):  # First half of z-axis
                cube = (x, y, z)
                has_data = cube in indexer.voxel_map
                is_center = cube == voxel_index
                is_neighbor = (
                    abs(cube[0] - voxel_index[0]) <= 1 and
                    abs(cube[1] - voxel_index[1]) <= 1 and
                    abs(cube[2] - voxel_index[2]) <= 1
                )

                if is_center:
                    color = '#ffffff'
                elif is_neighbor:
                    color = '#cccccc'
                elif has_data:
                    color = '#8888ff'
                else:
                    color = '#333333'

                grid_html_1 += f"<div style='width:40px;height:40px;background:{color};'></div>"
    grid_html_1 += "</div>"
    st.markdown(grid_html_1, unsafe_allow_html=True)

# Second half of the preview grid
with preview_col2:
    grid_html_2 = "<div style='display:grid; grid-template-columns: repeat(5, 40px); gap:4px;'>"
    for x in range(5):
        for y in range(5):
            for z in range(3, 5):  # Second half of z-axis
                cube = (x, y, z)
                has_data = cube in indexer.voxel_map
                is_center = cube == voxel_index
                is_neighbor = (
                    abs(cube[0] - voxel_index[0]) <= 1 and
                    abs(cube[1] - voxel_index[1]) <= 1 and
                    abs(cube[2] - voxel_index[2]) <= 1
                )

                if is_center:
                    color = '#ffffff'
                elif is_neighbor:
                    color = '#cccccc'
                elif has_data:
                    color = '#8888ff'
                else:
                    color = '#333333'

                grid_html_2 += f"<div style='width:40px;height:40px;background:{color};'></div>"
    grid_html_2 += "</div>"
    st.markdown(grid_html_2, unsafe_allow_html=True)

# --- Optional Cube Zoom Interaction ---

st.markdown(" **Explore Another Voxel Cube**")

available_cubes = sorted(indexer.voxel_map.keys())
selected_cube = st.selectbox("Select a voxel cube to explore:", available_cubes, index=available_cubes.index(voxel_index))

# Show details for selected cube
selected_colors = indexer.voxel_map.get(selected_cube, [])

st.markdown(f"**Selected Cube Index:** `{selected_cube}`")
if selected_colors:
    avg_selected_color = tuple(int(np.mean(c)) for c in zip(*selected_colors))
    st.markdown(f"Average Color: RGB{avg_selected_color}")
    avg_hex = f'rgb({avg_selected_color[0]}, {avg_selected_color[1]}, {avg_selected_color[2]})'
    st.markdown(
        f'<div style="width:100px; height:40px; background-color:{avg_hex}; border:1px solid #000;"></div>',
        unsafe_allow_html=True
    )

    # Show color points as a mini 3D scatter
    selected_df = pd.DataFrame(selected_colors, columns=['R', 'G', 'B'])
    selected_df['Color'] = selected_df.apply(lambda row: f'rgb({int(row.R)},{int(row.G)},{int(row.B)})', axis=1)
    fig_zoom = px.scatter_3d(selected_df, x='R', y='G', z='B', color='Color', opacity=0.8)
    st.plotly_chart(fig_zoom)
else:
    st.markdown("⚠️ No color points in this cube.")

# Find closest neighbors
neighbors = indexer.find_neighbors(r, g, b, k=10)

# Prepare DataFrame for 3D plot
df = pd.DataFrame(neighbors, columns=['R', 'G', 'B'])
df['Color'] = df.apply(lambda row: f'rgb({int(row.R)},{int(row.G)},{int(row.B)})', axis=1)
df['hover_label'] = df.apply(lambda row: f"Match<br>RGB: ({int(row.R)}, {int(row.G)}, {int(row.B)})", axis=1)

# Add query point
query_point = pd.DataFrame([{
    'R': r,
    'G': g,
    'B': b,
    'Color': 'Query',
    'hover_label': f"Query Point<br>RGB: ({r}, {g}, {b})"
}])

# Combine all for plotting
df_plot = pd.concat([df, query_point], ignore_index=True)

# Create the 3D plot
fig = px.scatter_3d(
    df_plot,
    x='R', y='G', z='B',
    color='Color',
    hover_name='hover_label',
    opacity=0.8,
    color_discrete_map={'Query': '#FFFFFF'}  # white for the query point
)

fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
st.plotly_chart(fig)