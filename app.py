import streamlit as st
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go

# Hardcoded class map
CLASS_MAP = {
    0: 'sofa',
    1: 'dresser',
    2: 'chair',
    3: 'toilet',
    4: 'desk',
    5: 'night_stand',
    6: 'table',
    7: 'bed',
    8: 'monitor',
    9: 'bathtub'
}

# Load the trained model
@st.cache_resource
def load_pointnet_model():
    class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
        def __init__(self, num_features, l2reg=0.001):
            self.num_features = num_features
            self.l2reg = l2reg
            self.eye = tf.eye(num_features)

        def __call__(self, x):
            x = tf.reshape(x, (-1, self.num_features, self.num_features))
            xxt = tf.tensordot(x, x, axes=(2, 2))
            xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
            return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

        def get_config(self):
            return {"num_features": self.num_features, "l2reg": self.l2reg}

    model = load_model(
        "pointnet_model.keras",  # Replace with the path to your saved model
        custom_objects={"OrthogonalRegularizer": OrthogonalRegularizer},
    )
    return model


# Function to display a 3D mesh with Plotly
def plotly_3d_mesh(mesh):
    x, y, z = mesh.vertices.T
    i, j, k = mesh.faces.T
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k)])
    fig.update_layout(scene=dict(aspectmode="data"))
    return fig


# Function to display point cloud using Matplotlib
def display_point_cloud(points):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1)
    ax.set_title("Point Cloud")
    ax.set_axis_off()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img


# Streamlit App
st.title("Point Cloud Classification")

# File upload section
uploaded_file = st.file_uploader("Upload a 3D Mesh file (.off)", type=["off"])

if uploaded_file is not None:
    # Load and process the mesh file
    mesh = trimesh.load(uploaded_file, file_type="off")
    st.write("### 3D Mesh View")
    st.plotly_chart(plotly_3d_mesh(mesh), use_container_width=True)

    # Sample points from the mesh
    points = mesh.sample(2048)

    # Display the sampled point cloud
    st.write("### Sampled Point Cloud (Matplotlib)")
    point_cloud_image = display_point_cloud(points)
    st.image(point_cloud_image, caption="3D Point Cloud (Matplotlib)")

    # Predict the class of the point cloud
    if st.button("Predict Class"):
        model = load_pointnet_model()

        # Prepare input for the model
        points_batch = np.expand_dims(points, axis=0)  # Add batch dimension
        predictions = model.predict(points_batch)
        predicted_class = np.argmax(predictions, axis=1)

        # Map predicted class index to class name
        predicted_class_name = CLASS_MAP.get(predicted_class[0], "Unknown")

        st.write("### Predicted Class:")
        st.write(f"Class Index: {predicted_class[0]}")
        st.write(f"Class Name: {predicted_class_name}")
