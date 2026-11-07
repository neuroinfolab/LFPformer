import os

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


class PCAVisualizer:
    """
    A class for creating and saving interactive PCA visualizations of high-dimensional embeddings.

    This class uses scikit-learn for PCA computation and Plotly for generating
    interactive 2D or 3D scatter plots, which are saved as HTML files.
    """
    def __init__(self, label_json, output_path):
        """
        Initializes the PCAVisualizer.

        Args:
            label_json (dict): A dictionary mapping integer labels to string representations.
            output_path (str): The directory path where the output HTML files will be saved.
        """
        self.output_path = output_path
        self.label_json = label_json

    def create_pca(self, embeddings, labels, n_components, session, stage, color_map=None):
        """
        Performs PCA on embeddings and generates a 2D or 3D scatter plot.

        Args:
            embeddings (np.ndarray): The high-dimensional data to be visualized.
            labels (list or np.ndarray): The labels corresponding to each embedding.
            n_components (int): The number of principal components to compute (must be 2 or 3).
            session (str or int): The session identifier, used for naming the output file.
            stage (str): The data stage (e.g., 'train', 'test'), used for the plot title and filename.
            color_map (dict, optional): A dictionary mapping label names to colors. A default map is used if not provided.
        """
        # Convert integer labels to string labels if necessary.
        if len(str(labels[0])) == 1:
            labels = [self.label_json[str(label)] for label in labels]
        
        # Perform PCA to reduce dimensionality.
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Create a pandas DataFrame for easy plotting with Plotly.
        df = pd.DataFrame(reduced_embeddings, columns=[f"PC{i + 1}" for i in range(n_components)])
        df["Label"] = labels
        
        # Define a default color map if one is not provided.
        if not color_map:
            color_map = {
                "Cortex": "#f0553b", "CA1": "#ffa15a",
                "CA2": "#636efa", "CA3": "#00cc96", "DG": "#ab63fa",
                "Basal_Ganglia": "#19d3f3", "Suppl_Motor_Area": "#e763fa",
                "Primary_Motor_Cortex": "#f0b400"
            }

        # Create a complete color map for all unique labels present in the data.
        unique_labels = df["Label"].astype(str).unique()
        full_color_map = {
            label: next((color for key, color in color_map.items() if key.lower() in label.lower()), "#999999")
            for label in unique_labels
        }

        # Generate a 2D or 3D scatter plot based on the number of components.
        if n_components == 2:
            fig = px.scatter(
                df,
                x="PC1",
                y="PC2",
                color="Label",
                color_discrete_map=full_color_map,
                title=f"2D PCA of {stage} Embeddings (Session: {session})",
                labels={"Label": "Class Label"}
            )
        elif n_components == 3:
            fig = px.scatter_3d(
                df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="Label",
                color_discrete_map=full_color_map,
                title=f"3D PCA of {stage} Embeddings (Session: {session})",
                labels={"Label": "Class Label"},
            )
        else:
            raise ValueError("n_components must be 2 or 3 for visualization.")

        # Define the output file path and save the interactive plot as an HTML file.
        output_file_path = os.path.join(self.output_path,
                                        f'session_{session}_{stage}_pca_{n_components}d_visualization.html')
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        fig.write_html(output_file_path)
        print(f"PCA visualization for {stage} embeddings (Session: {session}) saved at {output_file_path}")

    def create_combined_pca(self, datasets, n_components, session, title="combined", color_map=None):
        """
        Combines embeddings from multiple datasets/stages and creates a single PCA plot.

        Args:
            datasets (dict): A dictionary where keys are stage names (e.g., 'train', 'val') and
                             values are dictionaries containing 'embeddings' and 'labels'.
            n_components (int): The number of principal components (2 or 3).
            session (str or int): The session identifier for file naming.
            title (str): A title for the plot and filename, defaults to "combined".
            color_map (dict, optional): A custom color map for the plot.
        """
        combined_embeddings = []
        combined_labels = []
        
        # Aggregate embeddings and labels from all provided datasets.
        for stage, data in datasets.items():
            combined_embeddings.append(data["embeddings"])
            # Create descriptive labels by prepending the stage name.
            if title == "combined":
                combined_labels.extend([f"{stage}_{self.label_json[str(label)[-1]]}"
                                        for label in data["labels"]])
            else:
                combined_labels.extend([f"{stage}_{label}" for label in data["labels"]])

        combined_embeddings = np.vstack(combined_embeddings)

        # Use the create_pca method to generate the plot for the combined data.
        self.create_pca(combined_embeddings, combined_labels, n_components, session, title, color_map)
