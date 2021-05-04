from flask import jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import io
import base64
import json
plt.switch_backend('agg')


def kmeans_algorithm(client_id, clients):
    # Create a dataframe with the users
    df_bc = pd.DataFrame(list(clients))
    df_bc_id = df_bc

    # Dropping ids
    df_bc = df_bc.drop(['_id', 'CLIENTNUM'], axis=1)

    # One hot encoding for Categorical variables
    df_bc_encoded_norm = pd.get_dummies(df_bc)

    # Apply KMeans
    kmeans = KMeans(3, max_iter=300)  # Create the model
    kmeans.fit(df_bc_encoded_norm)  # Apply the model

    KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300,
           tol=0.0001, precompute_distances='deprecated', verbose=0, random_state=None,
           copy_x=True, n_jobs='none', algorithm='auto')

    # The number of clusters created
    clusters = kmeans.predict(df_bc_encoded_norm)
    print(clusters)

    # Adding the clusters classification to the original dataset
    df_bc["Kmeans_Clusters"] = kmeans.labels_
    # To return the id of the client
    df_bc_id["Kmeans_Clusters"] = kmeans.labels_

    # Analysis of components to visualize the data in 2 dimentions
    pca = PCA(n_components=2)
    pca_clients = pca.fit_transform(df_bc_encoded_norm)
    pca_clients_df = pd.DataFrame(data=pca_clients, columns=[
                                  "Component_1", "Component_2"])
    pca_names_clients = pd.concat(
        [pca_clients_df, df_bc[["Kmeans_Clusters"]]], axis=1)
    # print(pca_names_clients)

    # Generate the figure **without using pyplot**.
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Component 1", fontsize=15)
    ax.set_ylabel("Component 2", fontsize=15)
    ax.set_title("Mains components", fontsize=15)
    color_theme = np.array(["blue", "red", "orange"])
    ax.scatter(x=pca_names_clients.Component_1, y=pca_names_clients.Component_2,
               c=color_theme[pca_names_clients.Kmeans_Clusters], s=1)

    # Save it to a temporary buffer.
    img_cluster = io.BytesIO()
    fig.savefig(img_cluster, format='png')
    fig_url_cluster = base64.b64encode(img_cluster.getvalue()).decode()

    def graphics_kmeans(value):
        img = io.BytesIO()

        plt.Figure(figsize=(8, 5))
        plot = sns.countplot(x=df_bc.Kmeans_Clusters, hue=value)
        for p in plot.patches:
            plot.annotate(p.get_height(), (p.get_x() +
                                           p.get_width()/2, p.get_height()+50))

        plt.savefig(img, format='png')
        plt.close()
        plot_url = base64.b64encode(img.getvalue()).decode()
        return plot_url

    # List to save the different analysis from the cluster
    list_images = []
    list_images.append(fig_url_cluster)  # The cluster segmentation
    list_images.append(graphics_kmeans(df_bc.Attrition_Flag))
    list_images.append(graphics_kmeans(df_bc.Gender))
    list_images.append(graphics_kmeans(df_bc.Card_Category))
    list_images.append(graphics_kmeans(df_bc.Total_Relationship_Count))
    list_images.append(graphics_kmeans(df_bc.Contacts_Count_12_mon))
    list_images.append(graphics_kmeans(df_bc.Income_Category))

    # Get the category of the user
    user = df_bc_id.loc[(df_bc_id['CLIENTNUM'] == client_id)]
    user_id = user['Kmeans_Clusters'].to_string().split()
    user_cluster = user_id[1]
    print(user_cluster)

    return render_template('kmeans.html', user_id={user_cluster}, imagesData={'image': list_images})
