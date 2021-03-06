"""Perform clustering on the bbox data."""
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


style.use("fivethirtyeight")


def draw_scatter_plot(bbox_with_img_data_df):
    """."""
    h = bbox_with_img_data_df["h"].values
    w = bbox_with_img_data_df["w"].values

    plt.scatter(w, h, s=30, color='b')
    # label the axes
    plt.xlabel('w')
    plt.ylabel('h')
    plt.show()
    # clear the figure
    plt.clf()


def draw_kmeans_knee_plot(X):
    """."""
    cost = []
    for i in range(1, 11):
        KM = KMeans(n_clusters=i, max_iter=500)
        KM.fit(X)
        # calculates squared error
        # for the clustered points
        cost.append(KM.inertia_)

    # plot the cost against K values
    plt.plot(range(1, 11), cost, color='g', linewidth='3')
    plt.xlabel("Value of K")
    plt.ylabel("Sqaured Error (Cost)")
    plt.show()
    # clear the plot
    plt.clf()


def draw_silhoutte_score_plot(X):
    """."""
    sil = []
    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, 
    # minimum number of clusters should be 2
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(X)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels, metric = 'euclidean'))

    plt.plot(sil)
    plt.xlabel("Value of k")
    plt.ylabel("Silhouette score")
    plt.show()
    # clear the plot
    plt.clf()



   