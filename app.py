from itertools import cycle, islice
import numpy as np
import sklearn
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# SideBar Content
ml_algorithm = st.sidebar.selectbox(f"Select Machine Learning Algorithm", ("Classification", "Regression", "Clustering"))
if ml_algorithm == "Regression":
    dataset = st.sidebar.selectbox("Select Dataset", ("Boston", "Diabetes"))
    algorithm = st.sidebar.selectbox(f"Select {ml_algorithm} Algorithm", ("Linear Regression", "KNN", "Random Forest"))
    if algorithm == "KNN":
        n_neighbors = st.sidebar.slider("K (Neighbors)", 10, 50, 25)
    elif algorithm == "Random Forest":
        max_depth = st.sidebar.slider("Max Depth", 2, 15)
    else:
        normalize = st.sidebar.checkbox("Use Normalization")
elif ml_algorithm == "Classification":
    dataset = st.sidebar.selectbox("Select Dataset", ("Breast Cancer", "Iris", "Wine"))
    algorithm = st.sidebar.selectbox(f"Select {ml_algorithm} Algorithm", ("Random Forest", "KNN", "SVM"))
    if algorithm == "KNN":
        attr = "K"
        param = st.sidebar.slider("K (Neighbors)", 1, 10)
    elif algorithm == "SVM":
        attr = "C"
        param = st.sidebar.slider("C Value", 0.01, 10.0)
    else:
        attr = "Max Depth"
        param = st.sidebar.slider("Max Depth", 2, 15)
else:
    dataset = st.sidebar.selectbox(f"Select Dataset for {ml_algorithm} Algorithm", ("Circles", "Moons", "Blobs"))

ok = st.sidebar.button("Visualize")

# MainPage Content

st.title("Machine Learning Algorithms Visualization")
st.markdown("<br><hr><br>", unsafe_allow_html=True)

if ok:
    if ml_algorithm == "Regression":
        if dataset == "Boston":
            df = datasets.load_boston()
        else:
            df = datasets.load_diabetes()
        if algorithm == "Linear Regression":
            model = LinearRegression(normalize=normalize)
        elif algorithm == "KNN":
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        else:
            model = RandomForestRegressor(max_depth=max_depth)

        X, y = df.data, df.target
        X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        body = """
        <h2><b>Model and Dataset Characteristics</b></h2>
        <font face='Monospace'>
        <table>
            <tr>
                <td>Dataset</td>
                <td>""" + dataset + """</td>
            </tr>
            <tr>
                <td>Regressor</td>
                <td>""" + algorithm + """</td>
            </tr>
            <tr>
                <td>Features Shape</td>
                <td>""" + str(X.shape) + """</td>
            </tr>
            <tr>
                <td># of Features</td>
                <td>""" + str(X.shape[1]) + """</td>
            </tr>
            <tr>
                <td>Feature Names</td>
                <td>""" + str(df.feature_names) + """</td>
            </tr>
            <tr>
                <td>Target Shape</td>
                <td>""" + str(y.shape) + """</td>
            </tr>
            <tr>
                <td># of Targets</td>
                <td>""" + str(1) + """</td>
            </tr>
            <tr>
                <td>Score</td>
                <td>""" + str(score) + """</td>
            </tr>
        </table>
        </font>
        """
        st.markdown(body, unsafe_allow_html=True)

        
        # TO DO

    elif ml_algorithm == "Classification":
        if dataset == "Iris":
            df = datasets.load_iris()
        elif dataset == "Wine":
            df = datasets.load_wine()
        else:
            df = datasets.load_breast_cancer()
        
        if algorithm == "KNN":
            model = KNeighborsClassifier(n_neighbors=param)
        elif algorithm == "SVM":
            model = SVC(C=param)
        else:
            model = RandomForestClassifier(max_depth=param)

        X, y = df.data, df.target
        X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        pca = PCA(2)
        X_projected = pca.fit_transform(X)
        X1 = X_projected[:, 0]
        X2 = X_projected[:, 1]
        st.write(f"\n\n ## *{dataset} Dataset* in Two Dimensions\n")
        fig = plt.figure()
        plt.scatter(X1, X2, c=y, alpha=0.8, cmap='viridis')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar()
        st.pyplot(fig)

        body = """
        <h2><b>Model and Dataset Characteristics</b></h2>
        <font face='Monospace'>
        <table>
            <tr>
                <td>Dataset</td>
                <td>""" + dataset + """</td>
            </tr>
            <tr>
                <td>Classifier</td>
                <td>""" + algorithm + """ (""" + attr + """ : """ + str(param) + """)</td>
            </tr>
            <tr>
                <td>Features Shape</td>
                <td>""" + str(X.shape) + """</td>
            </tr>
            <tr>
                <td># of Features</td>
                <td>""" + str(X.shape[1]) + """</td>
            </tr>
            <tr>
                <td>Feature Names</td>
                <td>""" + str(df.feature_names) + """</td>
            </tr>
            <tr>
                <td>Target Shape</td>
                <td>""" + str(y.shape) + """</td>
            </tr>
            <tr>
                <td># of Classes</td>
                <td>""" + str(len(np.unique(y))) + """</td>
            </tr>
            <tr>
                <td>Class Names</td>
                <td>""" + str(df.target_names) + """</td>
            </tr>
            <tr>
                <td>Accuracy Score</td>
                <td>""" + str(acc) + """</td>
            </tr>
        </table>
        </font>
        """
        st.markdown(body, unsafe_allow_html=True)


    else:
        if dataset == "Circles":
            df = datasets.make_circles(1500, factor=0.5, noise=0.05)
        elif dataset == "Moons":
            df = datasets.make_moons(1500, noise=0.05)
        else:
            df = datasets.make_blobs(1500, random_state=8)
        


        fig = plt.figure(figsize=(10, 10))
        X, y = df
        X = StandardScaler().fit_transform(X)

        n_clusters = 2
        if dataset == "Blobs":
            n_clusters = 3

        st.subheader("Various Clustering Algorithms Performance")

        kmeans = cluster.KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a']), int(max(y_pred) + 1))))
        colors = np.append(colors, ["#000000"])
        plt.subplot(2, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=colors[y_pred], s=10)
        plt.title("K Means")

        dbscan = cluster.DBSCAN(eps=.3)
        y_pred = dbscan.fit_predict(X)
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a']), int(max(y_pred) + 1))))
        colors = np.append(colors, ["#000000"])
        plt.subplot(2, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], c=colors[y_pred], s=10)
        plt.title("DBSCAN")

        xi = 0.05
        if dataset == "Circles":
            xi = 0.25
        optics = cluster.OPTICS(min_samples=20, xi=xi, min_cluster_size=0.1)
        y_pred = optics.fit_predict(X)
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a']), int(max(y_pred) + 1))))
        colors = np.append(colors, ["#000000"])
        plt.subplot(2, 2, 3)
        plt.scatter(X[:, 0], X[:, 1], c=colors[y_pred], s=10)
        plt.title("OPTICS")

        birchs = cluster.Birch(n_clusters=n_clusters)
        y_pred = birchs.fit_predict(X)
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a']), int(max(y_pred) + 1))))
        colors = np.append(colors, ["#000000"])
        plt.subplot(2, 2, 4)
        plt.scatter(X[:, 0], X[:, 1], c=colors[y_pred], s=10)
        plt.title("BIRCH")
        st.pyplot(fig)


        st.header("{} Dataset".format(dataset))
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(df[0][:, 0], df[0][:, 1])
        st.pyplot(fig)