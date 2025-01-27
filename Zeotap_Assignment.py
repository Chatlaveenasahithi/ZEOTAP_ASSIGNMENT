import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# Task 1: Exploratory Data Analysis (EDA)
customers = pd.read_csv("Customers (1).csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')
summary = data.describe()
missing_values = data.isnull().sum()
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Region', order=data['Region'].value_counts().index)
plt.title('Number of Customers by Region')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='Category', y='TotalValue', ci=None, estimator=np.sum)
plt.title('Total Sales by Product Category')
plt.show()

insights = [
    "1. Most customers are from specific regions, which can be targeted for marketing campaigns.",
    "2. Product categories with higher sales can be prioritized in inventory management.",
    "3. Some customers make repeated high-value purchases, representing valuable segments.",
    "4. Seasonal trends in transaction dates suggest optimizing stock accordingly.",
    "5. Average transaction values vary across regions, highlighting potential pricing strategies."
]

# Task 2: Lookalike Model
customer_data = customers.set_index('CustomerID').join(transactions.groupby('CustomerID').mean(), how='left')
encoded_data = customer_data[['Region', 'TotalValue']]
encoded_data['Region'] = encoded_data['Region'].astype('category').cat.codes
similarities = cosine_similarity(encoded_data)
recommendations = {}

for idx, customer in enumerate(encoded_data.index[:20]):
    similar_customers = np.argsort(-similarities[idx])[1:4]  
    recommendations[customer] = [(encoded_data.index[i], similarities[idx][i]) for i in similar_customers]

lookalike_df = pd.DataFrame({
    "CustomerID": recommendations.keys(),
    "Similar_Customers": [x for x in recommendations.values()]
})

# Task 3: Customer Segmentation/Clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_data)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_data)
data['Cluster'] = kmeans.labels_


db_index = davies_bouldin_score(scaled_data, kmeans.labels_)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='TotalValue', y='Price', hue='Cluster', palette='viridis')
plt.title('Customer Clusters')
plt.show()

clustering_results = {
    "Number of Clusters": 4,
    "Davies-Bouldin Index": db_index
}

lookalike_df.to_csv('FirstName_LastName_Lookalike.csv', index=False)
data[['CustomerID', 'Cluster']].to_csv('FirstName_LastName_Clustering.csv', index=False)
