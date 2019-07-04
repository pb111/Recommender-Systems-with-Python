# Recommender Systems with Python


Recommender Systems are one of the most popular and widely used application of data science. In this project, I build a Recommender System with Python. I discuss various types of recommender systems including - `Content-based` and `Collaborative filtering` recommender systems. Also, I discuss `matrix factorization` and how to evaluate recommender systems.

===============================================================================

## Table of Contents

1.	Introduction to Recommender Systems
2.	Recommender Systems mechanism
    -	Data collection
    -	Data storage
    -	Filtering the data
3.	Collaborative filtering recommender system
4.	Content-based filtering recommender system
5.	Multi-criteria recommender systems
6.	Risk aware recommender systems
7.	Mobile recommender systems
8.	Hybrid recommender systems
9.	Introduction to matrix factorization
10.	Evaluating recommender systems
11.	Applications of recommender systems
12.	References

===============================================================================


## 1. Introduction to Recommender Systems

-	Recommender systems are one of the most popular data science applications today.
-	A recommender system is a data science application that is used to predict or offer products to customers based on their past purchase or browsing history.
-	At the core, a recommender system employs a machine learning algorithm whose job is to predict user's ratings for a particular entity.
-	It is based on the similarity based on the entities or users that previously rated those entities.
-	The idea is that similar types of users are likely to have similar ratings for a set of entities.
-	Recommender systems have wide variety of applications.
-	Many of the big technology companies use a recommender system in some form to recommend products to customers.
-	They are used by Amazon for product recommendations, YouTube for video recommendations, Netflix and IMDB for movie recommendations and Facebook for friend recommendations.
-	The ability to recommend relevant products or services to users can be very profitable for a company.  Hence, it is so common to find this application by many companies.

===============================================================================

## 2. Recommender Systems mechanism

-	In this section, I will focus on recommender systems mechanism, i.e. how a recommender systems work.
-	Basically, a recommendation engine filters the data using different algorithms and recommends the most relevant items to users.
-	It first studies the past behaviour of a customer and based on that recommends products which he might buy.
-	The working of recommender systems is shown in the following diagram-

! [Recommender System Mechanism](https://github.com/pb111/Recommender-Systems-with-Python/blob/master/Images/Recommender%20system.jpg)

-	Now, we can recommend products to users in different ways.
-	We can recommend items to a user which are most popular among all the users.
-	We can divide the users into multiple segments and based on their preferences we recommend items to them.
-	The working of a recommendation engine can be categorized in three steps-
1.	Data collection
2.	Data storage
3.	Filtering the data


-	These steps are explained below:-

### 1. Data collection

-	The first step in building a recommendation engine is data collection.
-	There are two forms of data collection techniques employed in recommender systems.
-	These are **explicit** and **implicit** forms of data collection.
-	**Explicit data** is information that is provided intentionally, i.e. input from the users such as movie ratings. 
-	**Implicit data** is information that is not provided intentionally but gathered from available data streams like search history, clicks, order history, etc.


**Examples of explicit data collection include the following:**
-	Asking a user to rate an item on a sliding scale.
-	Asking a user to search.
-	Asking a user to rank a collection of items from favorite to least favorite.
-	Presenting two items to a user and asking him/her to choose the better one of them.
-	Asking a user to create a list of items that he/she likes.


**Examples of implicit data collection include the following:**
-	Observing the items that a user views in an online store.
-	Analyzing item/user viewing times. 
-	Keeping a record of the items that a user purchases online.
-	Obtaining a list of items that a user has listened to or watched on his/her computer.
-	Analyzing the user's social network and discovering similar likes and dislikes.


### 2. Data storage

-	The second step in building a recommendation engine is data storage.
-	The amount of data storage dictates how good the recommendations of the model are.
-	For example, in a movie recommendation system, the more ratings users give to movies, the better the recommendations get for other users. 
-	The type of data plays an important role in deciding the type of storage that has to be used. 
-	This type of storage could include a standard SQL database, a NoSQL database or some kind of object storage.


### 3. Filtering the data

-	The third and final step in building a recommendation engine is filter the data to extract relevant information required to make final recommendations.
-	There are two major approaches to filter the data to extract relevant information. These are as follows:- 
1.	Collaborative Filtering – based on similar users.
2.	Content-Based Filtering – based on product attributes.
-	The difference between the above two approaches are shown in the following diagram-

# D ! [Collaborative filtering vs Content-based filtering]


There are several other approaches for recommender systems used in practice. These are discussed in later sections. At first, I will discuss the above two approaches.

===============================================================================

## 3. Collaborative filtering recommender system

-	In the collaborative filtering recommender system, the behaviour of a group of users is used to make recommendations to other users.
-	In this case, the system don’t have any knowledge about the product. 
-	Collaborative filtering approach build a model from a user’s past behaviour (items previously purchased or selected and/or numerical ratings given to those items) as well as similar decisions made by other users. 
-	This model is then used to predict items (or ratings for items) that the user may have an interest in.
-	It recommends based on the user’s rating in the past.
-	These systems try to predict the user’s rating or preferences based on past rating or preferences of other users.
-	These filters do not require item metadata to make predictions.
-	There are two types of collaborative filtering recommender system. They are:-

1.	**User-based collaborative filtering**

-	In this method products are recommended to a user based on the fact that the products have been liked by users similar to the user. 

2.	**Item-based collaborative filtering**
-	This method identifies and predict similar items based on users’ previous ratings. 

-	Collaborative filtering methods are also classified as **memory-based** and **model-based**.
-	An example of memory-based approach is the user-based algorithm while that of model-based approach is Kernel-Mapping Recommender.
-	Collaborative filtering approaches often suffer from three problems - **cold start**, **scalability** and **sparsity**. These are discussed below:-

1.	**Cold start**
-	Cold start refers to a problem, when for a new user or item there is not enough data to make recommendations.

2.	**Scalability**
-	To make recommendations, we need to choose from millions of users and products. So scalability means a large amount of computation power is required to make recommendations.

3.	**Sparsity**
-	The number of items sold on e-commerce portals are extremely large. The most active users will only have rated a small subset of overall database. Thus, even the most popular items have very few ratings.
-	Most famous example of collaborative filtering is item-to-item collaborative filtering (people who buy x also buy y). This algorithm is popularized by Amazon recommender system.
-	Social network companies like Facebook, originally used collaborative filtering to recommend new friends and groups by examining the network of connections between a user and their friends.
-	The diagram below demonstrates collaborative filtering recommender systems.

# D ! [Collaborative filtering recommender system]

===============================================================================

## 4. Content-based filtering recommender system

-	Another common approach when designing recommender systems is **content-based filtering**. 
-	Content-based filtering methods are based on a description of the item and a profile of the user’s preferences.
-	These methods are best suited to situations where there is known data on an item (name, location, description, etc.), but not on the user. 
-	In content based filtering recommender system, the similarity between different products is calculated on the basis of the attributes of the products.
-	The system uses the knowledge of each product to recommend a new product.
-	Content-based filtering approaches utilize a series of discrete characteristics of an item in order to recommend additional items with similar properties.
-	For example, in a content based movie recommender system, the similarities between the movies is calculated on the basis of genres, the actors and the director.
-	The general idea behind these recommender systems is that if a person liked a particular item, then he will also like an item similar to it.
-	Content-based recommenders treat recommendation as a user-specific classification problem and learn a classifier for the user's likes and dislikes based on product features.

-	The diagram below demonstrates content-based filtering recommender systems.


# D ! [Content-based filtering recommender system]

===============================================================================

## 5. Multi-criteria recommender systems

-	Multi-criteria recommender systems (MCRS) can be defined as recommender systems that incorporate preference information upon multiple criteria. 
-	Instead of developing recommendation techniques based on a single criterion values, the overall preference of user u for the item i, these systems try to predict a rating for unexplored items of u by exploiting preference information on multiple criteria that affect this overall preference value. 
-	Researchers approach MCRS as a multi-criteria decision making (MCDM) problem, and apply MCDM methods and techniques to implement MCRS systems.

===============================================================================

## 6. Risk-aware recommender systems

-	The existing approaches to recommender systems focus on recommending the most relevant content to users using contextual information. 
-	They do not take into account the risk of disturbing the user with unwanted notifications. 
-	It is important to consider the risk of disturbing the user by pushing recommendations in certain circumstances. 
-	For example, during a professional meeting, early morning, or late at night. 
-	Therefore, the performance of the recommender system depends to an extent how much risk it has incorporated into the recommendation process. 
-	One option to manage this issue is DRARS, a system which models the context-aware recommendation as a bandit problem. 
-	This system combines a content-based technique and a contextual bandit algorithm.

===============================================================================

## 7. Mobile recommender systems

-	Mobile recommender systems make use of internet-accessing smart phones to offer personalized and context-sensitive recommendations. 
-	There are three factors that could affect the mobile recommender systems and the accuracy of prediction results: the context, the recommendation method and privacy.
-	Additionally, mobile recommender systems suffer from a transplantation problem – recommendations may not apply in all regions.
-	One example of a mobile recommender system are the approaches taken by companies such as Uber and Lyft to generate driving routes for taxi drivers in a city.
-	This system uses GPS data of the routes that taxi drivers take while working. It includes location (latitude and longitude), time stamps, and operational status (with or without passengers). 
-	It uses this data to recommend a list of pickup points along a route, with the goal of optimizing occupancy times and profits.

===============================================================================

## 8. Hybrid recommender systems

-	Most recommender systems now use a hybrid approach. 
-	It means to combine collaborative filtering, content-based filtering and other approaches.
-	Hybrid approaches can be implemented in several ways which are as follows: 
1.	By making content-based and collaborative-based predictions separately and then combining them. 
2.	By adding content-based capabilities to a collaborative-based approach (and vice versa) or 
3.	By combining the approaches into one model.

-	An example of hybrid recommender systems is Netflix website. 

-	The website makes recommendations by comparing the watching and searching habits of similar users (collaborative filtering) as well as by offering movies that share characteristics with films that a user has rated highly (content-based filtering).

Some hybridization techniques include:

•	**Weighted**: Combining the score of different recommendation components numerically.

•	**Switching**: Choosing among recommendation components and applying the selected one.

•	**Mixed**: Recommendations from different recommenders are presented together to give the recommendation.

•	**Feature Combination**: Features derived from different knowledge sources are combined together and given to a single recommendation algorithm.

•	**Feature Augmentation**: Computing a feature or set of features, which is then part of the input to the next technique.

•	**Cascade**: Recommenders are given strict priority, with the lower priority ones breaking ties in the scoring of the higher ones.

•	**Meta-level**: One recommendation technique is applied and produces some sort of model, which is then the input used by the next technique. 


===============================================================================

## 9. Introduction to Matrix Factorization

-	**Matrix factorization** is a class of collaborative filtering algorithms used in recommender systems.
-	**Matrix factorization** algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices.
-	It became widely known during the Netflix prize challenge due to its effectiveness as reported by Simon Funk in 2006.
-	The idea behind matrix factorization is to represent users and items in a lower dimensional latent space.
-	Matrix factorization can be demonstrated by the following diagram-

# D ! [Matrix Factorization]

-	Nowadays, different types of matrix factorization approaches are used in practice.
-	Most popular used matrix factorization approaches are discussed below.

### Funk SVD

-	**Funk SVD** is the original algorithm proposed by Simon Funk.
-	He factorized the user-item rating matrix as the product of two lower dimensional matrices – the first one has a row for each user while the second one has a column for each item.
-	The row or column associated to a specific user or item is referred to as **latent factors**.
-	Despite its name, in Funk SVD, no singular value decomposition is applied.


### SVD ++

-	FunkSVD is able to provide very good recommendation quality, its ability to use only explicit numerical ratings as user-items interactions constitutes a limitation. 
-	Modern day recommender systems should exploit all available interactions both explicit (e.g. numerical ratings) and implicit (e.g. likes, purchases, skipped, bookmarked). 
-	Keeping this in mind, SVD++ was designed to take into account implicit interactions as well. 
-	As compared to FunkSVD, SVD++ takes also into account user and item bias.


### Asymmetric SVD

-	Asymmetric SVD aims at combining the advantages of SVD++ while being a model based algorithm.
-	Therefore it is able to consider new users with a few ratings without needing to retrain the whole model. 

### Hybrid Matrix Factorization

-	In recent years many other matrix factorization models have been developed to exploit the ever increasing amount and variety of available interaction data and use cases. 
-	Hybrid matrix factorization algorithms are capable of merging explicit and implicit interactions or both content and collaborative data

===============================================================================

## 10. Evaluating recommender systems

-	Once we develop a recommender system, we want to evaluate them.
-	Evaluation is important in assessing the effectiveness of recommender systems.
-	To measure the effectiveness of recommender systems, three types of evaluations are available – **user studies**, **online evaluations (A/B tests)** and **offline evaluations**.
-	The commonly used metrics are the **mean squared error (MSE)** and **root mean squared error(RMSE)**.
-	The information retrieval metrics such as **precision** and **recall** are useful to assess the quality of a recommender system.
-	**Diversity**, **novelty** and **coverage** are also considered as important aspects in evaluation.
-	**User studies** are rather small scale judgement technique. A few dozens or hundreds of users are presented recommendations created by different recommendation approaches, and then the users’ judge which recommendations are the best.
-	In **A/B tests**, recommendations are shown to thousands of users of a real product, and the recommender system randomly picks at least two different recommendation approaches to generate recommendations. The effectiveness is measured with implicit measures of effectiveness such as conversion rate or click-through rate.
-	**Offline evaluations** are based on historic data. For example, a dataset that contains information about how users previously rated movies.
-	The effectiveness of recommendation system is then measured based on how well a recommendation approach can predict the users' ratings in the dataset.

### Beyond measures of accuracy
-	Research on recommender systems is focussed towards finding the most accurate recommendation algorithms. There are a number of factors that are also important. These are discussed below.

-	**Diversity** – Users tend to be more satisfied with recommendations when there is a higher intra-list diversity. For example, items from different artists.

-	**Recommender persistence** – Sometimes, it is more effective to re-show recommendations than showing new items. There are several reasons for this. Users may ignore items when they are shown for the first time. May be they do not inspect the recommendations initially.

-	**Privacy** – Recommender systems usually have to deal with privacy concerns because users have to reveal sensitive information. 

-	**User demographics** – User demographics may influence how satisfied users are with recommendations.


-	**Robustness** – When users can participate in the recommender system, the issue of fraud must be addressed.

-	**Serendipity** – Serendipity is a measure of how surprising the recommendations are. For instance, a recommender system that recommends milk to a customer in a grocery store might be perfectly accurate. But it is not a good recommendation because it is an obvious item for the customer to buy.

-	**Trust** – A recommender system is of little value for a user if the user does not trust the system. Trust can be built by a recommender system by explaining how it generates recommendations, and why it recommends an item.

-	**Labelling** – User satisfaction with recommendations may be influenced by the labelling of the recommendations.


===============================================================================

## 11. Applications of Recommender Systems

-	There are lots of applications of recommender systems in data science field.
-	The music and video companies like Netflix, Youtube and Spotify use them to generate music and video recommendations.
-	Amazon uses it for product recommendations.
-	Social media platforms such as Facebook and Twitter use them for friends and content recommendations.
-	Restaurants and hotels use recommender systems to generate food related recommendations.
-	Recommender systems found applications in research articles, financial services and life insurance.

===============================================================================

## 12. References

The work done in this project is inspired from the following courses and websites:-

1.	Udemy course – Building Recommender Systems with Machine Learning and AI

2.	https://en.wikipedia.org/wiki/Recommender_system

3.	https://www.datacamp.com/community/tutorials/recommender-systems-python

4.	https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/

5.	https://stackabuse.com/creating-a-simple-recommender-system-in-python-using-pandas/

6.	https://medium.com/coinmonks/recommendation-engine-python-401c080c583e

7.	https://towardsdatascience.com/how-to-build-a-simple-recommender-system-in-python-375093c3fb7d

8.	https://en.m.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)



