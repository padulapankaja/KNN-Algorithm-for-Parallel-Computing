# KNN-PC-OPENMP
Parallel computing is the practice of solving any problem by utilizing several processing elements concurrently. Tasks are divided into instructions and solved simultaneously, as each resource applied to the task is utilized concurrently.

Currently parallel computing become the trendy topic because of its usage so parallel computing need to because of
- The entire real world is dynamic in nature, which means that numerous things occur at the same moment but at various locations concurrently. This data is quite difficult to manage.
- Real-world data requires more dynamic simulation and modeling, and parallel processing is required to accomplish this.
- Parallel computing enables concurrency and enables significant time and cost savings
- Parallel computing's method is the only way to arrange complex, huge datasets and their maintenance.
- Assures efficient resource utilization. The hardware is guaranteed to be used efficiently, whereas with serial computation, just a portion of the hardware was utilized, and the remainder was left idle.
- Additionally, serial processing makes it impractical to construct real-time systems.

For the assignment I selected the K-Nearest Neighbors algorithm (KNN). When consider the K-Nearest Neighbors algorithm it is one of the simplest Machine Learning algorithms accessible. It is based on the Supervised Learning technique and assumes similarity between the new case/data and existing cases. It then assigns the new case to the category that is most like the existing categories.

Moreover, the K-NN algorithm saves all available data and classifies a new data point based on its resemblance to an existing data point. This means that when fresh data is generated, it may be quickly classified into a suitable category using the K-NN technique. This algorithm can be used for both regression and classification but is most usually utilized for classification problems. K-NN is a non-parametric algorithm, which means that no assumptions about the underlying data are required.

Additionally, it is sometimes referred to as a lazy learner algorithm since it does not instantly learn from the training set but instead saves the dataset and takes an action on it during classification. During in the training phase, the KNN algorithm simply saves the dataset and then classifies fresh data into a category that is highly comparable to the new data.

As the dataset to I selected Iris Dataset. This is possibly the most well-known database in the field of pattern recognition. Fisher's study is considered a classic in the subject and is widely cited to this day. (For instance, see Duda & Hart.) The data set has three classes with a total of 50 instances each, each class corresponding to a different species of iris plant. One class is linearly separable from the other two; the latter are not. Predicted attribute: iris plant class. This is a really straightforward domain.


As the steps of the KNN Algorithm,

1.Load the data
2.Initialize K to your chosen number of neighbors
3. For each example in the data
3.1 Calculate the distance between the query example and the current example from the data. 3.2 Add the distance and the index of the example to an ordered collection
4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
5. Pick the first K entries from the sorted collection
6. Get the labels of the selected K entries
7. If regression, return the mean of the K labels 8. If classification, return the mode of the K labels

Connect with me :- Linkedin[https://www.linkedin.com/in/padula-guruge/]
