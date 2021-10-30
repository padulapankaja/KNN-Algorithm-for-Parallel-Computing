#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#define K 5

#define _MASTER 0
#define MAX_LINE_LEN 200

int myRank, numProc;


void mySort(double* topKdistances, double* topKClasses);

void mySortArray(double* arr, int len);

int mostFrequentLabel(double* classes);

void insertValueInSorted(double* topKdistances, double* topKClasses, double newScore, double newLabel);

double getEuclideanDistance(double* query, double* target, int cols);

double** create2DArray(const int rows, const int cols);

double** readData(const char* file_name, double** labels, int* rows, int* cols);






int main(int argc, char* argv[])
{

	// initializing the MPI environment
	MPI_Init(&argc, &argv);
	// obtain myRank of current process
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	// obtain number of processes launched
	MPI_Comm_size(MPI_COMM_WORLD, &numProc);

	const char* file_name = "trainnew_small.csv";
	int rows, cols;
	double* labels;


	double** data = readData(file_name, &labels, &rows, &cols);



	// compute that how many rows are there which cannot be divided equally among the processes
	int extras = rows % numProc;

	// rows that each process will be responsible for
	int rowsPerProc = rows / numProc;


	// start row of each process
	int myStart = myRank * rowsPerProc;

	// end row of each process
	int myEnd = rowsPerProc + myStart - 1;

	// master process will handle all extra rows
	if (myRank == _MASTER)
	{
		myEnd += extras;
	}
	else
	{

		// start of other processes will move ahead
		myStart += extras;
		myEnd += extras;
	}



	double start = MPI_Wtime();
	int countDiff = 0;


	// compute the predictions for current process 
	for (int i = myStart; i <= myEnd; i++)
	{
		double* predictions = (double*)malloc(rows * sizeof(double));
		double* topKdistances = (double*)malloc(K * sizeof(double));
		double* topKClasses = (double*)malloc(K * sizeof(double));
		int curr_K = 0;

		// compare current row with all other rows
		for (int j = 0; j < rows; j++)
		{
			if (i != j)
			{

				// compute the euclidean distance
				double distance = getEuclideanDistance(&data[i][0], &data[j][0], cols);


				// if K neighbors are not added then simply add the current neighbors to the end of the array 
				if (curr_K < K)
				{
					topKClasses[curr_K] = labels[j];
					topKdistances[curr_K] = distance;
					curr_K++;
				}
				else
				{

					// if K neighbors are already existing then sort the neighbors
					if (curr_K == K)
					{
						mySort(topKdistances, topKClasses);
						curr_K++;
					}


					// insert the current neighbors in sorted order
					insertValueInSorted(topKdistances, topKClasses, distance, labels[j]);
				}
			}
		}


		// find the most frequent predicted class in neighbors and make the prediction
		predictions[i] = mostFrequentLabel(topKClasses);

		// if prediction is not same as actual label then this is inaccurate prediction
		if (predictions[i] != labels[i])
		{
			countDiff++;
		}

		free(predictions);
		free(topKClasses);
		free(topKdistances);
	}



	// accumelate the total incorrect prediction at the master process
	if (myRank == _MASTER)
	{
		MPI_Reduce(MPI_IN_PLACE, &countDiff, 1, MPI_INT, MPI_SUM, _MASTER, MPI_COMM_WORLD);
	}
	else
	{
		// each process will send its incorrect prediction
		MPI_Reduce(&countDiff, NULL, 1, MPI_INT, MPI_SUM, _MASTER, MPI_COMM_WORLD);
	}



	if (myRank == _MASTER)
	{
		printf("Accuracy: %g\n", (rows - countDiff) / (double)rows);
		printf("Total time taken is: %g sec\n", MPI_Wtime() - start);
	}

	free(data[0]);
	free(data);
	free(labels);

	MPI_Finalize();

	return 0;
}

// allocate memory for two dimensional matrix but of double type
double** create2DArray(const int rows, const int cols)
{
	// allocate 2d pointers 
	double** arr = (double**)malloc(rows * sizeof(double*));

	// allocate memory for all cells of 2d array
	double* temp = (double*)malloc(rows * cols * sizeof(double));

	// assign pointers for each row
	for (int i = 0; i < rows; i++)
	{
		arr[i] = &temp[i * cols];
	}
	return arr;
}

double** readData(const char* file_name, double** labels, int* rows, int* cols)
{
	*cols = *rows = 0;
	FILE* fin;


	// open the file
	fin = fopen(file_name, "r");

	if (fin)
	{
		char line[MAX_LINE_LEN];

		//first line is column names, so ignore it
		fgets(line, MAX_LINE_LEN - 1, fin);

		// get all the token splitted by comma
		char* ptr = strtok(line, ",");
		while (ptr != NULL)
		{

			// compute total number of columns in the first row
			ptr = strtok(NULL, ",");
			(*cols)++;
		}

		// last column is the label so exlude it from the features
		(*cols)--;



		//  compute total rows in the file
		while (fgets(line, MAX_LINE_LEN - 1, fin))
		{
			(*rows)++;
		}
		fclose(fin);

		fin = fopen(file_name, "r");


		// allocate memory to store all the rows and labels
		double** data = create2DArray(*rows, *cols);
		*labels = (double*)malloc((*rows) * sizeof(double));



		// reopen the file 
		fgets(line, MAX_LINE_LEN - 1, fin);


		//reiterate the file and read all the values
		int i = 0;
		while (fgets(line, MAX_LINE_LEN - 1, fin))
		{
			char* pt = strtok(line, ",");
			int j = 0;
			do
			{
				// initial cells are features
				if (j < (*cols))
				{

					// convert string to float
					sscanf(pt, "%lf", &data[i][j]);
					j++;

				}
				else
				{
					// last value is label
					sscanf(pt, "%lf", &(*labels)[i]);
				}
				pt = strtok(NULL, ",");
			} while (pt != NULL);
			i++;
		}

		return data;


	}
	else
	{
		printf("File doesn't exist\n");
		exit(0);
	}

	return NULL;
}

void mySortArray(double* arr, int len)
{

	// apply insertion sort on the given array
	int i = 0;
	for (i = 1; i < len; i++)
	{
		double key = arr[i];
		int j = i - 1;
		while (j >= 0 && arr[j] < key)
		{
			arr[j + 1] = arr[j];
			j = j - 1;
		}
		arr[j + 1] = key;
	}
}

void mySort(double* topKdistances, double* topKClasses)
{
	// insertion sort sort the distance and corresponding classes
	for (int i = 1; i < K; i++)
	{
		double key = topKdistances[i];
		double keyClass = topKClasses[i];
		int j = i - 1;
		while (j >= 0 && topKdistances[j] < key)
		{
			topKdistances[j + 1] = topKdistances[j];
			topKClasses[j + 1] = topKClasses[j];
			j = j - 1;
		}
		topKdistances[j + 1] = key;
		topKClasses[j + 1] = keyClass;
	}
}

int mostFrequentLabel(double* classes)
{

	// sort the array of classes/labels
	mySortArray(classes, K);




	// since each label is sorted so  keep track of most occurence of a number 
	double mostFreq = classes[0];

	int count = 1;
	int maxOccur = 1;

	for (int i = 0; i < K; i++)
	{
		// if the two consecutive values are same then add to count
		if (classes[i] == classes[i - 1])
		{
			count++;
		}
		else
		{


			// if count exceeds the most frequent then it is most occured label
			if (count > maxOccur)
			{
				maxOccur = count;
				mostFreq = classes[i - 1];
			}

			// if two values are not same set count to one
			count = 1;
		}
	}


	// check for last label if it is most frequent
	if (count > maxOccur)
	{
		maxOccur = count;
		mostFreq = classes[K - 1];
	}

	return mostFreq;
}

void insertValueInSorted(double* topKdistances, double* topKClasses, double newScore, double newLabel)
{
	if (newScore >= topKdistances[0])
		return;
	int i = 0;

	// move all the labels above unless  get to the exact position of label and insert it there
	while (i + 1 < K && newScore < topKdistances[i + 1])
	{
		topKdistances[i] = topKdistances[i + 1];
		topKClasses[i] = topKClasses[i + 1];
		i++;
	}

	topKdistances[i] = newScore;
	topKClasses[i] = newLabel;
}

// euclidean distance between two points
double getEuclideanDistance(double* query, double* target, int cols)
{
	double distance = 0;
	for (int j = 0; j < cols; j++)
	{

		double diff = query[j] - target[j];
		distance += diff * diff;
	}

	distance = sqrt(distance);

	return distance;
}
