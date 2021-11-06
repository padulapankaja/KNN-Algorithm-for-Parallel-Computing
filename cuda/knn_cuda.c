%%cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#define K 5
#define MAX_LINE_LEN 200


void mySort(double* topKdistances, double* topKClasses);

__host__ __device__ void mySortArray(double* arr, int len);

int mostFrequentLabel(double* classes);

__host__ __device__ void insertValueInSorted(double* topKdistances, double* topKClasses, double newScore, double newLabel);

__host__ __device__ double getEuclideanDistance(double* query, double* target, int cols);

double** create2DArray(const int rows, const int cols);

double** readData(const char* file_name, double** labels, int* rows, int* cols);


// Track CPU Time
double cpuSecond()
{
	

    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


__global__ void knnMain(double* labels, double** d_data, double* topKdistances, double* topKClasses, int curr_K, int d_i, int cols)
// d_topKdistances, d_topKClasses, d_curr_K, d_i
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (d_i != j)
			{
				double distance = getEuclideanDistance(&d_data[d_i][0], &d_data[j][0], cols);

				if (curr_K < 5)
				{
					topKClasses[curr_K] = labels[j];
					topKdistances[curr_K] = distance;
				}
				else
				{
					if (curr_K == 5)
					{
            for (int i = 1; i < 5; i++)
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

					insertValueInSorted(topKdistances, topKClasses, distance, labels[j]);
				}
			}
}


int main()
{

printf("Start\n");
    double start_time, round_time = 0;
    double main_start_time;

    main_start_time = cpuSecond();

	const char* file_name = "/content/trainnew_small.csv";
	int rows, cols;
	double* labels;
	double** data = readData(file_name, &labels, &rows, &cols);

	
	clock_t start = clock();
	int countDiff = 0;

    double **d_data;
    cudaMalloc((double **)&d_data, rows * cols * sizeof(double));
    double *d_predction;
    double *d_topKdistances;
    double *d_topKClasses;

	double* d_labels;
    cudaMalloc((double **)&d_labels, rows * cols * sizeof(double));

	cudaMemcpy(d_data, data, rows * cols  * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_labels, labels, rows * cols  * sizeof(double), cudaMemcpyHostToDevice);

	for (int i = 0; i < rows; i++)
	{
		start_time = cpuSecond();

		double* predictions = (double*)malloc(rows * sizeof(double));
		double* topKdistances = (double*)malloc(K * sizeof(double));
		double* topKClasses = (double*)malloc(K * sizeof(double));
		
		int curr_K = 0;

    	cudaMemcpy(d_topKdistances, topKdistances, K  * sizeof(double), cudaMemcpyHostToDevice);
    	cudaMemcpy(d_topKClasses, topKClasses, K  * sizeof(double), cudaMemcpyHostToDevice);

		//Kernal Function
        knnMain<<<ceil(rows / 2000), 2000>>>(d_labels, d_data, d_topKdistances, d_topKClasses, curr_K, i, cols);
		
		// cuda copy topKClasses back to host
    	cudaMemcpy(d_topKdistances, topKdistances, K  * sizeof(double), cudaMemcpyDeviceToHost);

		predictions[i] = mostFrequentLabel(topKClasses);

		if (predictions[i] != labels[i])
		{
			countDiff++;
		}

		free(predictions);
		free(topKClasses);
		free(topKdistances);
	 round_time += (cpuSecond() - start_time);


	}




	printf("Accuracy: %g\n", (rows - countDiff) / (double)rows);
	// double timeTaken = ((double)clock() - start)/CLOCKS_PER_SEC;
    // printf("Serial version took %g secs\n", timeTaken);

	free(data[0]);
	free(data);
	free(labels);
	printf("Total Kernal Time : %f\n", round_time );
    printf("Total Execution Time : %f\n",  cpuSecond() - main_start_time);
    printf("\nEnd");
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

	fin = fopen(file_name, "r");

	if (fin)
	{
		char line[MAX_LINE_LEN];
		fgets(line, MAX_LINE_LEN - 1, fin);
		char* ptr = strtok(line, ",");
		while (ptr != NULL)
		{
			ptr = strtok(NULL, ",");
			(*cols)++;
		}
		(*cols)--;


		while (fgets(line, MAX_LINE_LEN - 1, fin))
		{
			(*rows)++;
		}
		fclose(fin);

		fin = fopen(file_name, "r");

		double** data = create2DArray(*rows, *cols);
		*labels = (double*)malloc((*rows) * sizeof(double));

		fgets(line, MAX_LINE_LEN - 1, fin);

		int i = 0;
		while (fgets(line, MAX_LINE_LEN - 1, fin))
		{
			char* pt = strtok(line, ",");
			int j = 0;
			do
			{
				if (j < (*cols))
				{
					sscanf(pt, "%lf", &data[i][j]);
					j++;

				}
				else
				{
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
	mySortArray(classes, K);
	double mostFreq = classes[0];

	int count = 1;
	int maxOccur = 1;

	for (int i = 0; i < K; i++)
	{
		if (classes[i] == classes[i - 1])
		{
			count++;
		}
		else
		{
			if (count > maxOccur)
			{
				maxOccur = count;
				mostFreq = classes[i - 1];
			}
			count = 1;
		}
	}

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
	while (i + 1 < K && newScore < topKdistances[i + 1])
	{
		topKdistances[i] = topKdistances[i + 1];
		topKClasses[i] = topKClasses[i + 1];
		i++;
	}

	topKdistances[i] = newScore;
	topKClasses[i] = newLabel;
}




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
