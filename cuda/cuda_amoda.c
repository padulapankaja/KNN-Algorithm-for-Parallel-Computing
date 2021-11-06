%%cu
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <limits.h>
#include <cstdlib>
#include <string.h>
#include <sys/time.h>

#define DATA_FILE "/content/mainData.csv"
#define GUESS_ENDPOINTS "/content/guess.csv"

    float **
    makearray2d(int rows, int cols)
{
    float *data = (float *)malloc(rows * cols * sizeof(float));
    float **array = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
        array[i] = &(data[cols * i]);

    return array;
}

__host__ __device__ inline static float dist(int N_features, float *x, float *y)
{
    float xsum = 0.0, ysum = 0.0, xysum = 0.0, xsqr_sum = 0.0, ysqr_sum = 0.0;
    for (int j = 0; j < N_features; j++)
    {
        xsum = xsum + x[j];
        ysum = ysum + y[j];
        xysum = xysum + x[j] * y[j];
        xsqr_sum = xsqr_sum + x[j] * x[j];
        ysqr_sum = ysqr_sum + y[j] * y[j];
    }

    float num = ((N_features * xysum) - (xsum * ysum));
    float deno = ((N_features * xsqr_sum - xsum * xsum) * (N_features * ysqr_sum - ysum * ysum));
    float coeff = num / sqrt(deno);
    return (coeff);
}

__global__ void kmeansMain(float **data, float **old_centers, int features, float *new_sum_distance, int *labels)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    float min_distance_zero = dist(features, data[i], old_centers[0]);
    float min_distance_one = dist(features, data[i], old_centers[1]);
    if (min_distance_zero < min_distance_one)
    {
        labels[i] = 0;
    }
    else
    {
        labels[i] = 1;
    }
}

void parse_csv(const char *file_name, float ***data_p, const int rows, const int cols)
{
    FILE *csv_file;
    csv_file = fopen(file_name, "r");
    if (csv_file == NULL)
    {
        printf("Error: can't open file: %s\n", file_name);
        exit(-1);
    }

    char *delimiter = (char *)malloc(sizeof(char));
    *delimiter = ',';

    char *buffer = (char *)malloc(BUFSIZ);
    char *token;

    int row = 0;
    int idx = 0;
    int rowx = 0;

    // Reach each line of the file into the buffer.
    while (row <= rows && fgets(buffer, BUFSIZ, csv_file) != NULL)
    {
        if (++row == 1)
            continue;

        // Get every token and print it.
        token = strtok(buffer, delimiter);
        idx = 0;
        while (token != NULL && idx < cols)
        {
            //printf("%d, %d\n", rowx, idx);
            (*data_p)[rowx][idx] = atof(token);
            ++idx;

            // Get the next token.
            token = strtok(NULL, delimiter);
        }
        ++rowx;
    }

    printf("read %d rows from file %s\n\n", row - 1, file_name);

    fclose(csv_file);
    free(buffer);
}

// Track CPU Time
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(void)
{
    printf("Start\n");
    double start_time, round_time = 0;
    double main_start_time;

    main_start_time = cpuSecond();
    int rows = 88000;
    int features = 6;
    int clusters = 2;
    int repeat = 5;

    int i = 0;
    int j = 0;
    int k = 0;

    int k_best, init;

    float **data = makearray2d(rows, features);
    float **guess = makearray2d(repeat, clusters);

    float all_time_best = FLT_MAX;
    float old_sum_distance, new_sum_distance;

    // The position of each cluster center.
    float **old_centers = makearray2d(clusters, features);
    float **new_centers = makearray2d(clusters, features);

    // read data files and cluster starting point guesses
    parse_csv(DATA_FILE, &data, rows, features);
    parse_csv(GUESS_ENDPOINTS, &guess, repeat, clusters);

    // each data point belongs to which cluster [0 or 1]
    int *labels = (int *)malloc(rows * sizeof(int));
    int *best_labels = (int *)malloc(rows * sizeof(int));

    // how many data points in the cluster
    int *c_size = (int *)malloc(clusters * sizeof(int));

    float **d_data;
    cudaMalloc((float **)&d_data, rows * features * sizeof(float));

    int *d_labels;
    cudaMalloc((void **)&d_labels, rows * sizeof(int));

    float **d_old_centers;
    cudaMalloc((float **)&d_old_centers, clusters * features * sizeof(float));

    float *d_new_sum_distance;
    cudaMalloc((void **)&d_new_sum_distance, sizeof(float));

    //copy from host to device data
    cudaMemcpy(d_data, data, rows * features * sizeof(float), cudaMemcpyHostToDevice);

    printf("\nStart Model Traning\n");

    /* Run the K-mean algorithm for repeat times with 
     * different starting points
     */
    for (int ir = 0; ir < repeat; ir++)
    {
        // guess initial centers
        for (k = 0; k < clusters; k++)
        {
            c_size[k] = 0; // for accumulating
            // the index of data points as the initial guess for cluster centers
            init = (int)guess[ir][k];

            for (j = 0; j < features; j++)
            {
                //printf("%f\t", data[init][j]);
                old_centers[k][j] = data[init][j];
                //set the "new" array to 0 for accumulating
                new_centers[k][j] = 0.0;
            }
            //printf("\n");
        }

        // core K - meanbegins here !!

        int iterated_times = 0;
        new_sum_distance = 0.0;

        //copy from host to device new_sum_distance
        //cudaMemcpy(d_new_sum_distance, new_sum_distance, sizeof(float), cudaMemcpyHostToDevice);

        //copy from host to device old_centers
        cudaMemcpy(d_old_centers, old_centers, clusters * features * sizeof(float), cudaMemcpyHostToDevice);

        //copy from host to device labels
        cudaMemcpy(d_labels, labels, rows * sizeof(int), cudaMemcpyHostToDevice);

        do
        {
            iterated_times++;
            old_sum_distance = new_sum_distance;
            new_sum_distance = 0.0;

            start_time = cpuSecond();
            //main function-----------------------
            kmeansMain<<<ceil(rows / 100), 100>>>(d_data, d_old_centers, features, d_new_sum_distance, d_labels);
            //------------------------------------

            // Copy result back to host
            cudaMemcpy(labels, d_labels, rows * sizeof(int), cudaMemcpyDeviceToHost);

            round_time += (cpuSecond() - start_time);

            // Set the cluster centers to the mean
            for (i = 0; i < rows; i++)
            {
                k_best = labels[i];
                c_size[k_best] = c_size[k_best] + 1;
                for (j = 0; j < features; j++)
                    new_centers[k_best][j] += data[i][j];
            }

            //Convert the sum to the mean
            for (k = 0; k < clusters; k++)
            {
                for (j = 0; j < features; j++)
                {
                    if (c_size[k] > 0)
                        old_centers[k][j] = new_centers[k][j] / c_size[k];
                    new_centers[k][j] = 0.0; // for the next round
                }
                c_size[k] = 0; // for the next round
            }

        } while (iterated_times == 1 || (iterated_times < 100));
        printf("it %d in round %d %f", iterated_times, ir, (old_sum_distance - new_sum_distance));

        // record the best results
        if (new_sum_distance < all_time_best)
        {
            all_time_best = new_sum_distance;
            for (i = 0; i < rows; i++)
                best_labels[i] = labels[i];
        }

        printf("Repeat %d Done\n\n", ir);
    }
    printf("Total Kernal Time : %f\n", round_time );
    printf("Total Execution Time : %f\n",  cpuSecond() - main_start_time);
    printf("\nEnd");

    return 0;
}