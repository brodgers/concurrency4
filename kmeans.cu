#include <iostream>
#include <fstream>
#include <getopt.h>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <cassert>
#include <cfloat>
#include <climits>
#include <chrono>
#include <pthread.h>
#include <map>

//to compile:  export PATH=/opt/cuda-8.0/bin:$PATH
//             nvcc -std=c++11 -g -o kmeans kmeans.cu

#define no_argument 0
#define required_argument 1
#define optional_argument 2

using namespace std;

struct params {
	int num_clusters;
	double threshold;
	int max_iters;
	int num_features;
	int num_points;
};

/* Arguments */
int NUM_CLUSTERS = 2;
double THRESHOLD = .0000001f;
int MAX_ITERS = 0;
int NUM_WORKERS = 1;
string PATH = "kmeans-sample-inputs/color100";

/* Models a data point */
typedef struct Point {
	vector<double> vals;
	int label;

	Point() : vals(), label(0) {}

	int size() {
		return vals.size();
	}

	double& operator [] (int i) {
		return vals[i];
	}
} Point;

/* Input data */
vector<Point> data;

/* Models a centroid */
typedef struct Centroid {
	vector<double> c;
	int label;

	Centroid(int dims, int l) : c(dims), label(l) {
		for(int i = 0; i < dims; i++) {
			double r = rand() / static_cast <double> (RAND_MAX);
			c[i] = r;
		}
	}

	Centroid(int dims, int l, int i) : c(dims), label(l) {
		Point p = data[i];
		for(int x = 0; x < p.size(); x++) {
			c[x] = p[x];
		}
	}

	int size() {
		return c.size();
	}

	double& operator [] (int i) {
		return c[i];
	}

	void out() {
		cout << "Cluster " << label << " center: ";
		for(double f : c) {
			cout << "[" << f << "]";
		}
		cout << endl;
	}
} Centroid;

/* Cluster centers */
vector<Centroid> centroids;

/* Cuda Variables */
double* d_data;
double* d_centroids;
params* d_params;
int* d_labels;

params p;

/* Handles command line arguments */
void args(int argc, char* argv[]) {
	const struct option longopts[] =
	{
		{"clusters",   required_argument, 0, 'c'},
		{"threshold",  required_argument, 0, 't'},
		{"iterations", required_argument, 0, 'i'},
		{"workers",    required_argument, 0, 'w'},
		{"input",      required_argument, 0, 'f'},
		{0, 0, 0, 0}
	};

	int index;
	int iarg = 0;

	opterr=1;

	while(iarg != -1) {
		iarg = getopt_long(argc, argv, "c:t:i:w:p:", longopts, &index);

		switch(iarg) {
			case 'c':
				NUM_CLUSTERS = atoi(optarg);
				break;
			case 't':
				THRESHOLD = atof(optarg);
				break;
			case 'i':
				MAX_ITERS = atoi(optarg);
				if(MAX_ITERS == 0) {
					MAX_ITERS = INT_MAX;
				}
				break;
			case 'w':
				NUM_WORKERS = atoi(optarg);
				break;
			case 'f':
				PATH = optarg;
				break;
		}
	}

}

/* Reads in input from file denoted by PATH */
void input() {
	ifstream file(PATH);
	if (file.is_open()) {
		string line;
		getline(file, line);
		int num_points = stoi(line);

		data = vector<Point>(num_points);
		double c;
		int id;
		while(getline(file, line)) {
			istringstream ss(line);
			ss >> id;
			while(ss >> c) {
				data[id - 1].vals.push_back(c);
			}
		}

		int size = sizeof(double) * data.size() * data[0].size();
		
		cudaMalloc((void**)&d_data, size);

		for(int i = 0; i < data.size(); i++) {
			for(int j = 0; j < data[0].size(); j++)
				cudaMemcpy(d_data + i * data[0].size() + j, &data[i].vals[j], sizeof(double), cudaMemcpyHostToDevice);
		}

		p.num_clusters = NUM_CLUSTERS;
		p.threshold = THRESHOLD;
		p.max_iters = MAX_ITERS;
		p.num_features = data[0].size();
		p.num_points = data.size();

		cudaMalloc((void**)&d_params, sizeof(params));
		cudaMemcpy(d_params, &p, sizeof(params), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&d_labels, sizeof(int) * data.size());
	}
}

/* Reports output */
// void output(auto dur) {
// 	// cout << "Converged in " << iterations << " iterations (max=" << MAX_ITERS << ")\n";
// 	typedef std::chrono::duration<float> float_seconds;
// 	auto secs = std::chrono::duration_cast<float_seconds>(dur);
// 	cout << secs.count() << endl;
// 	// for(Centroid& c : centroids) {
// 	// 	c.out();
// 	// }
// }

/* Creates centroids initially set at random points */
void randomCentroids(int num_features) {
	for(int i = 0; i < NUM_CLUSTERS; i++) {
		centroids.emplace_back(num_features, i, rand() % data.size());
	}
	// for (Centroid c : centroids) {
	// 	c.out();
	// }

	int size = NUM_CLUSTERS * num_features * sizeof(double);	
	cudaMalloc((void**) &d_centroids, size);
	for(int i = 0; i < NUM_CLUSTERS; i++) {
		for(int j = 0; j < num_features; j++) {
			cudaMemcpy(d_centroids + i * num_features + j, &centroids[i].c[j], sizeof(double), cudaMemcpyHostToDevice);
		}
	}
}

/* Calculate euclidean distance between two points */
__device__ double euclideanDistance(double* a, double* b, int num_features) {
	double d = 0.0;
	for(int i = 0; i < num_features; i++) {
		d += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(d);
}

__device__ double recalculate(int label, int* labels, double* d_data, double* centroid, params* p, double* sums) {
	double distance = 0.0;
	for(int i = 0; i < p->num_features; i++) {
		sums[i] = 0.0;
	}
	int count = 0;
	for(int i = 0; i < p->num_points; i++) {
		if(label == labels[i]) {
			for(int j = 0; j < p->num_features; j++) {
				sums[i] += d_data[i * p->num_features + j];
				count++;
			}
		}
	}
	if(count > 0) {
		for(int i = 0; i < p->num_features; i++) {
			sums[i] /= count;
		}
		distance = euclideanDistance(centroid, sums, p->num_features);
		for(int i = 0; i < p->num_features; i++) {
			centroid[i] = sums[i];
		}
	}
	return distance;
}

/* Recalculates centroid centers and returns the max difference between old and new centers */
__global__ void recalculateCentroids(double* d_centroids, double* d_data, int* labels, params* p, double* sums, double* max_diff) {
	double max = 0.0;
	for(unsigned int i = 0; i < p->num_clusters; i++) {
		double diff = recalculate(i, labels, d_data, &d_centroids[i * p->num_features], p, sums);
		if(diff > max) {
			max = diff;
		}
	}
	*max_diff = max;
}

/* Finds the nearest centroid to a point */
__device__ int nearest(double* d_centroids, double* point, params *p) {
	float distance = FLT_MAX;
	int index;
	for(unsigned int i = 0; i < p->num_clusters; i++) {
		float d = euclideanDistance(&d_centroids[p->num_features * i], point, p->num_features);
		if(d < distance) {
			distance = d;
			index = i;
		}
	}
	return index;
	
}

// void aggregate_clusters(vector<vector<int>>& local_clusters) {
// 	for(int i = 0; i < NUM_CLUSTERS; i++) {
// 		clusters[i].insert(clusters[i].end(), local_clusters[i].begin(), local_clusters[i].end());
// 		local_clusters[i].clear();
// 	}
// }

/* Finds the nearest centroids for all points and populates clusters */
__global__ void findNearestCentroids(double* d_centroids, double* d_data, int* d_labels, params* d_params) {
	int index = blockIdx.x * blockDim.x + threadIdx.x * d_params->num_features;
	if(index < d_params->num_points) {
		d_labels[blockIdx.x * blockDim.x + threadIdx.x] = nearest(d_centroids, &d_data[blockIdx.x * blockDim.x + threadIdx.x * d_params->num_features], d_params);
	}
}

/* Runs kmeans on dataset */
__global__ void kmeans(double* d_centroids, double* d_data, int* d_labels, params* d_params, double* sums, double* diff) {
	int iterations = 0;
	bool done = false;

	while(!done) {
		findNearestCentroids<<<20, 1024>>>(d_centroids, d_data, d_labels, d_params);
		// aggregate_clusters(local_clusters);
		cudaDeviceSynchronize();

		recalculateCentroids<<<1, 1>>>(d_centroids, d_data, d_labels, d_params, sums, diff);
		cudaDeviceSynchronize();

		done = ++iterations >= d_params->max_iters || *diff <= d_params->threshold;
	}
}

int main(int argc, char* argv[]) {
	args(argc, argv);
	input();

	double* sums;
	cudaMalloc((void**)&sums, sizeof(double) * p.num_features);
	double* diff;
	cudaMalloc((void**)&diff, sizeof(double));

	randomCentroids(data[0].size());

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	kmeans<<<1, 1>>>(d_centroids, d_data, d_labels, d_params, sums, diff);
	// cudaDeviceSynchronize();
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << milliseconds << endl;

}
