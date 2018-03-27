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
};

/* Arguments */
int NUM_CLUSTERS = 2;
double THRESHOLD = .0000001f;
int MAX_ITERS = 0;
int NUM_WORKERS = 1;
string PATH = "kmeans-sample-inputs/color100";

/* Global clusters array */
vector<vector<int>> clusters;


/* Models a data point */
typedef struct Point {
	vector<float> vals;
	int label;

	Point() : vals(), label(0) {}

	int size() {
		return vals.size();
	}

	float& operator [] (int i) {
		return vals[i];
	}
} Point;

/* Input data */
vector<Point> data;

/* Cuda Variables */
double* d_data;
double* d_centroids;
params* d_params;


/* Models a centroid */
typedef struct Centroid {
	vector<float> c;
	int label;

	Centroid(int dims, int l) : c(dims), label(l) {
		for(int i = 0; i < dims; i++) {
			float r = rand() / static_cast <float> (RAND_MAX);
			c[i] = r;
		}
	}

	Centroid(int dims, int l, int i) : c(dims), label(l) {
		Point p = data[i];
		for(int x = 0; x < p.size(); x++) {
			c[x] = p[x];
		}
	}

	double recalculate() {
		vector<float> sums(c.size(), 0.0f);
		for(int i : clusters[label]) {
			Point& p = data[i];
			for(unsigned int i = 0; i < c.size(); i++) {
				sums[i] += p.vals[i];
			}
		}

		double distance = 0.0f;
		for(unsigned int i = 0; i < c.size(); i++) {
			double n = sums[i] / clusters[label].size();
			distance += (n - c[i]) * (n - c[i]);
			c[i] = n;
		}
		clusters[label].clear();
		return sqrt(distance);
	}

	int size() {
		return c.size();
	}

	float& operator [] (int i) {
		return c[i];
	}

	void out() {
		cout << "Cluster " << label << " center: ";
		for(float f : c) {
			cout << "[" << f << "]";
		}
		cout << endl;
	}
} Centroid;

/* Cluster centers */
vector<Centroid> centroids;

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

		clusters = vector<vector<int>>(NUM_CLUSTERS);	

		data = vector<Point>(num_points);
		float c;
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
			cudaMemcpy(d_data + i * data[0].size(), &data[i], sizeof(double) * data[0].size(), cudaMemcpyHostToDevice);
		}

		params p;
		p.num_clusters = NUM_CLUSTERS;
		p.threshold = THRESHOLD;
		p.max_iters = MAX_ITERS;
		p.num_features = data[0].size();

		cudaMalloc((void**)&d_params, sizeof(params));
		cudaMemcpy(d_params, &p, sizeof(params), cudaMemcpyHostToDevice);
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

	int size = NUM_CLUSTERS * num_features * sizeof(double);	
	cudaMalloc((void**) &d_centroids, size);
	for(int i = 0; i < NUM_CLUSTERS; i++) {
		cudaMemcpy(d_centroids + i * num_features, &centroids[i], sizeof(double) * num_features, cudaMemcpyHostToDevice);
	}
}

/* Recalculates centroid centers and returns the max difference between old and new centers */
double recalculateCentroids(unsigned int id) {
	double max = 0.0f;
	for(unsigned int i = 0; i < centroids.size(); i++) {
		if(i % NUM_WORKERS == id) {
			Centroid& c = centroids[i];
			double diff = c.recalculate();
			if(diff > max) {
				max = diff;
			}
		}
	}

	return max;
}

/* Calculate euclidean distance between two points */
template <typename T1, typename T2>
float euclideanDistance(T1& a, T2& b) {
	assert(a.size() == b.size());
	float d = 0.0f;
	for(int i = 0; i < a.size(); i++) {
		d += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(d);
}

/* Finds the nearest centroid to a point */
int nearest(double* d_centroids) {
	// float distance = FLT_MAX;
	// int index;
	// for(unsigned int i = 0; i < centroids.size(); i++) {
	// 	float d = euclideanDistance(centroids[i], p);
	// 	if(d < distance) {
	// 		distance = d;
	// 		index = i;
	// 	}
	// }
	// return index;
	
}

void aggregate_clusters(vector<vector<int>>& local_clusters) {
	for(int i = 0; i < NUM_CLUSTERS; i++) {
		clusters[i].insert(clusters[i].end(), local_clusters[i].begin(), local_clusters[i].end());
		local_clusters[i].clear();
	}
}

/* Finds the nearest centroids for all points and populates clusters */
__global__ findNearestCentroid(double* d_centroids, double* d_data, params* p_params) {
	for(int i = 0; i < params->num_features; i++) {
		
	}
}

/* Runs kmeans on dataset */
__global__ void* kmeans(double* d_centroids, double* d_data, params* p_params) {
	// int id = (long) _id;
	// int start = data.size() / NUM_WORKERS * id;
	// int end = data.size() / NUM_WORKERS * (id + 1);
	// if(id == NUM_WORKERS - 1)
	// 	end = data.size();

	// vector<vector<int>> local_clusters(NUM_CLUSTERS);

	int iterations = 0;
	bool done = false;

	while(!done) {
		findNearestCentroids(d_centroids, d_data, p_params);
		// aggregate_clusters(local_clusters);

		double max_diff = recalculateCentroids(id);

		done = ++iterations >= MAX_ITERS || max_diff <= THRESHOLD;
	}

	return nullptr;
}

int main(int argc, char* argv[]) {
	args(argc, argv);
	input();

	/* Time start */
	auto before = chrono::system_clock::now();

	randomCentroids(data[0].size());

	vector<pthread_t> threads(NUM_WORKERS);
	for(long i = 0; i < NUM_WORKERS; i++) {
		pthread_create(&threads[i], NULL, kmeans, (void*)i);
	}

	for(int i = 0; i < NUM_WORKERS; i++) {
		pthread_join(threads[i], NULL);
	}

	/* Time finish */
	auto end = chrono::system_clock::now();
	auto dur = end - before;

	/* Output */
	// output(dur);

	// CALL THIS 
	// cudaDeviceSyncronize();
}
