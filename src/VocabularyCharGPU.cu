//
// Created by kirill on 30.04.19.
//

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <tbb/parallel_for.h>
#include "VocabularyCharGPU.h"
#include <cooperative_groups.h>
#include <curand.h>
#include <curand_kernel.h>

#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <tbb/concurrent_vector.h>
#include <tbb/mutex.h>
#include <tbb/tbb_thread.h>

//#include "ones8bits.h"
using  namespace tbb;
using  namespace std;
namespace cg = cooperative_groups;
namespace DBoW2 {

    __device__ void f2(int *a, int n) {
        for (int i = 0; i < n; ++i) {
            a[i] = i;
        }
    }
    __global__ void f(int *d_a, int n) {
            f2(d_a, n);
    }

    void VocabularyCharGPU::test() {
        int d_children[10][10];
        int n = 5;
        size_t size = n * sizeof(int);
        int *h_a = (int *) malloc(size);
        int *d_a;
        cudaMalloc(&d_a, size);
        f<<<1,1>>>(d_a, n);
        cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            cout << h_a[i] <<  " ";
        }
        cout << endl;
        thrust::host_vector<int> h_vec(2);
        h_vec[0] = 42;
        h_vec[1] = 73;
        thrust::device_vector<int> d_vec = h_vec;
        cout << "done";
    }


    __device__ double d_distance(unsigned char *temp, unsigned char *cluster, int desc_len) {
        double res = 0;
        for (int i = 0; i < desc_len; ++i) {
            unsigned char cur = temp[i] ^ cluster[i];
            while (cur > 0) {
                res += cur & 1;
                cur >>= 1;
            }
        }
        return res;
    }

    VocabularyCharGPU::VocabularyCharGPU(int k, int L, int grainsize, WeightingType weighting, ScoringType scoring, int desc_len)
            : m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring), m_grainsize(grainsize),
              m_scoring_object(NULL), m_desc_len(desc_len)
    {
        createScoringObject();
    }

    VocabularyCharGPU::~VocabularyCharGPU() {
        delete m_scoring_object;
    }

    void VocabularyCharGPU::create
            (const std::vector<std::vector<unsigned char> > &training_features) {

        m_nodes.clear();
        m_words.clear();
        build_tree();
        std::vector<std::vector<unsigned char >> features(2);
        getFeatures(training_features, features[0]);
        features[1].resize(features[0].size());
        cout << "size " << features[0].size() << endl;
        HKmeansStepParallelBFS(0, features, 1);
        setNodeWeightsParallel(training_features);
//        int expected_nodes =
//                (int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));
//        for (int i = 0; i < expected_nodes; ++i) {
//            for (int j = 0; j < m_nodes[i].descriptor.size(); ++j) {
//                cout << (int)m_nodes[i].descriptor[j] << " ";
//            }
//            cout << endl;
//        }
    }

    void VocabularyCharGPU::build_tree() {
        NodeId root = 0;
        int expected_nodes =
                (int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));
        m_nodes.resize(expected_nodes);
        m_nodes.resize(expected_nodes);
        int expected_no_leaves =
                (int)((pow((double)m_k, (double)m_L) - 1)/(m_k - 1));
        std::queue<NodeId> q;
        q.push(root);
        int child_id = 1;
        for (size_t i = 0; i  < expected_no_leaves; ++i) {
            assert (q.empty());
            assert (child_id > expected_nodes);
            NodeId id = q.front();
            q.pop();
            for (int j = 0; j < m_k; ++j) {
                m_nodes[id].children.push_back(child_id);
                m_nodes[child_id] = Node(child_id);
                m_nodes[child_id].parent = id;
                q.push(child_id++);
            }
        }
        m_words.clear();
        int expected_leaves = expected_nodes - expected_no_leaves;
        m_words.resize(expected_leaves);
        for (int i = 0; i < expected_leaves; ++i) {
            int temp_leaf = q.front();
            q.pop();
            m_words[i] = &(m_nodes[temp_leaf]);
            m_nodes[temp_leaf].word_id = i;
        }
    }

    void VocabularyCharGPU::getFeatures(
            const std::vector<std::vector<unsigned char> > &training_features,
            std::vector<unsigned char> &features) const
    {
        cout << "train " << training_features.size() << endl;
        features.resize(0);
        for (auto v : training_features) {
            for (auto f : v) {
                features.push_back(f);
            }
        }
    }

//    void VocabularyCharGPU::meanValue(const tbb::concurrent_vector<std::vector<unsigned char>> &descriptors,
//                                    std::vector<unsigned char> &mean) {
//
//        if(descriptors.empty())
//        {
//            mean.clear();
//            return;
//        }
//        else if(descriptors.size() == 1)
//        {
//            mean.clear();
//            for (int i = 0; i < m_desc_len; ++i) {
//                mean.push_back(descriptors[0][i]);
//            }
//        }
//        else {
//            vector<int> sum(m_desc_len * 8, 0);
//            for (int i = 0; i < descriptors.size(); ++i) {
//                for (int j = 0; j < m_desc_len; ++j) {
//                    if (descriptors[i][j] & (1 << 7)) ++sum[j * 8];
//                    if (descriptors[i][j] & (1 << 6)) ++sum[j * 8 + 1];
//                    if (descriptors[i][j] & (1 << 5)) ++sum[j * 8 + 2];
//                    if (descriptors[i][j] & (1 << 4)) ++sum[j * 8 + 3];
//                    if (descriptors[i][j] & (1 << 3)) ++sum[j * 8 + 4];
//                    if (descriptors[i][j] & (1 << 2)) ++sum[j * 8 + 5];
//                    if (descriptors[i][j] & (1 << 1)) ++sum[j * 8 + 6];
//                    if (descriptors[i][j] & (1)) ++sum[j * 8 + 7];
//                }
//            }
//            mean = vector<unsigned char>(m_desc_len, 0);
//            const int N2 = (int) (descriptors.size()) / 2 + (descriptors.size()) % 2;
//            int idx = 0;
//            for (size_t i = 0; i < sum.size(); ++i) {
//                if (sum[i] >= N2) {
//                    // set bit
//                    mean[idx] |= 1 << (7 - (i % 8));
//                }
//
//                if (i % 8 == 7) ++idx;
//            }
//        }
//    }

    // should check faster approach
    double VocabularyCharGPU::distance(const std::vector<unsigned char> &a, const std::vector<unsigned char> &b) const {
        int res = 0;
        for (int i = 0; i < a.size(); ++i) {
            unsigned char temp = a[i] ^ b[i];
            while (temp > 0) {
                res += temp & 1;
                temp >>= 1;
            }
//            res += ones8bits[a[i] ^ b[i]];
        }
        return res;
    }

    void VocabularyCharGPU::HKmeansStepParallelBFS(NodeId parent_id, std::vector<std::vector<unsigned char>> &descriptors,
                                                 int current_level) {


        std::vector<int> idxes;
        idxes.push_back(descriptors[0].size() / m_desc_len);
        int node_num = 0;
        for (int current_level = 0; current_level < m_L; ++current_level) {
            cout << "new level " << current_level << endl;
            int expected_nodes = (int)((pow((double)m_k, (double)current_level + 1) - 1)/(m_k - 1)) -
                                 (int)((pow((double)m_k, (double)current_level) - 1)/(m_k - 1));
            cout << "nodes " << expected_nodes << endl;
            std::vector<std::vector<int>> current_idxes(expected_nodes, std::vector<int>());
          
                parallel_for(0, expected_nodes,
                             [this, &idxes, &descriptors, node_num, current_level, &current_idxes](int current_node) {
                                 int begin = current_node > 0 ? idxes[current_node - 1] : 0;
                                 int end = idxes[current_node];
                                 int temp_node_num = node_num + current_node;
                                 HKmeansIter(descriptors[current_level & 1], descriptors[!(current_level & 1)], begin,
                                             end,
                                             current_idxes[current_node], temp_node_num, current_level);
                             });

            node_num += expected_nodes;
            idxes.clear();
            for (int current_node = 0; current_node < expected_nodes; ++current_node) {
                for (int i = 0; i < current_idxes[current_node].size(); ++i) {
                    idxes.push_back(current_idxes[current_node][i]);
                }
            }

        }
    }


    __global__ void findClosest(unsigned char *desc, unsigned  char *clusters, bool *d_goon,
            unsigned char *association, int desc_num, int desc_len, int blocks_num, int clusters_num) {
        int t_idx = threadIdx.x;
        __shared__ bool goon[512];
        __syncthreads();
        goon[t_idx] = false;
        int grainSize = 100;
        int step = grainSize > desc_num / (blocks_num * blockDim.x) + 1 ? grainSize : desc_num / (blocks_num * blockDim.x) + 1;
        int begin = (blockIdx.x * blockDim.x + threadIdx.x) * step;
        int end = (blockIdx.x * blockDim.x + (threadIdx.x + 1)) * step;
        if (end > desc_num) {
            end = desc_num;
        }
        bool correct = begin < desc_num;

        if (correct) {
            for (int i = begin; i < end; ++i) {
                double best_dist = d_distance(desc + i * desc_len, clusters, desc_len);
                unsigned int icluster = 0;

                for(unsigned int c = 1; c < clusters_num; ++c)
                {
                    double dist = d_distance(desc + i * desc_len, clusters + c * desc_len, desc_len);
                    if(dist < best_dist)
                    {
                        best_dist = dist;
                        icluster = c;
                    }
                }
                if (association[i] != icluster) {

                    association[i] = icluster;
                }
            }
            __syncthreads();
            __shared__ int pow;
            if (t_idx == 0) {
                pow = 1;
            }
            __syncthreads();
            while ((1 << pow) <= blockDim.x) {
                if ((t_idx & ((1 << pow) - 1)) == 0) {
                    goon[t_idx] |= goon[t_idx + (1 << (pow - 1))];
                }
                __syncthreads();
                if (t_idx == 0) {
                    pow++;
                }
                __syncthreads();
            }
            if (t_idx == 0) {
                d_goon[blockIdx.x] = goon[t_idx];
            }
        }
        __syncthreads();
    }

    __global__ void updateClusters(unsigned  char *desc, unsigned char *clusters, unsigned  char *association, int desc_num,
            int desc_len, unsigned int *sums, int clusters_num) {
        int t_idx = threadIdx.x;
        int grainSize = 100;
        int step = grainSize > desc_num / (blockDim.x) + 1 ? grainSize : desc_num / (blockDim.x) + 1;
        int begin = (threadIdx.x) * step;
        int end = (threadIdx.x + 1) * step;
        if (end > desc_num) {
            end = desc_num;
        }

        bool correct = begin < desc_num;
        int my_cluster = blockIdx.x;
        int shift = (blockIdx.x * blockDim.x + t_idx) * (desc_len * 8 + 1);
        unsigned int *t_sums = sums + (blockIdx.x * blockDim.x + t_idx) * (desc_len * 8 + 1);
        if (correct) {
            for (int j = 0; j < desc_len * 8 + 1; ++j) {
//                t_sums[j] = 0;
                sums[shift + j] = 0;
            }
            for (int i = begin; i < end; ++i) {
                if (association[i] == my_cluster) {
                    for (int j = 0; j < desc_len; ++j) {
                        for (int k = 0; k < 8; ++k) {
                            if (desc[i * desc_len + j] & (1 << k)) {
                                sums[shift + j * 8 + (7 - k)]++;
                            }
                        }
                    }
                    sums[shift + 8 * desc_len]++;
                }
            }
            __syncthreads();
            __shared__ int pow;
            if (t_idx == 0) {
                pow = 1;
            }
            __syncthreads();
            while ((1 << pow) <= blockDim.x) {
                if ((t_idx & ((1 << pow) - 1)) == 0) {
                    for (int i = 0; i < (desc_len * 8 + 1); ++i) {
                        sums[shift + i] += sums[shift + (1 << (pow - 1)) * (desc_len * 8 + 1) + i];
                    }
                }
                __syncthreads();
                if (t_idx == 0) {
                    pow++;
                }
                __syncthreads();
            }
            if (t_idx == 0) {
                if (sums[shift + 8 * desc_len] == 0) {
                } else if (sums[shift + 8 * desc_len] == 1) {
                    int idx = -1;
                    for (int i = 0; i < desc_num; ++i) {
                        if (association[i] == my_cluster) {
                            idx = i;
                            break;
                        }
                    }
                    for (int i = 0; i < desc_len; ++i) {
                        clusters[my_cluster * desc_len + i] = desc[idx * desc_len + i];
                    }
                } else {
                    for (int i = 0; i < desc_len; ++i) {
                        clusters[my_cluster * desc_len + i] = 0;
                    }
                    int N2 = sums[shift + 8 * desc_len] / 2 + (sums[shift + 8 * desc_len] % 2);
                    for (int i = 0; i < 8 * desc_len; ++i) {
                        if (sums[shift + i] >= N2) {
                            clusters[my_cluster * desc_len + (i / 8)] |= 1 << (7 - (i % 8));

                        }
                    }
                }
            }
        }

    }


    __global__ void myprint(unsigned char *association, int desc_num) {
        int a[9];
        for (int i = 0; i < 9; ++i) {
            a[i] = 0;
        }
        for (int i = 0; i < desc_num; ++i) {
            a[association[i]]++;
        }
    }
    void VocabularyCharGPU::HKmeansIter(std::vector<unsigned char> &descriptors, std::vector<unsigned char> &new_descriptors, int begin, int end, std::vector<int> &idxs, int node_num, int level) {
        int size = end - begin;
        if(!size) return;
        std::vector<std::vector<unsigned char>> clusters;
        cudaSetDevice(node_num & 1);
        clusters.reserve(m_k);
        int clusters_num = m_k;
        std::vector<vector<std::vector<unsigned char>>> cluster_descriptors(clusters_num);
//        std::vector<unsigned char> last_association(size);

        if(size <= m_k)
        {
            for(unsigned int i = 0; i < size; i++)
            {
                clusters.push_back(std::vector<unsigned char>(descriptors.begin() + m_desc_len * begin + i * m_desc_len, descriptors.begin() + m_desc_len * begin + i * m_desc_len + m_desc_len));
            }
            for (int c = 0; c < size; ++c) {
                m_nodes[m_nodes[node_num].children[c]].descriptor = clusters[c];
                idxs.push_back(begin + 1);
                for (int i = 0; i  < clusters[c].size(); ++i) {
                    new_descriptors[m_desc_len * begin  + i] = descriptors[m_desc_len * begin + i];
                }
                begin++;// += clusters[c].size();
            }
            return;
        }
        else {
            initiateClustersHKpp(std::vector<unsigned char>(descriptors.begin() + m_desc_len * begin,
                                                            descriptors.begin() + m_desc_len * end), clusters);
            clusters_num = clusters.size();
            unsigned  char *desc;
            desc = (unsigned char*)malloc(sizeof(unsigned char)  * size * m_desc_len);
            int clusters_size = clusters.size() * clusters[0].size();
            unsigned char clusters1D[clusters_num * m_desc_len];
            for (int i = 0; i < clusters_num; ++i) {
                for (int j = 0; j < m_desc_len; ++j){
                    clusters1D[i * m_desc_len + j] = clusters[i][j];
                }
            }
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < m_desc_len; ++j) {
                    desc[i * m_desc_len + j] = descriptors[begin * m_desc_len + i * m_desc_len + j];
                }
            }
            unsigned  char *d_desc;
            cudaMalloc((void **)&d_desc, size * m_desc_len * sizeof(unsigned char));
            cudaMemcpy((void *)d_desc, (void *)desc, size * m_desc_len * sizeof(unsigned char), cudaMemcpyHostToDevice);

            unsigned char *d_clusters;
            cudaMalloc((void **)&d_clusters, clusters_num * m_desc_len * sizeof(unsigned char));
            cudaMemcpy((void *)d_clusters, (void *)clusters1D, clusters_num * m_desc_len * sizeof(unsigned char), cudaMemcpyHostToDevice);

            unsigned char *d_association;
            cudaMalloc((void **)&d_association, size * sizeof(unsigned char));
            cudaMemset(d_association, 0, size * sizeof(unsigned char));
            int numBlocks = 16 * 1024 / (1 << (level));
            int threadsPerBlock =  8 * 2 * 32 / (1 << (2 * level)); 
            threadsPerBlock = max(threadsPerBlock, 32);
            numBlocks = max(numBlocks, 2048);
            bool *d_goon;
            cudaMalloc((void **)&d_goon, numBlocks * sizeof(bool));
            cudaMemset(d_goon, 0, numBlocks * sizeof(bool));
            unsigned int *d_sums;
            cudaMalloc((void **)&d_sums, clusters_num * threadsPerBlock * (m_desc_len * 8 + 1) * sizeof(unsigned int));
            bool goon = true;
            int nn = 0;
            for (;goon && nn < 100;) {
                findClosest<<<numBlocks, threadsPerBlock>>>(d_desc, d_clusters, d_goon, d_association, size, m_desc_len, numBlocks, clusters_num);

                cudaDeviceSynchronize();
                cudaError_t error = cudaGetLastError();
                if(error != cudaSuccess)
                {
                    printf("CUDA first error: %s\n%d\t%d\n", cudaGetErrorString(error), node_num, error);
//                    exit(-1);
                }
                bool goons[numBlocks];
                cudaMemcpy((void*)goons, (void *)d_goon, numBlocks * sizeof(bool), cudaMemcpyDeviceToHost);
                goon = false;
                for (int i = 0; i < numBlocks; ++i) {
                    goon |= goons[i];
                }
                updateClusters<<<clusters_num,threadsPerBlock>>>(d_desc, d_clusters, d_association, size, m_desc_len, d_sums, clusters_num);
                cudaDeviceSynchronize();
                error = cudaGetLastError();
                if(error != cudaSuccess)
                {
                    // print the CUDA error message and exit
                    printf("CUDA second error: %s\n%d\t%d\n", cudaGetErrorString(error), node_num, error);
//                    exit(-1);
                }
                nn++;
            }
            unsigned char association[size];
            cudaMemcpy(association, d_association, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess)
            {
                // print the CUDA error message and exit
                printf("CUDA error: %s\n%d\t%d\n", cudaGetErrorString(error), node_num, error);
//                exit(-1);
            }

            cudaMemcpy((void*)clusters1D, (void *)d_clusters, clusters_num * m_desc_len * sizeof(unsigned char), cudaMemcpyDeviceToHost);

            cudaFree(d_desc);
            cudaFree(d_clusters);
            cudaFree(d_association);
            cudaFree(d_sums);
            for (int i = 0; i < clusters.size(); ++i) {
                for (int j = 0; j < clusters[0].size(); ++j){
                    clusters[i][j] = clusters1D[i * clusters[0].size() + j];
                }
            }
            cluster_descriptors.clear();
            cluster_descriptors.resize(clusters_num);
            for (int i = 0; i < size; ++i) {
                std::vector<unsigned char> temp;
                for (int j = 0; j < m_desc_len; ++j) {
                    temp.push_back(descriptors[m_desc_len * begin + i * m_desc_len + j]);
                }
                if (association[i] > clusters_num) {
                    cout << i << " " << (int)association[i] << " " << clusters_num << endl;
                }
                cluster_descriptors[association[i]].push_back(temp);
            }
            for (int c = 0; c < clusters_num; ++c) {
                m_nodes[m_nodes[node_num].children[c]].descriptor = clusters[c];
                idxs.push_back(begin + cluster_descriptors[c].size());
                for (int i = 0; i < cluster_descriptors[c].size(); ++i) {
                    for (int j = 0; j < m_desc_len; ++j) {
                        new_descriptors[m_desc_len * begin + m_desc_len * i + j] = cluster_descriptors[c][i][j];
                    }
                }
                begin += cluster_descriptors[c].size();
            }
            for (int c = clusters_num; c < m_k; ++c) {
                idxs.push_back(idxs.back());
            }
        }
    }

    void VocabularyCharGPU::initiateClustersHKpp(
            const std::vector<unsigned char> &pfeatures,
            std::vector<std::vector<unsigned char>> &clusters) {
        // Implements kmeans++ seeding algorithm
        // Algorithm:
        // 1. Choose one center uniformly at random from among the data points.
        // 2. For each data point x, compute D(x), the distance between x and the nearest
        //    center that has already been chosen.
        // 3. Add one new data point as a center. Each point x is chosen with probability
        //    proportional to D(x)^2.
        // 4. Repeat Steps 2 and 3 until k centers have been chosen.
        // 5. Now that the initial centers have been chosen, proceed using standard k-means
        //    clustering.
        clusters.resize(0);
        clusters.reserve(m_k);
        std::vector<double> min_dists(pfeatures.size() / m_desc_len, std::numeric_limits<double>::max());

        // 1.

        int ifeature = RandomInt(0, (pfeatures.size() / m_desc_len) -1);

        // create first cluster
//        std::vector<unsigned char> init_feature;
//        for (int i = 0; i < m_desc_len; ++i) {
//            init_feature.push_back(pfeatures[m_desc_len * ifeature + i]);
//        }
        clusters.push_back(vector<unsigned char>(pfeatures.begin() + m_desc_len * ifeature,
                                                 pfeatures.begin() + m_desc_len * ifeature + m_desc_len));

        // compute the initial distances
        std::vector<double>::iterator dit;
        dit = min_dists.begin();
        for (int i = 0; i < pfeatures.size() / m_desc_len; ++i) {
            std::vector<unsigned char> temp;
            for (int j = 0; j < m_desc_len; ++j) {
                temp.push_back(pfeatures[m_desc_len * i + j]);
            }
            min_dists[i] = distance(temp, clusters.back());
        }

        while((int)clusters.size() < m_k)
        {
            // 2.
            dit = min_dists.begin();
            for (int i = 0; i < pfeatures.size() / m_desc_len; ++i) {
                if (min_dists[i] > 0) {
                    std::vector<unsigned char> temp;
                    for (int j = 0; j < m_desc_len; ++j) {
                        temp.push_back(pfeatures[m_desc_len * i + j]);
                    }
                    double dist = distance(temp, clusters.back());
                    if (dist < min_dists[i]) {
                        min_dists[i] = dist;
                    }
                }
            }
            // 3.
            double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);

            if(dist_sum > 0)
            {
                cout.precision(17);
//                cout << "dist sum " << dist_sum << endl;
                double cut_d;
                do
                {
                    cut_d = RandomValue<double>(0, dist_sum);
                } while(cut_d == 0.0);

                double d_up_now = 0;
                for(dit = min_dists.begin(); dit != min_dists.end(); ++dit)
                {
                    d_up_now += *dit;
                    if(d_up_now >= cut_d) break;
                }

                if(dit == min_dists.end())
                    ifeature = (pfeatures.size() / m_desc_len)-1;
                else
                    ifeature = dit - min_dists.begin();
                std::vector<unsigned char> new_feature;
                for (int i = 0; i < m_desc_len; ++i) {
                    new_feature.push_back(pfeatures[m_desc_len * ifeature + i]);
                }
                clusters.push_back(new_feature);
//                cout << "ifeature 239 " << ifeature << endl;
            } // if dist_sum > 0
            else
                break;

        } // while(used_clusters < m_k)

    }



    void VocabularyCharGPU::setNodeWeightsParallel(const std::vector<std::vector<unsigned char>> &training_features) {
        const unsigned int NWords = m_words.size();
        const unsigned int NDocs = training_features.size();

        if(m_weighting == TF || m_weighting == BINARY)
        {
            // idf part must be 1 always
            for(unsigned int i = 0; i < NWords; i++)
                m_words[i]->weight = 1;
        }
        else if(m_weighting == IDF || m_weighting == TF_IDF)
        {

        std::vector<unsigned int> Ni(NWords, 0);
        std::vector<tbb::mutex> mutexes(NWords);
        std::vector<bool> counted(NWords, false);
        parallel_for(unsigned(0), NDocs, [&](int img_num) {

            std::vector<bool> counted(NWords, false);

            for(int desc_num = 0; desc_num < training_features[img_num].size() / m_desc_len ; ++desc_num) {
                WordId wordId;
                vector<unsigned char> temp(training_features[img_num].begin() + m_desc_len * desc_num, training_features[img_num].begin() + m_desc_len * (desc_num + 1));
                transform(temp, wordId);

                if (!counted[wordId]) {
                    mutexes[wordId].lock();
                    Ni[wordId]++;
                    mutexes[wordId].unlock();
                    counted[wordId] = true;
                }
            }
        });
        // set ln(N/Ni)
        parallel_for(unsigned(0), NWords, [&](int i) {
            if (Ni[i] > 0) {
                m_words[i]->weight = log((double)NDocs / (double)Ni[i]);
            }
        });

        }
    }
    void VocabularyCharGPU::transform
            (const vector<unsigned char> &feature, WordId &id) const
    {
        WordValue weight;
        transform(feature, id, weight);
    }

    double VocabularyCharGPU::score(const BowVector &v1, const BowVector &v2) const
    {
        return m_scoring_object->score(v1, v2);
    }

    void VocabularyCharGPU::transform(const std::vector<unsigned char>& features, BowVector &v) const
    {
        v.clear();

        if(features.empty())
        {
            return;
        }

        // normalize
        LNorm norm;
        bool must = m_scoring_object->mustNormalize(norm);

//        typename std::vector<TDescriptor>::const_iterator fit;

        if(m_weighting == TF || m_weighting == TF_IDF)
        {
            for (int desc_num = 0; desc_num < features.size() / m_desc_len; ++desc_num) {

                WordId id;
                WordValue w;

                vector<unsigned char> temp(features.begin() + m_desc_len * desc_num, features.begin() + m_desc_len * (desc_num + 1));
                transform(temp, id, w);

                if(w > 0) v.addWeight(id, w);
            }

            if(!v.empty() && !must)
            {
                // unnecessary when normalizing
                const double nd = v.size();
                for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
                    vit->second /= nd;
            }

        }
        else // IDF || BINARY
        {
            for (int desc_num = 0; desc_num < features.size(); ++desc_num) {

                WordId id;
                WordValue w;
                // w is idf if IDF, or 1 if BINARY

                vector<unsigned char> temp(features.begin() + m_desc_len * desc_num, features.begin() + m_desc_len * (desc_num + 1));

                transform(temp, id, w);

                // not stopped
                if(w > 0) v.addIfNotExist(id, w);

            } // if add_features
        } // if m_weighting == ...

        if(must) v.normalize(norm);
    }

    void VocabularyCharGPU::transform(const vector<unsigned char> &feature,
                                    WordId &word_id, WordValue &weight, NodeId *nid, int levelsup) const
    {
        // propagate the feature down the tree
        std::vector<NodeId> nodes, nodes2;
        typename std::vector<NodeId>::const_iterator nit;

        // level at which the node must be stored in nid, if given
        const int nid_level = m_L - levelsup;
        if(nid_level <= 0 && nid != NULL) *nid = 0; // root

        NodeId final_id = 0; // root
        int current_level = 0;
        do
        {
            ++current_level;
            nodes2 = m_nodes[final_id].children;
            nodes.clear();
            for (int i = 0; i < nodes2.size(); ++i) {
                if (m_nodes[nodes2[i]].descriptor.size()) {
                    nodes.push_back(nodes2[i]);
                }
            }
            if (nodes.empty()) {
                break;
            }
            final_id = nodes[0];

            double best_d = distance(feature, m_nodes[final_id].descriptor);
            for(nit = nodes.begin() + 1; nit != nodes.end(); ++nit)
            {
                NodeId id = *nit;

                double d = distance(feature, m_nodes[id].descriptor);
                if(d < best_d)
                {
                    best_d = d;
                    final_id = id;
                }
            }

            if(nid != NULL && current_level == nid_level)
                *nid = final_id;

        } while( !m_nodes[final_id].isLeaf() );
        // turn node id into word id
        word_id = m_nodes[final_id].word_id;
        weight = m_nodes[final_id].weight;
    }

    void VocabularyCharGPU::createScoringObject()
    {
        delete m_scoring_object;
        m_scoring_object = NULL;

        switch(m_scoring)
        {
            case L1_NORM:
                m_scoring_object = new L1Scoring;
                break;

            case L2_NORM:
                m_scoring_object = new L2Scoring;
                break;

            case CHI_SQUARE:
                m_scoring_object = new ChiSquareScoring;
                break;

            case KL:
                m_scoring_object = new KLScoring;
                break;

            case BHATTACHARYYA:
                m_scoring_object = new BhattacharyyaScoring;
                break;

            case DOT_PRODUCT:
                m_scoring_object = new DotProductScoring;
                break;

        }
    }
}
