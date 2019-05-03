//
// Created by kirill on 30.04.19.
//

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <tbb/parallel_for.h>
#include "VocabularyCharGPU.h"
#include <cooperative_groups.h>

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
//        cout << "train " << training_features.size() << endl;
        features.resize(0);
        for (auto v : training_features) {
            for (auto f : v) {
                features.push_back(f);
            }
        }
//        cout << "res " << features.size() << endl;
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
            std::vector<std::vector<int>> current_idxes(expected_nodes, std::vector<int>());
            parallel_for(0, expected_nodes,
                         [this, &idxes, &descriptors, node_num, current_level, &current_idxes](int current_node) {
                             int begin = current_node > 0 ? idxes[current_node - 1] : 0;
                             int end = idxes[current_node];
                             int temp_node_num = node_num + current_node;
                             HKmeansIter(descriptors[current_level & 1], descriptors[!(current_level & 1)], begin,
                                         end,
                                         current_idxes[current_node], temp_node_num);
                         });
//            for (int current_node = 0; current_node < expected_nodes; ++current_node) {
////                            cout << "next node\n";
//                             int begin = current_node > 0 ? idxes[current_node - 1] : 0;
//                             int end = idxes[current_node];
//                             cout << begin << " " << end << endl;
//                             int temp_node_num = node_num + current_node;
//                             HKmeansIter(descriptors[current_level & 1], descriptors[!(current_level & 1)], begin,
//                                         end,
//                                         current_idxes[current_node], temp_node_num);
//            }
            node_num += expected_nodes;
            idxes.clear();
            for (int current_node = 0; current_node < expected_nodes; ++current_node) {
                for (int i = 0; i < current_idxes[current_node].size(); ++i) {
                    idxes.push_back(current_idxes[current_node][i]);
                }
            }

        }
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
    __global__ void HKmeansIterGPU(int desc_num, int desc_len, int cluster_num, unsigned char *descriptors, unsigned char *clusters,
                                   unsigned int * last_association, unsigned int *current_association, unsigned int *sums, int blocks_num, int *goon) {
//        __shared__ bool goon;
        __shared__ int pow;
        cg::thread_block cta = cg::this_thread_block();
//        if (threadIdx.x == 0) {
//            goon = true;
//        }
        bool first_time = true;
        int grainSize = 100;
        int step = grainSize > desc_num / (blocks_num * blockDim.x) + 1 ? grainSize : desc_num / (blocks_num * blockDim.x) + 1;
        int begin = (blockIdx.x * blockDim.x + threadIdx.x) * step;
        int end = (blockIdx.x * blockDim.x + (threadIdx.x + 1)) * step;
        if (end > desc_num) {
            end = desc_num;
        }
        int t_idx = threadIdx.x;
        bool correct = begin < desc_num;
//
        cta.sync();
        unsigned int *t_sums = sums + (blockIdx.x * blockDim.x * cluster_num + threadIdx.x * cluster_num) * (desc_len * 8 + 1);
        while (*goon) {
            for (int i = 0; i < cluster_num * (desc_len * 8 + 1) && correct; ++i) {
                t_sums[i] = 0;
            }
            for (int i = begin; i < end && correct; ++i) {
//                printf("%d\n", threadIdx.x);
                unsigned char *temp;
                unsigned char *cluster_temp;
                temp = descriptors + i * desc_len;

                cluster_temp = clusters;
                double best_dist = d_distance(temp, cluster_temp, desc_len);
//                printf("best dist\n");
                unsigned int icluster = 0;
                cluster_temp += desc_len;
                for (unsigned int c = 1; c < cluster_num; ++c) {
                    double dist = d_distance(temp, cluster_temp, desc_len);
                    if (dist < best_dist) {
                        best_dist = dist;
                        icluster = c;
                    }
                    cluster_temp += desc_len;
                }
                unsigned int *c_t_sums = t_sums + icluster * (desc_len * 8 + 1);
                c_t_sums[desc_len * 8]++;
                current_association[i] = icluster;
                for (int j = 0; j < desc_len; ++j) {

                    const unsigned char cur = *temp;
                    if (cur &
                        (1 << 7))
                        ++c_t_sums[j * 8];
                    if (cur &
                        (1 << 6))
                        ++c_t_sums[j * 8 + 1];
                    if (cur &
                        (1 << 5))
                        ++c_t_sums[j * 8 + 2];
                    if (cur &
                        (1 << 4))
                        ++c_t_sums[j * 8 + 3];
                    if (cur &
                        (1 << 3))
                        ++c_t_sums[j * 8 + 4];
                    if (cur &
                        (1 << 2))
                        ++c_t_sums[j * 8 + 5];
                    if (cur &
                        (1 << 1))
                        ++c_t_sums[j * 8 + 6];
                    if (cur &
                        (1))
                        ++c_t_sums[j * 8 + 7];
                    temp++;
                }
            }

            if (t_idx == 0) {
//                printf("here\n");
                pow = 1;
            }
            __syncthreads();
            while ((1 << pow) <= blockDim.x) {
                if (t_idx & ((1 << pow) - 1) == 0) {
                    unsigned int *sums_a;
                    sums_a = sums + (blockIdx.x * blockDim.x * cluster_num + t_idx * cluster_num) * (desc_len * 8 + 1);
                    unsigned int *sums_b;
                    sums_b = sums + (blockIdx.x * blockDim.x * cluster_num +
                             (t_idx + (1 << (pow - 1))) * cluster_num) * (desc_len * 8 + 1);
                    for (int i = 0; i < cluster_num * (desc_len * 8 + 1); ++i) {
                        sums_a[i] += sums_b[i];
                    }
                }
                __syncthreads();
                if (t_idx == 0) {
                    pow++;
                }
                __syncthreads();
            }
            cta.sync();
            if (blockIdx.x == 0) {
                if (threadIdx.x == 0) {
//                    printf("sadly we are here\n");
                    for (int p = 1; p < blocks_num; ++p) {
                        for (int i = 0; i < cluster_num * (desc_len * 8 + 1); ++i) {
//                            if ( sums[p * blockDim.x  * cluster_num * (desc_len * 8 + 1)+ i] != 0) {
//                                printf("fuck\n");
//                            }
                            sums[i] += sums[p * blockDim.x  * cluster_num * (desc_len * 8 + 1)+ i];
//                            printf("%d\n", sums[p * blockDim.x  * cluster_num * (desc_len * 8 + 1) + i]);
                        }
                    }
                }
//                __syncthreads();
                if (t_idx < cluster_num) {
                    int c = t_idx;
//                for (int c = 0; c < cluster_num; ++c) {

                    if (sums[c * (desc_len * 8 + 1) + desc_len * 8] == 0) {
//                    clusters[c].clear();
                    } else if (sums[c * (desc_len * 8 + 1) + desc_len * 8] == 1) {
//                    clusters[c].clear();
                        int idx = -1;
                        for (int i = 0; i < desc_num; ++i) {
                            if (current_association[i] == c) {
                                idx = i;
                                break;
                            }
                        }
                        for (int i = 0; i < desc_len; ++i) {
                            clusters[c * desc_len + i] = descriptors[idx * desc_len + i];
                        }
                    } else {
                        for (int i = 0; i < desc_len; ++i) {
                            clusters[c * desc_len + i] = 0;
                        }
                        int cluster_size = sums[c * (desc_len * 8 + 1) + 8 * desc_len];
                        const int N2 = (int) cluster_size / 2 + cluster_size % 2;
                        int idx = 0;
                        for (size_t i = 0; i < 8 * desc_len; ++i) {
                            if (sums[c * (desc_len * 8 + 1) + i] >= N2) {
                                // set bit
                                clusters[c * desc_len + idx] |= 1 << (7 - (i % 8));
                            }

                            if (i % 8 == 7) {
                                ++idx;
                            }
                        }
                    }
                }

                if (t_idx == 0) {
                    *goon = false;
                    for (int i = 0; i < desc_num; ++i) {
                        if (last_association[i] != current_association[i]) {
                             *goon = true;
                            break;
                        }
                    }
                    for (int i = 0; i < desc_num; ++i) {
                        last_association[i] = current_association[i];
                    }
                }
                __syncthreads();
            }
            cta.sync();
        }
    }

    void VocabularyCharGPU::HKmeansIter(std::vector<unsigned char> &descriptors, std::vector<unsigned char> &new_descriptors, int begin, int end, std::vector<int> &idxs, int node_num) {
//        cout << node_num << endl;
        int size = end - begin;
        if(!size) return;
        std::vector<std::vector<unsigned char>> clusters;

        clusters.reserve(m_k);
        int clusters_num = m_k;
        std::vector<vector<std::vector<unsigned char>>> cluster_descriptors(clusters_num);
        std::vector<int> last_association(size);

        if(size <= m_k)
        {
//            cout << "small\n";
            for(unsigned int i = 0; i < size; i++)
            {
//                std::vector<unsigned char> new_cluster;
//                for (int j = 0; j < m_desc_len; ++j) {
//                    new_cluster.push_back(descriptors[m_desc_len * begin + i * m_desc_len + j]);
//                }
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
//            cout << endl << endl;
            unsigned  char *desc;
            desc = (unsigned char*)malloc(sizeof(unsigned char)  * size * m_desc_len);
            int clusters_size = clusters.size() * clusters[0].size();
//            cout << clusters_size << endl;
            unsigned char clusters1D[clusters_size];
            for (int i = 0; i < clusters.size(); ++i) {
                for (int j = 0; j < clusters[i].size(); ++j){
                    clusters1D[i * clusters[0].size() + j] = clusters[i][j];
                }
            }
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < m_desc_len; ++j) {
                    desc[i * m_desc_len + j] = descriptors[begin * m_desc_len + i * m_desc_len + j];
                }
            }
            unsigned  char *d_desc;
            cudaMalloc(&d_desc, size * m_desc_len * sizeof(unsigned char));
            cudaMemcpy((void *)d_desc, (void *)desc, size * m_desc_len * sizeof(unsigned char), cudaMemcpyHostToDevice);
            unsigned char *d_clusters;
            cudaMalloc(&d_clusters, clusters_size * sizeof(unsigned char));
            cudaMemcpy((void *)d_clusters, (void *)clusters1D, clusters_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
            unsigned int *d_last_association;
            cudaMalloc(&d_last_association, size * sizeof(unsigned int));
            unsigned int *d_current_association;
            cudaMalloc(&d_current_association, size * sizeof(unsigned int));
            int numBlocks = 1;
            int threadsPerBlock = (1 << 10);
            unsigned int *d_sums;
            cudaMalloc(&d_sums, numBlocks * threadsPerBlock * clusters_num * (m_desc_len * 8 + 1) * sizeof(unsigned int));
            int goon[] = {true};
            int *d_goon;
            cudaMalloc(&d_goon, sizeof(int));
            cudaMemcpy((void *)d_goon, (void *)goon, sizeof(char), cudaMemcpyHostToDevice);

//            cout << "start kernel\n";
            HKmeansIterGPU<<<numBlocks,threadsPerBlock>>>(size, m_desc_len, clusters_num, d_desc, d_clusters, d_last_association, d_current_association, d_sums, numBlocks, d_goon);
//            cudaLaunchCooperativeKernel(HKmeansIterGPU,numBlocks, threadsPerBlock,kernelArgs );
//            HKmeansIterGPU<<<numBlocks,threadsPerBlock>>>(size, m_desc_len, clusters_num, d_desc, d_clusters, d_last_association, d_current_association, d_sums, d_mutex1, d_mutex2, numBlocks);
            cudaDeviceSynchronize();
//            cout << "done kernel\n";
            unsigned int *association;
            association = (unsigned int*)malloc(size * sizeof(unsigned int));
            cudaMemcpy((void *)association, (void *)d_last_association, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

            auto res = cudaMemcpy((void*)clusters1D, (void *)d_clusters, clusters_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
            cudaFree(d_desc);
            cudaFree(d_clusters);
            cudaFree(d_last_association);
            cudaFree(d_current_association);
            cudaFree(d_sums);
            for (int i = 0; i < clusters.size(); ++i) {
                for (int j = 0; j < clusters[i].size(); ++j){
                    clusters[i][j] = clusters1D[i * clusters[0].size() + j];
                }
            }
            last_association = vector<int>(association, association + size);
            cluster_descriptors.clear();
            cluster_descriptors.resize(clusters_num);
            for (int i = 0; i < last_association.size(); ++i) {
                std::vector<unsigned char> temp;
                for (int j = 0; j < m_desc_len; ++j) {
                    temp.push_back(descriptors[m_desc_len * begin + i * m_desc_len + j]);
                }
                cluster_descriptors[last_association[i]].push_back(temp);
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
//            cout << "finish\n";
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
            // IDF and TF-IDF: we calculte the idf path now

            // Note: this actually calculates the idf part of the tf-idf score.
            // The complete tf-idf score is calculated in ::transform

            std::vector<unsigned int> Ni(NWords, 0);
            std::vector<bool> counted(NWords, false);

            typename std::vector<std::vector<unsigned char>>::const_iterator mit;
            typename std::vector<unsigned char>::const_iterator fit;

            for (int img_num = 0; img_num < training_features.size(); ++img_num) {
                fill(counted.begin(), counted.end(), false);

                for (int desc_num = 0; desc_num < training_features[img_num].size() / m_desc_len; ++desc_num) {
                    WordId word_id;
                    vector<unsigned char> temp(training_features[img_num].begin() + m_desc_len * desc_num, training_features[img_num].begin() + m_desc_len * (desc_num + 1));
                    transform(temp, word_id);

                    if(!counted[word_id])
                    {
                        Ni[word_id]++;
                        counted[word_id] = true;
                    }
                }
            }

            // set ln(N/Ni)
            for(unsigned int i = 0; i < NWords; i++)
            {
                if(Ni[i] > 0)
                {
                    m_words[i]->weight = log((double)NDocs / (double)Ni[i]);
                }// else // This cannot occur if using kmeans++
            }

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
//            for(fit = features.begin(); fit < features.end(); ++fit)
//            {

                WordId id;
                WordValue w;
                // w is idf if IDF, or 1 if BINARY

//                transform(*fit, id, w);
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