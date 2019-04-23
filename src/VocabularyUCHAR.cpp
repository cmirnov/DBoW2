//
// Created by kirill on 20.04.19.
//

#include <vector>
#include <string>
#include <sstream>

#include "VocabularyUCHAR.h"
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include "ones8bits.h"
using namespace std;

namespace DBoW2 {

    VocabularyUCHAR::VocabularyUCHAR(int k, int L, int grainsize, WeightingType weighting, ScoringType scoring)
            : m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring), m_grainsize(grainsize),
              m_scoring_object(NULL)
    {
        createScoringObject();
    }

    VocabularyUCHAR::~VocabularyUCHAR() {
        delete m_scoring_object;
    }

    void VocabularyUCHAR::create
            (const std::vector<std::vector<uchar> > &training_features) {
        m_nodes.clear();
        m_words.clear();
//        auto statrt = std::chrono::high_resolution_clock::now();
        build_tree();
//        auto end = std::chrono::high_resolution_clock::now();
//        cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - statrt).count() << endl;
        std::vector<std::vector<uchar >> features(2);
//        statrt = std::chrono::high_resolution_clock::now();
        getFeatures(training_features, features[0]);
//        end = std::chrono::high_resolution_clock::now();
//        cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - statrt).count() << endl;
        features[1].resize(features[0].size());
//        statrt = std::chrono::high_resolution_clock::now();
        HKmeansStepParallelBFS(0, features, 1);
//        end = std::chrono::high_resolution_clock::now();
//        cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - statrt).count() << endl;
////        HKmeansStepParallelDFS(0, features, 0, features.size());
//
//        statrt = std::chrono::high_resolution_clock::now();
        setNodeWeightsParallel(training_features);
//        end = std::chrono::high_resolution_clock::now();
//        cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - statrt).count() << endl;

    }

    void VocabularyUCHAR::build_tree() {
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

    void VocabularyUCHAR::getFeatures(
            const std::vector<std::vector<uchar> > &training_features,
            std::vector<uchar> &features) const
    {
        features.resize(0);
        for (auto v : training_features) {
            for (auto f : v) {
                features.push_back(f);
            }
        }
    }

    void VocabularyUCHAR::meanValue(const tbb::concurrent_vector<std::vector<uchar>> &descriptors,
                   std::vector<uchar> &mean) {

        if(descriptors.empty())
        {
            mean.clear();
            return;
        }
        else if(descriptors.size() == 1)
        {
            mean.clear();
            for (int i = 0; i < m_desc_len; ++i) {
                mean.push_back(descriptors[0][i]);
            }
        }
        else {
            vector<int> sum(m_desc_len * 8, 0);
            for (int i = 0; i < descriptors.size(); ++i) {
                for (int j = 0; j < m_desc_len; ++j) {
                    if (descriptors[i][j] & (1 << 7)) ++sum[j * 8];
                    if (descriptors[i][j] & (1 << 6)) ++sum[j * 8 + 1];
                    if (descriptors[i][j] & (1 << 5)) ++sum[j * 8 + 2];
                    if (descriptors[i][j] & (1 << 4)) ++sum[j * 8 + 3];
                    if (descriptors[i][j] & (1 << 3)) ++sum[j * 8 + 4];
                    if (descriptors[i][j] & (1 << 2)) ++sum[j * 8 + 5];
                    if (descriptors[i][j] & (1 << 1)) ++sum[j * 8 + 6];
                    if (descriptors[i][j] & (1)) ++sum[j * 8 + 7];
                }
            }
            mean = vector<uchar>(m_desc_len, 0);
            const int N2 = (int) (descriptors.size()) / 2 + (descriptors.size()) % 2;
            int idx = 0;
            for (size_t i = 0; i < sum.size(); ++i) {
                if (sum[i] >= N2) {
                    // set bit
                    mean[idx] |= 1 << (7 - (i % 8));
                }

                if (i % 8 == 7) ++idx;
            }
        }
    }

    // should check faster approach
    double VocabularyUCHAR::distance(const std::vector<uchar> &a, const std::vector<uchar> &b) const {
        int res = 0;
        for (int i = 0; i < a.size(); ++i) {
            res += ones8bits[a[i] ^ b[i]];
        }
        return res;
    }

    void VocabularyUCHAR::HKmeansStep(NodeId parent_id, const std::vector<uchar> &descriptors,
                     int current_level) {}

    void VocabularyUCHAR::HKmeansStepParallelBFS(NodeId parent_id, std::vector<std::vector<uchar>> &descriptors,
                                int current_level) {
        std::vector<int> idxes;
        idxes.push_back(descriptors[0].size() / m_desc_len);
        int node_num = 0;
        for (int current_level = 0; current_level < m_L; ++current_level) {
//            cout << current_level << endl;
            int expected_nodes = (int)((pow((double)m_k, (double)current_level + 1) - 1)/(m_k - 1)) -
                                 (int)((pow((double)m_k, (double)current_level) - 1)/(m_k - 1));
//            task_group g;
            std::vector<std::vector<int>> current_idxes(expected_nodes, std::vector<int>());
//            int begin = 0;
//            int end = idxes[0];
            parallel_for(0, expected_nodes, [this, &idxes, &descriptors, node_num, current_level, &current_idxes](int current_node) {
                int begin = current_node > 0 ? idxes[current_node - 1] : 0;
                int end = idxes[current_node];
                int temp_node_num = node_num + current_node;
//            for (int current_node = 0; current_node < expected_nodes; ++current_node) {
                HKmeansIter(descriptors[current_level & 1], descriptors[!(current_level & 1)], begin, end,
                            current_idxes[current_node], temp_node_num);

//                end = idxes[current_node];
//                g.run([this, &descriptors, current_level, begin, end, &current_idxes, current_node, node_num] {
//                    HKmeansIter(descriptors[current_level & 1], descriptors[!(current_level & 1)], begin, end,
//                                current_idxes[current_node], node_num);
//                });
//
//
//                begin = end;
//                node_num++;
//            }
//            g.wait();
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

    void VocabularyUCHAR::HKmeansStepParallelDFS(NodeId parent_id, std::vector<uchar> &descriptors,
                                int begin, int end) {}

    void VocabularyUCHAR::HKmeansIter(std::vector<uchar> &descriptors, std::vector<uchar> &new_descriptors, int begin, int end, std::vector<int> &idxs, int node_num) {
        int size = end - begin;
//        cout << "kmeansiter " << begin <<  " " << end << endl;
        if(!size) return;
        // features associated to each cluster
        std::vector<std::vector<uchar>> clusters;

        clusters.reserve(m_k);

        int clusters_num = m_k;
        std::vector<vector<std::vector<uchar>>> cluster_descriptors(clusters_num);

        if(size <= m_k)
        {
//            cout << "kek\n";
            for(unsigned int i = 0; i < size; i++)
            {
                std::vector<uchar> new_cluster;
                for (int j = 0; j < m_desc_len; ++j) {
                    new_cluster.push_back(descriptors[m_desc_len * begin + i * m_desc_len + j]);
                }
                clusters.push_back(new_cluster);
            }
//            cout << "between\n";
            for (int c = 0; c < size; ++c) {
                m_nodes[m_nodes[node_num].children[c]].descriptor = clusters[c];
                idxs.push_back(begin + 1);
                for (int i = 0; i  < clusters[c].size(); ++i) {
                    new_descriptors[m_desc_len * begin  + i] = descriptors[m_desc_len * begin + i];
                }
                begin++;// += clusters[c].size();
            }
//            cout << "chpek\n";
            return;
        }
        else {

            initiateClustersHKpp(std::vector<uchar>(descriptors.begin() + m_desc_len * begin,
                                                    descriptors.begin() + m_desc_len * end), clusters);

            bool goon = true;
            bool first_time = true;
            std::vector<int> last_association(size), current_association(size);
//            int grainsize = 20;
//            vector<int> clustart_size(clusters_num, 0);
            while (goon) {
//                cout << "Goon\n";
//                clustart_size.clear();
//                clustart_size.resize(clusters_num, 0);
//                typedef tbb::blocked_range<vector<const unsigned char*>::iterator> range_type;
                typedef tbb::blocked_range<vector<const unsigned char *>::iterator> range_type;
//                cout << "first\n";
                auto sums = tbb::parallel_reduce(tbb::blocked_range<int>(0, size, m_grainsize),
                                                 vector<vector<int>>(clusters_num, vector<int>(m_desc_len * 8 + 1, 0)),
                                                 [this, &clusters, &descriptors, &current_association, begin](
                                                         blocked_range<int> r, vector<vector<int>> sums) {
                                                     for (int i = r.begin(); i < r.end(); ++i) {
                                                         std::vector<uchar> temp;
                                                         for (int j = 0; j < m_desc_len; ++j) {
                                                             temp.push_back(
                                                                     descriptors[begin * m_desc_len + i * m_desc_len +
                                                                                 j]);
                                                         }
                                                         double best_dist = distance(temp, clusters[0]);
                                                         unsigned int icluster = 0;

                                                         for (unsigned int c = 1; c < clusters.size(); ++c) {
                                                             double dist = distance(temp, clusters[c]);
                                                             if (dist < best_dist) {
                                                                 best_dist = dist;
                                                                 icluster = c;
                                                             }
                                                         }
                                                         current_association[i] = icluster;
                                                         sums[icluster].back()++;
                                                         for (int j = 0; j < m_desc_len; ++j) {
                                                             if (descriptors[begin * m_desc_len + i * m_desc_len + j] &
                                                                 (1 << 7))
                                                                 ++sums[icluster][j * 8];
                                                             if (descriptors[begin * m_desc_len + i * m_desc_len + j] &
                                                                 (1 << 6))
                                                                 ++sums[icluster][j * 8 + 1];
                                                             if (descriptors[begin * m_desc_len + i * m_desc_len + j] &
                                                                 (1 << 5))
                                                                 ++sums[icluster][j * 8 + 2];
                                                             if (descriptors[begin * m_desc_len + i * m_desc_len + j] &
                                                                 (1 << 4))
                                                                 ++sums[icluster][j * 8 + 3];
                                                             if (descriptors[begin * m_desc_len + i * m_desc_len + j] &
                                                                 (1 << 3))
                                                                 ++sums[icluster][j * 8 + 4];
                                                             if (descriptors[begin * m_desc_len + i * m_desc_len + j] &
                                                                 (1 << 2))
                                                                 ++sums[icluster][j * 8 + 5];
                                                             if (descriptors[begin * m_desc_len + i * m_desc_len + j] &
                                                                 (1 << 1))
                                                                 ++sums[icluster][j * 8 + 6];
                                                             if (descriptors[begin * m_desc_len + i * m_desc_len + j] &
                                                                 (1))
                                                                 ++sums[icluster][j * 8 + 7];
                                                         }
                                                     }
                                                     return sums;
                                                 },
                                                 [this, clusters_num](vector<vector<int>> a,
                                                                      vector<vector<int>> b) -> vector<vector<int>> {
                                                     vector<vector<int>> res(clusters_num,
                                                                             vector<int>(m_desc_len * 8 + 1, 0));
                                                     for (int i = 0; i < a.size(); ++i) {
                                                         for (int j = 0; j < a[i].size(); ++j) {
                                                             res[i][j] = a[i][j] + b[i][j];
                                                         }
                                                     }
                                                     return res;
                                                 }
                );
//                vector<vector<int>> sum = tbb::parallel_reduce(range_type(descriptors.begin() +, descriptors2.end()),
////            descriptors2,
//                                                       vector<int>(FORB::L * 8, 0),
//                                                       [](range_type& desc, vector<int> sum)->vector<int>{
//
//                                                           for(auto it = desc.begin(); it != desc.end(); ++it)
//                                                           {
//                                                               const unsigned char *p = (*it);
//                                                               for(int j = 0; j < FORB::L; ++j, ++p)
//                                                               {
//                                                                   if(*p & (1 << 7)) ++sum[ j*8     ];
//                                                                   if(*p & (1 << 6)) ++sum[ j*8 + 1 ];
//                                                                   if(*p & (1 << 5)) ++sum[ j*8 + 2 ];
//                                                                   if(*p & (1 << 4)) ++sum[ j*8 + 3 ];
//                                                                   if(*p & (1 << 3)) ++sum[ j*8 + 4 ];
//                                                                   if(*p & (1 << 2)) ++sum[ j*8 + 5 ];
//                                                                   if(*p & (1 << 1)) ++sum[ j*8 + 6 ];
//                                                                   if(*p & (1))      ++sum[ j*8 + 7 ];
//                                                               }
//                                                           }
//                                                           return sum;
//                                                       },
//                                                       [](vector<int> a, vector<int> b)->vector<int> {
//                                                           vector<int> sum(FORB::L * 8, 0);
//                                                           for (int i = 0; i < FORB::L * 8; ++i) {
//                                                               sum[i] = a[i] + b[i];
//                                                           }
//                                                           return sum;
//                                                       });
//                tbb::parallel_for(0,size, [&](int i) {
//                    std::vector<uchar> temp;
//                    for (int j = 0; j < m_desc_len; ++j) {
//                        temp.push_back(descriptors[m_desc_len * begin + i * m_desc_len + j]);
//                    }
//                    double best_dist = distance(temp, clusters[0]);
//                    unsigned int icluster = 0;
//
//                    for (unsigned int c = 1; c < clusters.size(); ++c) {
//                        double dist = distance(temp, clusters[c]);
//                        if (dist < best_dist) {
//                            best_dist = dist;
//                            icluster = c;
//                        }
//                    }
//                    current_association[i] = icluster;
//                    cluster_descriptors[icluster].push_back(temp);
//                });
//                cout << "second\n";
                tbb::parallel_for(0, clusters_num, [&](int c) {
                    if(sums[c].back() == 0)
                    {
                        clusters[c].clear();
                    }
                    else if(sums[c].back() == 1)
                    {
                        clusters[c].clear();
                        int idx = -1;
                        for (int i = 0; i < current_association.size(); ++i) {
                            if (current_association[i] == c) {
                                idx = i;
                                break;
                            }
                        }
                        for (int i = 0; i < m_desc_len; ++i) {
                            clusters[c].push_back(descriptors[m_desc_len * begin + idx * m_desc_len + i]);
                        }
                    } else {
                        clusters[c] = vector<uchar>(m_desc_len, 0);
                        int cluster_size = sums[c].back();
                        const int N2 = (int) cluster_size / 2 + cluster_size % 2;
                        int idx = 0;
                        for (size_t i = 0; i < sums[c].size() - 1; ++i) {
                            if (sums[c][i] >= N2) {
                                // set bit
                                clusters[c][idx] |= 1 << (7 - (i % 8));
                            }

                            if (i % 8 == 7) {
                                ++idx;
                            }
                        }
                    }
//                    meanValue(cluster_descriptors[c], clusters[c]);
                });
//                cout << "third\n";
                if (!first_time) {
                    goon = false;
                    for (int i = 0; i < size; ++i) {
                        if (last_association[i] != current_association[i]) {
//                            cout << last_association[i] << " " << current_association[i] << endl;
                            goon = true;
                            break;
                        }
                    }
                } else {
                    first_time = false;
                }
                last_association = current_association;
            }
//            cout << "done2\n";
            for (int i = 0; i < last_association.size(); ++i) {
                std::vector<uchar> temp;
                for (int j = 0; j < m_desc_len; ++j) {
                    temp.push_back(descriptors[m_desc_len * begin + i * m_desc_len + j]);
                }
                cluster_descriptors[last_association[i]].push_back(temp);
            }
//            cout << "start\n";
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
//            cout << "finish\n";
        }
    }

    void VocabularyUCHAR::initiateClustersHKpp(
            const std::vector<uchar> &pfeatures,
            std::vector<std::vector<uchar>> &clusters) {
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
        std::vector<uchar> init_feature;
        for (int i = 0; i < m_desc_len; ++i) {
            init_feature.push_back(pfeatures[m_desc_len * ifeature + i]);
        }
        clusters.push_back(init_feature);

        // compute the initial distances
        std::vector<double>::iterator dit;
        dit = min_dists.begin();
        for (int i = 0; i < pfeatures.size() / m_desc_len; ++i) {
            std::vector<uchar> temp;
            for (int j = 0; j < m_desc_len; ++j) {
                temp.push_back(m_desc_len * i + j);
            }
            min_dists[i] = distance(temp, clusters.back());
        }

        while((int)clusters.size() < m_k)
        {
            // 2.
            dit = min_dists.begin();
            for (int i = 0; i < pfeatures.size() / m_desc_len; ++i) {
                if (min_dists[i] > 0) {
                    std::vector<uchar> temp;
                    for (int j = 0; j < m_desc_len; ++j) {
                        temp.push_back(m_desc_len * i + j);
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
                std::vector<uchar> new_feature;
                for (int i = 0; i < m_desc_len; ++i) {
                    new_feature.push_back(m_desc_len * ifeature + i);
                }
                clusters.push_back(new_feature);

            } // if dist_sum > 0
            else
                break;

        } // while(used_clusters < m_k)

    }

    void VocabularyUCHAR::kmeansIter(const std::vector<uchar> &descriptors,
                    std::vector<uchar> &clusters, std::vector<concurrent_vector<unsigned int>> &groups) const {}


    void VocabularyUCHAR::setNodeWeightsParallel(const std::vector<std::vector<uchar>> &training_features) {
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

            typename std::vector<std::vector<uchar>>::const_iterator mit;
            typename std::vector<uchar>::const_iterator fit;

            for (int img_num = 0; img_num < training_features.size(); ++img_num) {
                fill(counted.begin(), counted.end(), false);

                for (int desc_num = 0; desc_num < training_features[img_num].size() / m_desc_len; ++desc_num) {
                    WordId word_id;
                    vector<uchar> temp(training_features[img_num].begin() + m_desc_len * desc_num, training_features[img_num].begin() + m_desc_len * (desc_num + 1));
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
    void VocabularyUCHAR::transform
            (const vector<uchar> &feature, WordId &id) const
    {
        WordValue weight;
        transform(feature, id, weight);
    }

    double VocabularyUCHAR::score(const BowVector &v1, const BowVector &v2) const
    {
        return m_scoring_object->score(v1, v2);
    }

    void VocabularyUCHAR::transform(const std::vector<uchar>& features, BowVector &v) const
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

                vector<uchar> temp(features.begin() + m_desc_len * desc_num, features.begin() + m_desc_len * (desc_num + 1));
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
//            for(fit = features.begin(); fit < features.end(); ++fit)
//            {
//                WordId id;
//                WordValue w;
//                // w is idf if IDF, or 1 if BINARY
//
//                transform(*fit, id, w);
//
//                // not stopped
//                if(w > 0) v.addIfNotExist(id, w);
//
//            } // if add_features
        } // if m_weighting == ...

        if(must) v.normalize(norm);
    }

    void VocabularyUCHAR::transform(const vector<uchar> &feature,
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

    void VocabularyUCHAR::createScoringObject()
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