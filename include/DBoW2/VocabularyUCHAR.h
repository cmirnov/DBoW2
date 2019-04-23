/**
 * File: TemplatedVocabulary.h
 * Date: February 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated vocabulary
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_VOCABULARY_UCHAR__
#define __D_T_VOCABULARY_UCHAR__

#include <cassert>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <algorithm>
#include <queue>
#include <chrono>
#include <opencv2/core.hpp>

#include "FeatureVector.h"
#include "BowVector.h"
#include "ScoringObject.h"
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <tbb/concurrent_vector.h>
#include <tbb/mutex.h>
#include <tbb/tbb_thread.h>

using namespace tbb;

namespace DBoW2 {

    class VocabularyUCHAR
    {
    public:
        /**
         * Initiates an empty vocabulary
         * @param k branching factor
         * @param L depth levels
         * @param weighting weighting type
         * @param scoring scoring type
         */
        VocabularyUCHAR(int k = 10, int L = 5, int grainsize = 1,
                            WeightingType weighting = TF_IDF, ScoringType scoring = L1_NORM);

        /**
         * Destructor
         */
        ~VocabularyUCHAR();


        /**
         * Creates a vocabulary from the training features with the already
         * defined parameters
         * @param training_features
         */
        void create
                (const std::vector<std::vector<uchar> > &training_features);

        void build_tree();

        double distance(const std::vector<uchar> &a, const std::vector<uchar> &b) const;

        void meanValue(const tbb::concurrent_vector<std::vector<uchar>> &descriptors,
                       std::vector<uchar> &mean);

        /**
         * Transforms a set of descriptores into a bow vector
         * @param features
         * @param v (out) bow vector of weighted words
         */
        void transform(const std::vector<uchar>& features, BowVector &v)
        const;

        /**
         * Transform a set of descriptors into a bow vector and a feature vector
         * @param features
         * @param v (out) bow vector
         * @param fv (out) feature vector of nodes and feature indexes
         * @param levelsup levels to go up the vocabulary tree to get the node index
         */
        void transform(const std::vector<uchar>& features,
                       WordId &id, WordValue &weight, NodeId* nid = NULL, int levelsup = 0) const;


        void transform(const std::vector<uchar> &feature, WordId &id) const;

        double score(const BowVector &a, const BowVector &b) const;

    protected:


        /// Tree node
        struct Node
        {
            /// Node id
            NodeId id;
            /// Weight if the node is a word
            WordValue weight;
            /// Children
            std::vector<NodeId> children;
            /// Parent node (undefined in case of root)
            NodeId parent;
            /// Node descriptor
            std::vector<uchar> descriptor;

            /// Word id if the node is a word
            WordId word_id;

            /**
             * Empty constructor
             */
            Node(): id(0), weight(0), parent(0), word_id(0){}

            /**
             * Constructor
             * @param _id node id
             */
            Node(NodeId _id): id(_id), weight(0), parent(0), word_id(0){}

            /**
             * Returns whether the node is a leaf node
             * @return true iff the node is a leaf
             */
            inline bool isLeaf() const { return children.empty(); }
        };

    protected:

        /**
         * Creates an instance of the scoring object accoring to m_scoring
         */
        void createScoringObject();

        /**
         * Returns a set of pointers to descriptores
         * @param training_features all the features
         * @param features (out) pointers to the training features
         */
        void getFeatures(
                const std::vector<std::vector<uchar> > &training_features,
                std::vector<uchar> &features) const;

        /**
         * Creates a level in the tree, under the parent, by running kmeans with
         * a descriptor set, and recursively creates the subsequent levels too
         * @param parent_id id of parent node
         * @param descriptors descriptors to run the kmeans on
         * @param current_level current level in the tree
         */
        void HKmeansStep(NodeId parent_id, const std::vector<uchar> &descriptors,
                         int current_level);

        void HKmeansStepParallelBFS(NodeId parent_id, std::vector<std::vector<uchar>> &descriptors,
                                    int current_level);
        void HKmeansStepParallelDFS(NodeId parent_id, std::vector<uchar> &descriptors,
                                    int begin, int end);
        void HKmeansIter(std::vector<uchar> &descriptors, std::vector<uchar> &new_descriptors, int begin, int end, std::vector<int> &idxs, int node_num);


        void kmeansIter(const std::vector<uchar> &descriptors,
                        std::vector<uchar> &clusters, std::vector<concurrent_vector<unsigned int>> &groups) const;


        void setNodeWeightsParallel(const std::vector<std::vector<uchar>> &training_features);

        /**
         * Creates k clusters from the given descriptors with some seeding algorithm.
         * @note In this class, kmeans++ is used, but this function should be
         *   overriden by inherited classes.
         */
        void initiateClustersHKpp(const std::vector<uchar> &descriptors,
                                      std::vector<std::vector<uchar>> &clusters);

        /**
         * Returns a random number in the range [min..max]
         * @param min
         * @param max
         * @return random T number in [min..max]
         */
        template <class T>
        static T RandomValue(T min, T max){
            return ((T)rand()/(T)RAND_MAX) * (max - min) + min;
        }

        /**
         * Returns a random int in the range [min..max]
         * @param min
         * @param max
         * @return random int in [min..max]
         */
        static int RandomInt(int min, int max){
            int d = max - min + 1;
            return int(((double)rand()/((double)RAND_MAX + 1.0)) * d) + min;
        }

    protected:

        /// Branching factor
        int m_k;

        /// Depth levels
        int m_L;

        /// Weighting method
        WeightingType m_weighting;

        /// Scoring method
        ScoringType m_scoring;

        /// Object for computing scores
        GeneralScoring* m_scoring_object;

        /// Tree nodes
        std::vector<Node> m_nodes;

        /// Descriptor length
        int m_desc_len = 32;
        int m_grainsize;

        /// Words of the vocabulary (tree leaves)
        /// this condition holds: m_words[wid]->word_id == wid
        std::vector<Node*> m_words;

    };

} // namespace DBoW2

#endif
