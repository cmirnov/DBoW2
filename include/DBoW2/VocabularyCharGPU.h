//
// Created by kirill on 30.04.19.
//

#ifndef PROJECT_VOCABULARYCHARGPU_H
#define PROJECT_VOCABULARYCHARGPU_H

#include "BowVector.h"
#include "ScoringObject.h"
#include "FeatureVector.h"

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

namespace DBoW2 {

    class VocabularyCharGPU {
    public:
        void test();

    public:
        VocabularyCharGPU(int k = 10, int L = 5, int grainsize = 1,
                        WeightingType weighting = TF_IDF, ScoringType scoring = L1_NORM, int desc_len = 32);

        ~VocabularyCharGPU();

        void create
                (const std::vector<std::vector<unsigned char> > &training_features);

        void build_tree();

        double distance(const std::vector<unsigned char> &a, const std::vector<unsigned char> &b) const;

//        void meanValue(const tbb::concurrent_vector<std::vector<unsigned char>> &descriptors,
//                       std::vector<unsigned char> &mean);

        void transform(const std::vector<unsigned char>& features, BowVector &v)
        const;

        void transform(const std::vector<unsigned char>& features,
                       WordId &id, WordValue &weight, NodeId* nid = NULL, int levelsup = 0) const;


        void transform(const std::vector<unsigned char> &feature, WordId &id) const;

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
            std::vector<unsigned char> descriptor;

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

        void createScoringObject();

        void getFeatures(
                const std::vector<std::vector<unsigned char> > &training_features,
                std::vector<unsigned char> &features) const;


        void HKmeansStepParallelBFS(NodeId parent_id, std::vector<std::vector<unsigned char>> &descriptors,
                                    int current_level);
        void HKmeansStepParallelDFS(NodeId parent_id, std::vector<unsigned char> &descriptors,
                                    int begin, int end);
        void HKmeansIter(std::vector<unsigned char> &descriptors, std::vector<unsigned char> &new_descriptors, int begin, int end, std::vector<int> &idxs, int node_num);



        void setNodeWeightsParallel(const std::vector<std::vector<unsigned char>> &training_features);

        void initiateClustersHKpp(const std::vector<unsigned char> &descriptors,
                                  std::vector<std::vector<unsigned char>> &clusters);

        template <class T>
        static T RandomValue(T min, T max){
            return ((T)rand()/(T)RAND_MAX) * (max - min) + min;
        }

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
}
#endif //PROJECT_VOCABULARYCHARGPU_H
