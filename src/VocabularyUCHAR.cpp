//
// Created by kirill on 20.04.19.
//

#include <vector>
#include <string>
#include <sstream>

#include "VocabularyUCHAR.h"

using namespace std;

namespace DBoW2 {

    VocabularyUCHAR::VocabularyUCHAR(int k, int L, WeightingType weighting, ScoringType scoring)
            : m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring),
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
        build_tree();
        std::cout << "build tree\n";
        std::vector<uchar > features;
        getFeatures(training_features, features);
        std::cout << "features: " << training_features.size() << std::endl;
////    // create the tree
    HKmeansStepParallelBFS(0, features, 1);
//        HKmeansStepParallelDFS(0, features, 0, features.size());
////    HKmeansStep(0, features, 1);
////
////    // create the words
//        createWords();
////
////    // and set the weight of each node of the tree
//        setNodeWeightsParallel(training_features);

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
//        typename std::vector<std::vector<uchar > >::const_iterator vvit;
//        typename std::vector<uchar >::const_iterator vit;
//        for(vvit = training_features.begin(); vvit != training_features.end(); ++vvit)
//        {
//            features.reserve(features.size() + vvit->size());
//            for(vit = vvit->begin(); vit != vvit->end(); ++vit)
//            {
//                features.push_back(&(*vit));
//            }
//        }
    }

    void VocabularyUCHAR::HKmeansStep(NodeId parent_id, const std::vector<uchar> &descriptors,
                     int current_level) {}

    void VocabularyUCHAR::HKmeansStepParallelBFS(NodeId parent_id, std::vector<uchar> &descriptors,
                                int current_level) {}

    void VocabularyUCHAR::HKmeansStepParallelDFS(NodeId parent_id, std::vector<uchar> &descriptors,
                                int begin, int end) {}

    void VocabularyUCHAR::HKmeansIter(std::vector<uchar> &descriptors, int begin, int end, std::vector<int> &idxs, int node_num) {}


    void VocabularyUCHAR::kmeansIter(const std::vector<uchar> &descriptors,
                    std::vector<uchar> &clusters, std::vector<concurrent_vector<unsigned int>> &groups) const {}


    void VocabularyUCHAR::setNodeWeightsParallel(const std::vector<std::vector<uchar> > &features) {}

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