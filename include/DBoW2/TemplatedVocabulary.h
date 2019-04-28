/**
 * File: TemplatedVocabulary.h
 * Date: February 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated vocabulary 
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_TEMPLATED_VOCABULARY__
#define __D_T_TEMPLATED_VOCABULARY__

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
#include <iostream>

#include "FeatureVector.h"
#include "BowVector.h"
#include "ScoringObject.h"
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <tbb/concurrent_vector.h>
#include <tbb/mutex.h>
#include <tbb/tbb_thread.h>

//#include "VocabularyUCHAR.h"
#include <tbb/parallel_reduce.h>

using namespace tbb;

namespace DBoW2 {

/// @param TDescriptor class of descriptor
/// @param F class of descriptor functions
template<class TDescriptor, class F>
/// Generic Vocabulary
class TemplatedVocabulary
{		
public:
  
  /**
   * Initiates an empty vocabulary
   * @param k branching factor
   * @param L depth levels
   * @param weighting weighting type
   * @param scoring scoring type
   */
  TemplatedVocabulary(int k = 10, int L = 5, 
    WeightingType weighting = TF_IDF, ScoringType scoring = L1_NORM);
  
  /**
   * Creates the vocabulary by loading a file
   * @param filename
   */
  TemplatedVocabulary(const std::string &filename);
  
  /**
   * Creates the vocabulary by loading a file
   * @param filename
   */
  TemplatedVocabulary(const char *filename);
  
  /** 
   * Copy constructor
   * @param voc
   */
  TemplatedVocabulary(const TemplatedVocabulary<TDescriptor, F> &voc);
  
  /**
   * Destructor
   */
  virtual ~TemplatedVocabulary();
  
  /** 
   * Assigns the given vocabulary to this by copying its data and removing
   * all the data contained by this vocabulary before
   * @param voc
   * @return reference to this vocabulary
   */
  TemplatedVocabulary<TDescriptor, F>& operator=(
    const TemplatedVocabulary<TDescriptor, F> &voc);
  
  /** 
   * Creates a vocabulary from the training features with the already
   * defined parameters
   * @param training_features
   */
  virtual void create
    (const std::vector<std::vector<TDescriptor> > &training_features);
  
  /**
   * Creates a vocabulary from the training features, setting the branching
   * factor and the depth levels of the tree
   * @param training_features
   * @param k branching factor
   * @param L depth levels
   */
  virtual void create
    (const std::vector<std::vector<TDescriptor> > &training_features, 
      int k, int L);

  /**
   * Creates a vocabulary from the training features, setting the branching
   * factor nad the depth levels of the tree, and the weighting and scoring
   * schemes
   */
  virtual void create
    (const std::vector<std::vector<TDescriptor> > &training_features,
      int k, int L, WeightingType weighting, ScoringType scoring);

  /**
   * Creaets a vocabulary with preallocated tree
   * @param training_features
   * @param k
   * @param L
   * @param weighting
   * @param scoring
   */

    virtual void create2
            (const std::vector<std::vector<TDescriptor> > &training_features,
             int k, int L, WeightingType weighting, ScoringType scoring);

    virtual void create2
            (const std::vector<std::vector<TDescriptor> > &training_features, int grainsize);

    virtual void build_tree();

  /**
   * Returns the number of words in the vocabulary
   * @return number of words
   */
  virtual inline unsigned int size() const;
  
  /**
   * Returns whether the vocabulary is empty (i.e. it has not been trained)
   * @return true iff the vocabulary is empty
   */
  virtual inline bool empty() const;

  /**
   * Transforms a set of descriptores into a bow vector
   * @param features
   * @param v (out) bow vector of weighted words
   */
  virtual void transform(const std::vector<TDescriptor>& features, BowVector &v) 
    const;
  
  /**
   * Transform a set of descriptors into a bow vector and a feature vector
   * @param features
   * @param v (out) bow vector
   * @param fv (out) feature vector of nodes and feature indexes
   * @param levelsup levels to go up the vocabulary tree to get the node index
   */
  virtual void transform(const std::vector<TDescriptor>& features,
    BowVector &v, FeatureVector &fv, int levelsup) const;

  /**
   * Transforms a single feature into a word (without weight)
   * @param feature
   * @return word id
   */
  virtual WordId transform(const TDescriptor& feature) const;
  
  /**
   * Returns the score of two vectors
   * @param a vector
   * @param b vector
   * @return score between vectors
   * @note the vectors must be already sorted and normalized if necessary
   */
  inline double score(const BowVector &a, const BowVector &b) const;
  
  /**
   * Returns the id of the node that is "levelsup" levels from the word given
   * @param wid word id
   * @param levelsup 0..L
   * @return node id. if levelsup is 0, returns the node id associated to the
   *   word id
   */
  virtual NodeId getParentNode(WordId wid, int levelsup) const;
  
  /**
   * Returns the ids of all the words that are under the given node id,
   * by traversing any of the branches that goes down from the node
   * @param nid starting node id
   * @param words ids of words
   */
  void getWordsFromNode(NodeId nid, std::vector<WordId> &words) const;
  
  /**
   * Returns the branching factor of the tree (k)
   * @return k
   */
  inline int getBranchingFactor() const { return m_k; }
  
  /** 
   * Returns the depth levels of the tree (L)
   * @return L
   */
  inline int getDepthLevels() const { return m_L; }
  
  /**
   * Returns the real depth levels of the tree on average
   * @return average of depth levels of leaves
   */
  float getEffectiveLevels() const;
  
  /**
   * Returns the descriptor of a word
   * @param wid word id
   * @return descriptor
   */
  virtual inline TDescriptor getWord(WordId wid) const;
  
  /**
   * Returns the weight of a word
   * @param wid word id
   * @return weight
   */
  virtual inline WordValue getWordWeight(WordId wid) const;
  
  /** 
   * Returns the weighting method
   * @return weighting method
   */
  inline WeightingType getWeightingType() const { return m_weighting; }
  
  /** 
   * Returns the scoring method
   * @return scoring method
   */
  inline ScoringType getScoringType() const { return m_scoring; }
  
  /**
   * Changes the weighting method
   * @param type new weighting type
   */
  inline void setWeightingType(WeightingType type);
  
  /**
   * Changes the scoring method
   * @param type new scoring type
   */
  void setScoringType(ScoringType type);
  
  /**
   * Saves the vocabulary into a file
   * @param filename
   */
  void save(const std::string &filename) const;
  
  /**
   * Loads the vocabulary from a file
   * @param filename
   */
  void load(const std::string &filename);
  
  /** 
   * Saves the vocabulary to a file storage structure
   * @param fn node in file storage
   */
  virtual void save(cv::FileStorage &fs, 
    const std::string &name = "vocabulary") const;
  
  /**
   * Loads the vocabulary from a file storage node
   * @param fn first node
   * @param subname name of the child node of fn where the tree is stored.
   *   If not given, the fn node is used instead
   */  
  virtual void load(const cv::FileStorage &fs, 
    const std::string &name = "vocabulary");
  
  /** 
   * Stops those words whose weight is below minWeight.
   * Words are stopped by setting their weight to 0. There are not returned
   * later when transforming image features into vectors.
   * Note that when using IDF or TF_IDF, the weight is the idf part, which
   * is equivalent to -log(f), where f is the frequency of the word
   * (f = Ni/N, Ni: number of training images where the word is present, 
   * N: number of training images).
   * Note that the old weight is forgotten, and subsequent calls to this 
   * function with a lower minWeight have no effect.
   * @return number of words stopped now
   */
  virtual int stopWords(double minWeight);

protected:

  /// Pointer to descriptor
  typedef const TDescriptor *pDescriptor;

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
    TDescriptor descriptor;

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
    const std::vector<std::vector<TDescriptor> > &training_features,
    std::vector<pDescriptor> &features) const;

  /**
   * Returns the word id associated to a feature
   * @param feature
   * @param id (out) word id
   * @param weight (out) word weight
   * @param nid (out) if given, id of the node "levelsup" levels up
   * @param levelsup
   */
  virtual void transform(const TDescriptor &feature, 
    WordId &id, WordValue &weight, NodeId* nid = NULL, int levelsup = 0) const;

  /**
   * Returns the word id associated to a feature
   * @param feature
   * @param id (out) word id
   */
  virtual void transform(const TDescriptor &feature, WordId &id) const;
      
  /**
   * Creates a level in the tree, under the parent, by running kmeans with
   * a descriptor set, and recursively creates the subsequent levels too
   * @param parent_id id of parent node
   * @param descriptors descriptors to run the kmeans on
   * @param current_level current level in the tree
   */
  void HKmeansStep(NodeId parent_id, const std::vector<pDescriptor> &descriptors,
    int current_level);

  void HKmeansStepParallelBFS(NodeId parent_id, std::vector<pDescriptor> &descriptors,
          int current_level);
    void HKmeansStepParallelDFS(NodeId parent_id, std::vector<pDescriptor> &descriptors,
                            int begin, int end);
    void HKmeansIter(std::vector<pDescriptor> &descriptors, int begin, int end, std::vector<int> &idxs, int node_num);


        void kmeansIter(const std::vector<pDescriptor> &descriptors,
                          std::vector<TDescriptor> &clusters, std::vector<concurrent_vector<unsigned int>> &groups) const;


    void setNodeWeightsParallel(const std::vector<std::vector<TDescriptor> > &features);

  /**
   * Creates k clusters from the given descriptors with some seeding algorithm.
   * @note In this class, kmeans++ is used, but this function should be
   *   overriden by inherited classes.
   */
  virtual void initiateClusters(const std::vector<pDescriptor> &descriptors,
    std::vector<TDescriptor> &clusters) const;
  
  /**
   * Creates k clusters from the given descriptor sets by running the
   * initial step of kmeans++
   * @param descriptors 
   * @param clusters resulting clusters
   */
  void initiateClustersKMpp(const std::vector<pDescriptor> &descriptors,
    std::vector<TDescriptor> &clusters) const;
  
  /**
   * Create the words of the vocabulary once the tree has been built
   */
  void createWords();
  
  /**
   * Sets the weights of the nodes of tree according to the given features.
   * Before calling this function, the nodes and the words must be already
   * created (by calling HKmeansStep and createWords)
   * @param features
   */
  void setNodeWeights(const std::vector<std::vector<TDescriptor> > &features);
  
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

    int m_grainsize;

    /// Weighting method
  WeightingType m_weighting;
  
  /// Scoring method
  ScoringType m_scoring;
  
  /// Object for computing scores
  GeneralScoring* m_scoring_object;
  
  /// Tree nodes
  std::vector<Node> m_nodes;

  tbb::concurrent_vector<Node> m_nodes_concur;

//  concurrent_vector<task_group> g;
  
  /// Words of the vocabulary (tree leaves)
  /// this condition holds: m_words[wid]->word_id == wid
  std::vector<Node*> m_words;
  mutex kmeansMutex;
    mutex taskMutex;
    mutex initMutex;
  
};

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary
  (int k, int L, WeightingType weighting, ScoringType scoring)
  : m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring),
  m_scoring_object(NULL)
{
  createScoringObject();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary
  (const std::string &filename): m_scoring_object(NULL)
{
  load(filename);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary
  (const char *filename): m_scoring_object(NULL)
{
  load(filename);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::createScoringObject()
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

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::setScoringType(ScoringType type)
{
  m_scoring = type;
  createScoringObject();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::setWeightingType(WeightingType type)
{
  this->m_weighting = type;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary(
  const TemplatedVocabulary<TDescriptor, F> &voc)
  : m_scoring_object(NULL)
{
  *this = voc;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::~TemplatedVocabulary()
{
  delete m_scoring_object;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor, F>& 
TemplatedVocabulary<TDescriptor,F>::operator=
  (const TemplatedVocabulary<TDescriptor, F> &voc)
{  
  this->m_k = voc.m_k;
  this->m_L = voc.m_L;
  this->m_scoring = voc.m_scoring;
  this->m_weighting = voc.m_weighting;

  this->createScoringObject();
  
  this->m_nodes.clear();
  this->m_words.clear();
  
  this->m_nodes = voc.m_nodes;
  this->createWords();
  
  return *this;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::create(
  const std::vector<std::vector<TDescriptor> > &training_features)
{
  m_nodes.clear();
  m_words.clear();
	int expected_nodes =
		(int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));

  m_nodes.reserve(expected_nodes); // avoid allocations when creating the tree


  std::vector<pDescriptor> features;
  getFeatures(training_features, features);


  // create root
  m_nodes.push_back(Node(0)); // root
  // create the tree
  HKmeansStep(0, features, 1);

  // create the words
  createWords();

  // and set the weight of each node of the tree
//  setNodeWeights(training_features);
  setNodeWeights(training_features);

}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::create(
  const std::vector<std::vector<TDescriptor> > &training_features,
  int k, int L)
{
  m_k = k;
  m_L = L;
  
  create(training_features);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::create(
  const std::vector<std::vector<TDescriptor> > &training_features,
  int k, int L, WeightingType weighting, ScoringType scoring)
{
  m_k = k;
  m_L = L;
  m_weighting = weighting;
  m_scoring = scoring;
  createScoringObject();
  
  create(training_features);
}

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::create2(
        const std::vector<std::vector<TDescriptor> > &training_features,
        int k, int L, WeightingType weighting, ScoringType scoring)
{
    m_k = k;
    m_L = L;
    m_weighting = weighting;
    m_scoring = scoring;
    createScoringObject();
    create(training_features);
}

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::build_tree() {
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
using namespace std;
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::create2(
        const std::vector<std::vector<TDescriptor> > &training_features, int grainsize)
{
    m_grainsize = grainsize;
    m_nodes.clear();
    m_words.clear();
    build_tree();
    std::vector<pDescriptor> features;
    cout << "1\n";
    getFeatures(training_features, features);
    cout << "2\n";
    HKmeansStepParallelBFS(0, features, 1);
    cout << "3\n";
//        HKmeansStepParallelDFS(0, features, 0, features.size());
    setNodeWeightsParallel(training_features);
//    cout << "4\n";
    for (int i = 1; i < m_nodes.size(); ++i) {
//            cout << "5\n";
            const cv::Mat &d2 = m_nodes[i].descriptor;
//            cout << "6\n";
//            cout << d2.cols << " " << d2.rows << endl;
            vector<unsigned char> temp2(d2.begin<unsigned char>(), d2.end<unsigned char>());
//            cout << "7\n";
            for (int j = 0; j < temp2.size(); ++j) {
                cout << (int)temp2[j] << " ";
            }
            cout << endl;
    }
}

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::kmeansIter
        (const std::vector<pDescriptor> &descriptors,
         std::vector<TDescriptor> &clusters, std::vector<concurrent_vector<unsigned int>> &groups) const {
    groups.resize(clusters.size());
    bool goon = true;
    bool first_time = true;
    int descriptors_num = descriptors.size();
    int clusters_num = clusters.size();
    std::vector<int> last_association(descriptors_num), current_association(descriptors_num);
    while(goon) {
        std::vector<concurrent_vector<pDescriptor>> cluster_descriptors(clusters_num);
        groups.resize(clusters_num);
        tbb::parallel_for(0, descriptors_num, [&](int i) {
            double best_dist = F::distance(*descriptors[i], clusters[0]);
            unsigned int icluster = 0;

            for (unsigned int c = 1; c < clusters.size(); ++c) {
                double dist = F::distance(*descriptors[i], clusters[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    icluster = c;
                }
            }
            current_association[i] = icluster;
            groups[icluster].push_back(i);
            cluster_descriptors[icluster].push_back(descriptors[i]);
        });

        tbb::parallel_for(0, clusters_num, [&] (int c) {
            F::meanValue(cluster_descriptors[c], clusters[c]);
        });

        if (!first_time){
            goon = false;
            for(int i = 0; i < descriptors_num; ++i) {
                if (last_association[i] != current_association[i]) {
                    goon = true;
                    break;
                }
            }
        } else {
            first_time = false;
        }
        last_association = current_association;
    }
}

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::HKmeansIter(std::vector<pDescriptor> &descriptors, int begin, int end, std::vector<int> &idxs, int node_num) {
    if (node_num > 2) {
        return;
    }
//    for (int i = 0; i < descriptors.size(); ++i) {
//        const cv::Mat &d2 = *descriptors[i];
//        vector<unsigned char> temp2(d2.begin<unsigned char>(), d2.end<unsigned char>());
//        for (int j = 0; j < 32; ++j) {
//            cout << (int)temp2[j] << " ";
//        }
//        cout << endl;
//    }
//    throw 1;
    int size = end - begin;
//    cout << "desc size " << descriptors.size() << endl;
    if(!size) return;
    // features associated to each cluster
    std::vector<TDescriptor> clusters;

    clusters.reserve(m_k);

    int descriptors_num = descriptors.size();
        int clusters_num = m_k;
    std::vector<concurrent_vector<pDescriptor>> cluster_descriptors(clusters_num, concurrent_vector<pDescriptor>());
    if(size <= m_k)
    {
        for(unsigned int i = 0; i < size; i++)
        {
            clusters.push_back(*descriptors[begin + i]);
        }
        for (int c = 0; c < size; ++c) {
            m_nodes[m_nodes[node_num].children[c]].descriptor = clusters[c];
            idxs.push_back(begin + 1);
            begin++;
        }
        for (int i = size; i < m_k; ++i) {
            idxs.push_back(idxs.back());
        }
        return;
    }
    else {
        int grainsize =  85;
        initiateClusters(std::vector<pDescriptor>(descriptors.begin() + begin, descriptors.begin() + end), clusters);
        clusters_num = clusters.size();
        cout << endl;
        cout << "Im working but changed\n";

//        throw 1;
        bool goon = true;
        bool first_time = true;
        std::vector<int> last_association(size), current_association(size);
        int nn = 0;
        while(goon) {
            nn++;
            for (int i = 0; i < clusters.size(); ++i) {
                const cv::Mat &d2 = clusters[i];
                vector<unsigned char> temp2(d2.begin<unsigned char>(), d2.end<unsigned char>());
                for (int j = 0; j < 32; ++j) {
                    cout << (int)temp2[j] << " ";
                }
                cout << endl;
            }
            cout << endl;
            current_association.clear();
            current_association.resize(size);

            auto sums = tbb::parallel_reduce(tbb::blocked_range<int>(0, size, 73),
                                             vector<vector<int>>(clusters.size(), vector<int>(32 * 8 + 1, 0)),
                                             [this, &clusters, &descriptors, &current_association, begin](
                                                     blocked_range<int> r, vector<vector<int>> sums) {
                                                 for (int i = r.begin(); i < r.end(); ++i) {
                                                     double best_dist = F::distance(*descriptors[begin + i], clusters[0]);
                                                     unsigned int icluster = 0;

                                                     for (unsigned int c = 1; c < clusters.size(); ++c) {

                                                            double dist = F::distance(*descriptors[begin + i],
                                                                                      clusters[c]);
                                                            if (dist < best_dist) {
                                                                best_dist = dist;
                                                                icluster = c;
                                                            }
                                                     }
                                                     current_association[i] = icluster;
                                                     sums[icluster].back()++;
                                                     const cv::Mat &d = *descriptors[begin + i];
                                                     const unsigned char *p = d.ptr<unsigned char>();

                                                     for(int j = 0; j < d.cols; ++j, ++p)
                                                     {
                                                         if(*p & (1 << 7)) ++sums[icluster][ j*8     ];
                                                         if(*p & (1 << 6)) ++sums[icluster][ j*8 + 1 ];
                                                         if(*p & (1 << 5)) ++sums[icluster][ j*8 + 2 ];
                                                         if(*p & (1 << 4)) ++sums[icluster][ j*8 + 3 ];
                                                         if(*p & (1 << 3)) ++sums[icluster][ j*8 + 4 ];
                                                         if(*p & (1 << 2)) ++sums[icluster][ j*8 + 5 ];
                                                         if(*p & (1 << 1)) ++sums[icluster][ j*8 + 6 ];
                                                         if(*p & (1))      ++sums[icluster][ j*8 + 7 ];
                                                     }
                                                 }
                                                 return sums;
                                             },
                                             [this, clusters_num](vector<vector<int>> a,
                                                                  vector<vector<int>> b) -> vector<vector<int>> {
                                                 for (int i = 0; i < clusters_num; ++i) {
                                                     for (int j = 0; j < a[i].size(); ++j) {
                                                         a[i][j] += b[i][j];
                                                     }
                                                 }
                                                 return a;
                                             }
            );

//            cout << "sums size " << sums.size() << endl;
//            for (int i = 0; i < sums.size(); ++i) {
//                for (int j = 0; j < sums[i].size(); ++j) {
//                    cout << sums[i][j] << " ";
//                }
//                cout << endl << endl;
//            }
//            cout << endl;
//            cout << "association\n";
//            for (int i = 0; i < current_association.size(); ++i) {
//                cout << current_association[i] << " ";
//            }
//            cout << endl;
            tbb::parallel_for(0, clusters_num, [&](int c) {
//            for (int c = 0; c < clusters_num; ++c) {
                if (sums[c].back() == 0) {
                    clusters[c].release();
                } else if (sums[c].back() == 1) {
                    cout << "one\n";
                    int idx = -1;
                    for (int i = 0; i < current_association.size(); ++i) {
                        if (current_association[i] == c) {
                            idx = i;
                            break;
                        }
                    }
                    clusters[c] = descriptors[begin + idx]->clone();
                } else {
                    clusters[c] = cv::Mat::zeros(1, 32, CV_8U);//vector<uchar>(32, 0);
                    unsigned char *p = clusters[c].template ptr<unsigned char>();
                    int cluster_size = sums[c].back();
                    const int N2 = (int) cluster_size / 2 + cluster_size % 2;
                    for (size_t i = 0; i < sums[c].size() - 1; ++i) {
                        if (sums[c][i] >= N2) {
                            // set bit
                            *p |= 1 << (7 - (i % 8));
                        }

                        if (i % 8 == 7) ++p;
                    }
                }
//                    meanValue(cluster_descriptors[c], clusters[c]);
            });
//            }

            if (!first_time){
                goon = false;
                for(int i = 0; i < size; ++i) {
                    if (last_association[i] != current_association[i]) {
                        goon = true;
                        break;
                    }
                }
            } else {
                first_time = false;
            }
            last_association = current_association;
            if (nn == 3) {
                throw 1;
            }
        }
        cout << "norm";
        throw 1;
        cluster_descriptors.clear();
        cluster_descriptors.resize(clusters_num, concurrent_vector<pDescriptor>());

        vector<int> ttt(clusters_num, 0);
        for (int i = 0; i < last_association.size(); ++i) {
            ttt[last_association[i]]++;
            cout << last_association[i] << " ";
            cluster_descriptors[last_association[i]].push_back(descriptors[begin + i]);
        }
        cout << endl;
        for (auto x : ttt) {
            cout << x << endl;
        }
        throw 1;
        for (int c = 0; c < clusters_num; ++c) {
            m_nodes[m_nodes[node_num].children[c]].descriptor = clusters[c];
            idxs.push_back(begin + cluster_descriptors[c].size());
            for (int i = 0; i < cluster_descriptors[c].size(); ++i) {
                descriptors[begin + i] = cluster_descriptors[c][i];
            }
            begin += cluster_descriptors[c].size();
        }
        for (int c = clusters_num; c < m_k; ++c){
            idxs.push_back(idxs.back());
        }
    }
    if (node_num == 2) {
        throw 1;
    }
}


template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::HKmeansStepParallelDFS(NodeId parent_id,
                                                                std::vector<pDescriptor> &descriptors, int begin, int end)
{
    if (m_nodes[parent_id].isLeaf()) {
        return;
    }
    std::vector<int> idxes;
    int node_num = 0;
    task_group g;
    HKmeansIter(descriptors, begin, end, idxes, parent_id);
    int t_begin = begin;
    int t_end;
    for (int current_node  = 0; current_node < m_k; ++current_node) {
        t_end = idxes[current_node];
        int child_id = m_nodes[parent_id].children[current_node];
        g.run([this, child_id, &descriptors, t_begin, t_end]{HKmeansStepParallelDFS(child_id, descriptors, t_begin, t_end);});

        t_begin = t_end;
    }
    g.wait();
}

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::HKmeansStepParallelBFS(NodeId parent_id,
                                                     std::vector<pDescriptor> &descriptors, int current_level)
{

    std::vector<int> idxes;
    idxes.push_back(descriptors.size());
    int node_num = 0;
    for (int current_level = 0; current_level < m_L; ++current_level) {
        int expected_nodes = (int)((pow((double)m_k, (double)current_level + 1) - 1)/(m_k - 1)) -
                                             (int)((pow((double)m_k, (double)current_level) - 1)/(m_k - 1));

        std::vector<std::vector<int>> current_idxes(expected_nodes, std::vector<int>());

        parallel_for(0, expected_nodes, [this, &idxes, &descriptors, node_num, current_level, &current_idxes](int current_node) {
            int begin = current_node > 0 ? idxes[current_node - 1] : 0;
            int end = idxes[current_node];
            int temp_node_num = node_num + current_node;
            HKmeansIter(descriptors, begin, end,
                        current_idxes[current_node], temp_node_num);
        });
        node_num += expected_nodes;
//        throw 1;
        idxes.clear();
        for (int current_node = 0; current_node < expected_nodes; ++current_node) {
            for (int i = 0; i < current_idxes[current_node].size(); ++i) {
                idxes.push_back(current_idxes[current_node][i]);
            }
        }
    }
}

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::setNodeWeightsParallel
        (const std::vector<std::vector<TDescriptor> > &training_features)
{
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
//        std::cout << "begin\n";
        std::vector<unsigned int> Ni(NWords, 0);
        std::vector<tbb::mutex> mutexes(NWords);
        std::vector<bool> counted(NWords, false);
        parallel_for(unsigned(0), NDocs, [&](int i) {

            std::vector<bool> counted(NWords, false);

            for(int j = 0; j < training_features[i].size(); ++j) {
                WordId wordId;
                transform(training_features[i][j], wordId);

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


// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::getFeatures(
  const std::vector<std::vector<TDescriptor> > &training_features,
  std::vector<pDescriptor> &features) const
{
  features.resize(0);
  typename std::vector<std::vector<TDescriptor> >::const_iterator vvit;
  typename std::vector<TDescriptor>::const_iterator vit;
  for(vvit = training_features.begin(); vvit != training_features.end(); ++vvit)
  {
//      cout << "vvit size:  " << vvit->size() << endl;
    features.reserve(features.size() + vvit->size());
    for(vit = vvit->begin(); vit != vvit->end(); ++vit)
    {
      features.push_back(&(*vit));
    }
  }
}

// --------------------------------------------------------------------------
//45548250190
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::HKmeansStep(NodeId parent_id,
  const std::vector<pDescriptor> &descriptors, int current_level) {
  if(descriptors.empty()) return;

  // features associated to each cluster
  std::vector<TDescriptor> clusters;
  std::vector<std::vector<unsigned int> > groups; // groups[i] = [j1, j2, ...]
	// j1, j2, ... indices of descriptors associated to cluster i

  clusters.reserve(m_k);
	groups.reserve(m_k);

  if((int)descriptors.size() <= m_k)
  {
    // trivial case: one cluster per feature
    groups.resize(descriptors.size());

    for(unsigned int i = 0; i < descriptors.size(); i++)
    {
      groups[i].push_back(i);
      clusters.push_back(*descriptors[i]);
    }
  }
  else
  {
    // select clusters and groups with kmeans
//      auto statrt = std::chrono::high_resolution_clock::now();

    bool first_time = true;
    bool goon = true;

    // to check if clusters move after iterations
    std::vector<int> last_association, current_association;
    int num = 0;
    while(goon)
    {
        num++;
      // 1. Calculate clusters

//        kmeansMutex.lock();
      if(first_time)
      {
        // random sample
        kmeansMutex.lock();
        initiateClusters(descriptors, clusters);
        kmeansMutex.unlock();
      }
      else
      {
        // calculate cluster centres

//          auto start = std::chrono::high_resolution_clock::now();
        for(unsigned int c = 0; c < clusters.size(); ++c)
        {
          std::vector<pDescriptor> cluster_descriptors;
          cluster_descriptors.reserve(groups[c].size());


          std::vector<unsigned int>::const_iterator vit;
          for(vit = groups[c].begin(); vit != groups[c].end(); ++vit)
          {
            cluster_descriptors.push_back(descriptors[*vit]);
          }

          kmeansMutex.lock();
          std::cout << "mean\n";
          F::meanValue(cluster_descriptors, clusters[c]);
          kmeansMutex.unlock();
        }

      } // if(!first_time)

      // 2. Associate features with clusters

      groups.clear();
      groups.resize(clusters.size(), std::vector<unsigned int>());
      current_association.resize(descriptors.size());

      typename std::vector<pDescriptor>::const_iterator fit;

          for (fit = descriptors.begin(); fit != descriptors.end(); ++fit)//, ++d)
//          for (int i = 0; i < descriptors.size(); ++i)
        {
          double best_dist = F::distance(*(*fit), clusters[0]);
          unsigned int icluster = 0;
          for (unsigned int c = 1; c < clusters.size(); ++c) {
            double dist = F::distance(*(*fit), clusters[c]);
            if (dist < best_dist) {
              best_dist = dist;
              icluster = c;
            }
          }
          groups[icluster].push_back(fit - descriptors.begin());
          current_association[fit - descriptors.begin()] = icluster;
        }

      // kmeans++ ensures all the clusters has any feature associated with them

      // 3. check convergence
      if(first_time)
      {
        first_time = false;
      }
      else
      {
        //goon = !eqUChar(last_assoc, assoc);

        goon = false;
        for(unsigned int i = 0; i < current_association.size(); i++)
        {
          if(current_association[i] != last_association[i]){
            goon = true;
            break;
          }
        }
      }

      if(goon)
      {
          // copy last feature-cluster association
          last_association = current_association;
          //last_assoc = assoc.clone();
      }

    } // while(goon)
//      auto end = std::chrono::high_resolution_clock::now();
//      std::cout << "kmeans" << std::chrono::duration_cast<std::chrono::nanoseconds>(end - statrt).count() << std::endl;
  } // if must run kmeans
  // create nodes

  for(unsigned int i = 0; i < clusters.size(); ++i)
  {
      kmeansMutex.lock();

    NodeId id = m_nodes.size();
    m_nodes.push_back(Node(id));
    m_nodes.back().descriptor = clusters[i];
    m_nodes.back().parent = parent_id;
    m_nodes[parent_id].children.push_back(id);
    kmeansMutex.unlock();

  }
  if(current_level < m_L) {
    if (current_level <=1 ) {
        // iterate again with the resulting clusters
        const std::vector<NodeId> &children_ids = m_nodes[parent_id].children;
        task_group g;
        for (unsigned int i = 0; i < clusters.size(); ++i) {
            NodeId id = children_ids[i];
            std::vector<pDescriptor> child_features;
            child_features.reserve(groups[i].size());

            std::vector<unsigned int>::const_iterator vit;
            for (vit = groups[i].begin(); vit != groups[i].end(); ++vit) {
                child_features.push_back(descriptors[*vit]);
            }

            if (child_features.size() > 1) {
                g.run([=]{ HKmeansStep(id, child_features, current_level + 1);});
            }
        }
        g.wait();
    } else {
      // iterate again with the resulting clusters
      const std::vector<NodeId> &children_ids = m_nodes[parent_id].children;

      for (unsigned int i = 0; i < clusters.size(); ++i) {
        NodeId id = children_ids[i];
        std::vector<pDescriptor> child_features;
        child_features.reserve(groups[i].size());

        std::vector<unsigned int>::const_iterator vit;
        for (vit = groups[i].begin(); vit != groups[i].end(); ++vit) {
          child_features.push_back(descriptors[*vit]);
        }

        if (child_features.size() > 1) {
          HKmeansStep(id, child_features, current_level + 1);
        }
      }
//    kmeansMutex.unlock();
    }
  }

//  std::cout << "all done " << current_level << std::endl;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::initiateClusters
  (const std::vector<pDescriptor> &descriptors,
   std::vector<TDescriptor> &clusters) const
{
  initiateClustersKMpp(descriptors, clusters);  
}

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::initiateClustersKMpp(
        const std::vector<pDescriptor> &pfeatures,
        std::vector<TDescriptor> &clusters) const
{
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
    std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());

    // 1.

    int ifeature = RandomInt(0, pfeatures.size()-1);

    // create first cluster
    clusters.push_back(*pfeatures[ifeature]);

    // compute the initial distances
    typename std::vector<pDescriptor>::const_iterator fit;
    std::vector<double>::iterator dit;
    dit = min_dists.begin();
    for(fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
    {
        *dit = F::distance(*(*fit), clusters.back());
    }

    while((int)clusters.size() < m_k)
    {
        // 2.
        dit = min_dists.begin();
        for(fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
        {
            if(*dit > 0)
            {
                double dist = F::distance(*(*fit), clusters.back());
                if(dist < *dit) *dit = dist;
            }
        }

        // 3.
        double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);

        if(dist_sum > 0)
        {
            cout.precision(17);
            cout << ":dist sum " << dist_sum << endl;
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
                ifeature = pfeatures.size()-1;
            else
                ifeature = dit - min_dists.begin();

            clusters.push_back(*pfeatures[ifeature]);
            cout << "ifeature " << ifeature << endl;
        } // if dist_sum > 0
        else
            break;

    } // while(used_clusters < m_k)

}
// --------------------------------------------------------------------------
//
//template<class TDescriptor, class F>
//void TemplatedVocabulary<TDescriptor,F>::initiateClustersKMpp(
//  const std::vector<pDescriptor> &pfeatures,
//    std::vector<TDescriptor> &clusters) const
//{
//  // Implements kmeans++ seeding algorithm
//  // Algorithm:
//  // 1. Choose one center uniformly at random from among the data points.
//  // 2. For each data point x, compute D(x), the distance between x and the nearest
//  //    center that has already been chosen.
//  // 3. Add one new data point as a center. Each point x is chosen with probability
//  //    proportional to D(x)^2.
//  // 4. Repeat Steps 2 and 3 until k centers have been chosen.
//  // 5. Now that the initial centers have been chosen, proceed using standard k-means
//  //    clustering.
//
//  clusters.resize(0);
//  clusters.reserve(m_k);
////  tbb::concurrent_vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());
//  std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());
//  // 1.
//
//  int ifeature = RandomInt(0, pfeatures.size()-1);
//
//  // create first cluster
//  clusters.push_back(*pfeatures[ifeature]);
//
//  // compute the initial distances
//  typename std::vector<pDescriptor>::const_iterator fit;
////  tbb::concurrent_vector<double>::iterator dit;
//  std::vector<double>::iterator dit;
//  dit = min_dists.begin();
////  int size =  pfeatures.size();
////  std::cout << size << std::endl;
////  tbb::parallel_for(0, size, [&](int i) {
////      min_dists[i] = F::distance(*pfeatures[i], clusters.back());
////  });
//
////41896382142
////51134457109
////  std::cout << "inti: " << min_dists.size() << std::endl << pfeatures.size() << std::endl;
//  for(fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
//  {
////    std::cout << *fit << std::endl;
////    std::cout << *dit << std::endl;
//    *dit = F::distance(*(*fit), clusters.back());
//  }
////  std::cout << "running\n";
//  while((int)clusters.size() < m_k)
//  {
//    // 2.
//    dit = min_dists.begin();
////    tbb::parallel_for(0, size, [&](int i) {
////      if (min_dists[i] > 0) {
////        double dist = F::distance(*pfeatures[i], clusters.back());
////        if(dist < min_dists[i]) min_dists[i] = dist;
////      }
//////        min_dists[i] = F::distance(*pfeatures[i], clusters.back());
////    });
////    std::cout << "before for\n";
//    for(fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
//    {
////      std::cout << *dit << std::endl;
//      if(*dit > 0)
//      {
//        double dist = F::distance(*(*fit), clusters.back());
//        if(dist < *dit) *dit = dist;
//      }
//    }
////
//    // 3.
//    double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);
//    if(dist_sum > 0)
//    {
//      double cut_d;
//      do
//      {
//        cut_d = RandomValue<double>(0, dist_sum);
//      } while(cut_d == 0.0);
//
//      double d_up_now = 0;
//      for(dit = min_dists.begin(); dit != min_dists.end(); ++dit)
//      {
//        d_up_now += *dit;
//        if(d_up_now >= cut_d) break;
//      }
//
//      if(dit == min_dists.end())
//        ifeature = pfeatures.size()-1;
//      else
//        ifeature = dit - min_dists.begin();
//
//      clusters.push_back(*pfeatures[ifeature]);
//
//    } // if dist_sum > 0
//    else
//      break;
//
//  } // while(used_clusters < m_k)
//  for (auto cl: clusters) {
//    std::cout << cl << std::endl;
//  }
//  std::cout << std::endl;
/*
 * 125000
125000
67016
67016
39448
39448
27568
27568
57984
57984
35616
35616
22368
22368

 */
//}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::createWords()
{
  m_words.resize(0);
  
  if(!m_nodes.empty())
  {
    m_words.reserve( (int)pow((double)m_k, (double)m_L) );

    typename std::vector<Node>::iterator nit;
    
    nit = m_nodes.begin(); // ignore root
    for(++nit; nit != m_nodes.end(); ++nit)
    {
      if(nit->isLeaf())
      {
        nit->word_id = m_words.size();
        m_words.push_back( &(*nit) );
      }
    }
  }
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::setNodeWeights
  (const std::vector<std::vector<TDescriptor> > &training_features)
{
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
    
    typename std::vector<std::vector<TDescriptor> >::const_iterator mit;
    typename std::vector<TDescriptor>::const_iterator fit;

    for(mit = training_features.begin(); mit != training_features.end(); ++mit)
    {
      fill(counted.begin(), counted.end(), false);

      for(fit = mit->begin(); fit < mit->end(); ++fit)
      {
        WordId word_id;
        transform(*fit, word_id);

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

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline unsigned int TemplatedVocabulary<TDescriptor,F>::size() const
{
  return m_words.size();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline bool TemplatedVocabulary<TDescriptor,F>::empty() const
{
  return m_words.empty();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
float TemplatedVocabulary<TDescriptor,F>::getEffectiveLevels() const
{
  long sum = 0;
  typename std::vector<Node*>::const_iterator wit;
  for(wit = m_words.begin(); wit != m_words.end(); ++wit)
  {
    const Node *p = *wit;
    
    for(; p->id != 0; sum++) p = &m_nodes[p->parent];
  }
  
  return (float)((double)sum / (double)m_words.size());
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TDescriptor TemplatedVocabulary<TDescriptor,F>::getWord(WordId wid) const
{
  return m_words[wid]->descriptor;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
WordValue TemplatedVocabulary<TDescriptor, F>::getWordWeight(WordId wid) const
{
  return m_words[wid]->weight;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
WordId TemplatedVocabulary<TDescriptor, F>::transform
  (const TDescriptor& feature) const
{
  if(empty())
  {
    return 0;
  }
  
  WordId wid;
  transform(feature, wid);
  return wid;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::transform(
  const std::vector<TDescriptor>& features, BowVector &v) const
{
  v.clear();
  
  if(empty())
  {
    return;
  }

  // normalize 
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);

  typename std::vector<TDescriptor>::const_iterator fit;

  if(m_weighting == TF || m_weighting == TF_IDF)
  {
    for(fit = features.begin(); fit < features.end(); ++fit)
    {
      WordId id;
      WordValue w; 
      // w is the idf value if TF_IDF, 1 if TF
      
      transform(*fit, id, w);
      
      // not stopped
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
    for(fit = features.begin(); fit < features.end(); ++fit)
    {
      WordId id;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY
      
      transform(*fit, id, w);
      
      // not stopped
      if(w > 0) v.addIfNotExist(id, w);
      
    } // if add_features
  } // if m_weighting == ...
  
  if(must) v.normalize(norm);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F> 
void TemplatedVocabulary<TDescriptor,F>::transform(
  const std::vector<TDescriptor>& features,
  BowVector &v, FeatureVector &fv, int levelsup) const
{
  v.clear();
  fv.clear();
  
  if(empty()) // safe for subclasses
  {
    return;
  }
  
  // normalize 
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);
  
  typename std::vector<TDescriptor>::const_iterator fit;
  
  if(m_weighting == TF || m_weighting == TF_IDF)
  {
    unsigned int i_feature = 0;
    for(fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
    {
      WordId id;
      NodeId nid;
      WordValue w; 
      // w is the idf value if TF_IDF, 1 if TF
      
      transform(*fit, id, w, &nid, levelsup);
      
      if(w > 0) // not stopped
      { 
        v.addWeight(id, w);
        fv.addFeature(nid, i_feature);
      }
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
    unsigned int i_feature = 0;
    for(fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
    {
      WordId id;
      NodeId nid;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY
      
      transform(*fit, id, w, &nid, levelsup);
      
      if(w > 0) // not stopped
      {
        v.addIfNotExist(id, w);
        fv.addFeature(nid, i_feature);
      }
    }
  } // if m_weighting == ...
  
  if(must) v.normalize(norm);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F> 
inline double TemplatedVocabulary<TDescriptor,F>::score
  (const BowVector &v1, const BowVector &v2) const
{
  return m_scoring_object->score(v1, v2);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::transform
  (const TDescriptor &feature, WordId &id) const
{
  WordValue weight;
  transform(feature, id, weight);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::transform(const TDescriptor &feature,
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
//    ++current_level;
//    nodes = m_nodes[final_id].children;
//    final_id = nodes[0];

      ++current_level;
      nodes2 = m_nodes[final_id].children;
      nodes.clear();
      for (int i = 0; i < nodes2.size(); ++i) {
          if (m_nodes[nodes2[i]].descriptor.rows * m_nodes[nodes2[i]].descriptor.cols) {
              nodes.push_back(nodes2[i]);
          }
      }
      if (nodes.empty()) {
          break;
      }
      final_id = nodes[0];


 
    double best_d = F::distance(feature, m_nodes[final_id].descriptor);

    for(nit = nodes.begin() + 1; nit != nodes.end(); ++nit)
    {
      NodeId id = *nit;
      double d = F::distance(feature, m_nodes[id].descriptor);
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

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
NodeId TemplatedVocabulary<TDescriptor,F>::getParentNode
  (WordId wid, int levelsup) const
{
  NodeId ret = m_words[wid]->id; // node id
  while(levelsup > 0 && ret != 0) // ret == 0 --> root
  {
    --levelsup;
    ret = m_nodes[ret].parent;
  }
  return ret;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::getWordsFromNode
  (NodeId nid, std::vector<WordId> &words) const
{
  words.clear();
  
  if(m_nodes[nid].isLeaf())
  {
    words.push_back(m_nodes[nid].word_id);
  }
  else
  {
    words.reserve(m_k); // ^1, ^2, ...
    
    std::vector<NodeId> parents;
    parents.push_back(nid);
    
    while(!parents.empty())
    {
      NodeId parentid = parents.back();
      parents.pop_back();
      
      const std::vector<NodeId> &child_ids = m_nodes[parentid].children;
      std::vector<NodeId>::const_iterator cit;
      
      for(cit = child_ids.begin(); cit != child_ids.end(); ++cit)
      {
        const Node &child_node = m_nodes[*cit];
        
        if(child_node.isLeaf())
          words.push_back(child_node.word_id);
        else
          parents.push_back(*cit);
        
      } // for each child
    } // while !parents.empty
  }
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
int TemplatedVocabulary<TDescriptor,F>::stopWords(double minWeight)
{
  int c = 0;
  typename std::vector<Node*>::iterator wit;
  for(wit = m_words.begin(); wit != m_words.end(); ++wit)
  {
    if((*wit)->weight < minWeight)
    {
      ++c;
      (*wit)->weight = 0;
    }
  }
  return c;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::save(const std::string &filename) const
{
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
  if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
  
  save(fs);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::load(const std::string &filename)
{
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
  
  this->load(fs);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::save(cv::FileStorage &f,
  const std::string &name) const
{
  // Format YAML:
  // vocabulary 
  // {
  //   k:
  //   L:
  //   scoringType:
  //   weightingType:
  //   nodes 
  //   [
  //     {
  //       nodeId:
  //       parentId:
  //       weight:
  //       descriptor: 
  //     }
  //   ]
  //   words
  //   [
  //     {
  //       wordId:
  //       nodeId:
  //     }
  //   ]
  // }
  //
  // The root node (index 0) is not included in the node vector
  //
  
  f << name << "{";
  
  f << "k" << m_k;
  f << "L" << m_L;
  f << "scoringType" << m_scoring;
  f << "weightingType" << m_weighting;
  
  // tree
  f << "nodes" << "[";
  std::vector<NodeId> parents, children;
  std::vector<NodeId>::const_iterator pit;

  parents.push_back(0); // root

  while(!parents.empty())
  {
    NodeId pid = parents.back();
    parents.pop_back();

    const Node& parent = m_nodes[pid];
    children = parent.children;

    for(pit = children.begin(); pit != children.end(); pit++)
    {
      const Node& child = m_nodes[*pit];

      // save node data
      f << "{:";
      f << "nodeId" << (int)child.id;
      f << "parentId" << (int)pid;
      f << "weight" << (double)child.weight;
      f << "descriptor" << F::toString(child.descriptor);
      f << "}";
      
      // add to parent list
      if(!child.isLeaf())
      {
        parents.push_back(*pit);
      }
    }
  }
  
  f << "]"; // nodes

  // words
  f << "words" << "[";
  
  typename std::vector<Node*>::const_iterator wit;
  for(wit = m_words.begin(); wit != m_words.end(); wit++)
  {
    WordId id = wit - m_words.begin();
    f << "{:";
    f << "wordId" << (int)id;
    f << "nodeId" << (int)(*wit)->id;
    f << "}";
  }
  
  f << "]"; // words

  f << "}";

}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::load(const cv::FileStorage &fs,
  const std::string &name)
{
  m_words.clear();
  m_nodes.clear();
  
  cv::FileNode fvoc = fs[name];
  
  m_k = (int)fvoc["k"];
  m_L = (int)fvoc["L"];
  m_scoring = (ScoringType)((int)fvoc["scoringType"]);
  m_weighting = (WeightingType)((int)fvoc["weightingType"]);
  
  createScoringObject();

  // nodes
  cv::FileNode fn = fvoc["nodes"];

  m_nodes.resize(fn.size() + 1); // +1 to include root
  m_nodes[0].id = 0;

  for(unsigned int i = 0; i < fn.size(); ++i)
  {
    NodeId nid = (int)fn[i]["nodeId"];
    NodeId pid = (int)fn[i]["parentId"];
    WordValue weight = (WordValue)fn[i]["weight"];
    std::string d = (std::string)fn[i]["descriptor"];
    
    m_nodes[nid].id = nid;
    m_nodes[nid].parent = pid;
    m_nodes[nid].weight = weight;
    m_nodes[pid].children.push_back(nid);
    
    F::fromString(m_nodes[nid].descriptor, d);
  }
  
  // words
  fn = fvoc["words"];
  
  m_words.resize(fn.size());

  for(unsigned int i = 0; i < fn.size(); ++i)
  {
    NodeId wid = (int)fn[i]["wordId"];
    NodeId nid = (int)fn[i]["nodeId"];
    
    m_nodes[nid].word_id = wid;
    m_words[wid] = &m_nodes[nid];
  }
}

// --------------------------------------------------------------------------

/**
 * Writes printable information of the vocabulary
 * @param os stream to write to
 * @param voc
 */
template<class TDescriptor, class F>
std::ostream& operator<<(std::ostream &os, 
  const TemplatedVocabulary<TDescriptor,F> &voc)
{
  os << "Vocabulary: k = " << voc.getBranchingFactor() 
    << ", L = " << voc.getDepthLevels()
    << ", Weighting = ";

  switch(voc.getWeightingType())
  {
    case TF_IDF: os << "tf-idf"; break;
    case TF: os << "tf"; break;
    case IDF: os << "idf"; break;
    case BINARY: os << "binary"; break;
  }

  os << ", Scoring = ";
  switch(voc.getScoringType())
  {
    case L1_NORM: os << "L1-norm"; break;
    case L2_NORM: os << "L2-norm"; break;
    case CHI_SQUARE: os << "Chi square distance"; break;
    case KL: os << "KL-divergence"; break;
    case BHATTACHARYYA: os << "Bhattacharyya coefficient"; break;
    case DOT_PRODUCT: os << "Dot product"; break;
  }
  
  os << ", Number of words = " << voc.size();

  return os;
}

} // namespace DBoW2

#endif
