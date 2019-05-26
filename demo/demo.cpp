/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>


#include <tbb/parallel_for.h>
// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase
#include <chrono>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <iomanip>
#include <chrono>
#include <vector>
#include <string>

#include "VocabularyCharGPU.h"

using namespace DBoW2;
using namespace std;
using namespace tbb;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void loadFeaturesUCHAR(vector<vector<uchar > > &features);
void loadFeatures(vector<vector<cv::Mat > > &features);
void changeStructureUCHAR(const cv::Mat &plain, vector<uchar> &out);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreationUCHAR(const vector<vector<uchar > > &features, int grainsize);
void testVocCreation(const vector<vector<cv::Mat > > &features);
//void testDatabase(const vector<vector<cv::Mat > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// number of training images
const int NIMAGES = 12000;
const string path_dir = "/shared/datasets/KITTI/dataset/sequences/";
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait()
{
  cout << endl << "Pressdnoe" << endl;
  getchar();
}

int L = 4;
vector<vector<long long>> res;
// ----------------------------------------------------------------------------

int main()
{
    vector<vector<cv::Mat > > features;
//  loadFeaturesUCHAR(features);
    loadFeatures(features);
    testVocCreation(features);
//  testVocCreationUCHAR(features, 73);

    testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeaturesUCHAR(vector<vector<uchar > > &features)
{
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;

    string path = path_dir + "00/image_0/";
  for(int i = 0; i < 1000; ++i)
  {
    stringstream ss;

    ss << path << setfill('0') << setw(6) << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<uchar >());
    changeStructureUCHAR(descriptors, features.back());
  }
    path = path_dir + "02/image_0/";
    for(int i = 0; i < NIMAGES; ++i)
    {
        stringstream ss;

        ss << path << setfill('0') << setw(6) << i << ".png";

        cv::Mat image = cv::imread(ss.str(), 0);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orb->detectAndCompute(image, mask, keypoints, descriptors);

        features.push_back(vector<uchar >());
        changeStructureUCHAR(descriptors, features.back());
    }
    cout << features.size() << endl;
}

void loadFeatures(vector<vector<cv::Mat > > &features)
{
    features.clear();
    features.reserve(NIMAGES);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    cout << "Extracting ORB eaures..." << endl;
    string path = path_dir + "01/image_0/";
    for(int i = 0; i < 1000; ++i)
    {
        stringstream ss;

        ss << path << setfill('0') << setw(6) << i << ".png";

        cv::Mat image = cv::imread(ss.str(), 0);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orb->detectAndCompute(image, mask, keypoints, descriptors);

        features.push_back(vector<cv::Mat >());
        changeStructure(descriptors, features.back());
    }
	return;
    path = path_dir + "02/image_0/";
    for(int i = 0; i < 4000; ++i)
    {
        stringstream ss;

        ss << path << setfill('0') << setw(6) << i << ".png";

        cv::Mat image = cv::imread(ss.str(), 0);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orb->detectAndCompute(image, mask, keypoints, descriptors);

        features.push_back(vector<cv::Mat >());
        changeStructure(descriptors, features.back());
    }

    path = path_dir + "08/image_0/";
    for(int i = 0; i < 4000; ++i)
    {
        stringstream ss;

        ss << path << setfill('0') << setw(6) << i << ".png";

        cv::Mat image = cv::imread(ss.str(), 0);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orb->detectAndCompute(image, mask, keypoints, descriptors);

        features.push_back(vector<cv::Mat >());
        changeStructure(descriptors, features.back());
    }
    path = path_dir + "13/image_0/";
    for(int i = 0; i < 3000; ++i)
    {
        stringstream ss;

        ss << path << setfill('0') << setw(6) << i << ".png";

        cv::Mat image = cv::imread(ss.str(), 0);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orb->detectAndCompute(image, mask, keypoints, descriptors);

        features.push_back(vector<cv::Mat >());
        changeStructure(descriptors, features.back());
    }

}
// ----------------------------------------------------------------------------

void changeStructureUCHAR(const cv::Mat &plain, vector<uchar> &out)
{
    if (plain.isContinuous()) {
        out.assign(plain.datastart, plain.dataend);
    } else {
        for (int i = 0; i < plain.rows; ++i) {
            out.insert(out.end(), plain.ptr<uchar>(i), plain.ptr<uchar>(i)+plain.cols);
        }
    }

}

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
    out.resize(plain.rows);
    for(int i = 0; i < plain.rows; ++i)
    {
        out[i] = plain.row(i);
    }
}

// ----------------------------------------------------------------------------

void testVocCreationUCHAR(const vector<vector<uchar > > &features, int grainsize)
{
// branching factor and depth levels
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;
  VocabularyCharGPU voc(k, L, grainsize, weight, score);

    auto statrt = std::chrono::high_resolution_clock::now();
    voc.create(features);
    auto end = std::chrono::high_resolution_clock::now();
  cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - statrt).count() << endl;
////  cout << "... done!" << endl;
//
////  cout << "Vocabulary information: " << endl
////  << voc << endl << endl;
//
//  // lets do something with this vocabulary
////  cout << "Matching images against themselves (0 low, 1 high): " << endl;
//  BowVector v1, v2;
//  for(int i = 0; i < 5; i++)
//  {
//    voc.transform(features[i], v1);
//    for(int j = 0; j < 5; j++)
//    {
//      voc.transform(features[j], v2);
//
//      double score = voc.score(v1, v2);
//      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
//    }
//  }

  // save the vocabulary to disk
//  cout << endl << "Saving vocabulary..." << endl;
//  voc.save("small_voc.yml.gz");
//  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat > > &features)
{
// branching factor and depth levels
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    OrbVocabulary voc(k, L, weight, score);

//  // save the vocabulary to disk
    auto statrt = std::chrono::high_resolution_clock::now();
    voc.create2(features, 75);
    auto end = std::chrono::high_resolution_clock::now();
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - statrt).count() << endl;

    // lets do something with this vocabulary
//  cout << "Matching images against themselves (0 low, 1 high): " << endl;
//    BowVector v1, v2;
//    for(int i = 0; i < NIMAGES; i++)
//    {
//        voc.transform(features[i], v1);
//        for(int j = 0; j < NIMAGES; j++)
//        {
//            voc.transform(features[j], v2);
//
//            double score = voc.score(v1, v2);
//            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
//        }
//    }

    // save the vocabulary to disk
//  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "all Done" << endl;
}
void testDatabase(const vector<vector<cv::Mat > > &features)
{
//    return;

  cout << "Creating a small data..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");

  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < 5; i++)
  {
    db.add(features[i]);
  }

  cout << "... one!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;

    auto begin = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < 5; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

    auto end = std::chrono::high_resolution_clock::now();
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << endl;
  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;

  // once saved, we can load it again
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------


