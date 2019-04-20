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

using namespace DBoW2;
using namespace std;
using namespace tbb;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<uchar > > &features);
void changeStructure(const cv::Mat &plain, vector<uchar> &out);
void testVocCreation(const vector<vector<uchar > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 5;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main()
{
//  parallel_for(0, 3, [&](int i) {
//    int a = 2;
//  });
  vector<vector<uchar > > features;
  loadFeatures(features);
  for (int i = 0; i < 1; ++i)
    testVocCreation(features);

    return 0;
//  wait();
//
//  testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<uchar > > &features)
{
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    stringstream ss;

    ss << "/home/kirill/Desktop/UNI/visual-similarity-metrics/data/KITTi/dataset/sequences/00/image_0/" << setfill('0') << setw(6) << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    const int k = 3;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
    OrbVocabulary voc(k, L, weight, score);
    vector<vector<cv::Mat>> temp;
    temp.push_back({descriptors});
    voc.create(temp);
    features.push_back(vector<uchar >());
    changeStructure(descriptors, features.back());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<uchar> &out)
{
    if (plain.isContinuous()) {
        out.assign(plain.datastart, plain.dataend);
    } else {
        for (int i = 0; i < plain.rows; ++i) {
            out.insert(out.end(), plain.ptr<uchar>(i), plain.ptr<uchar>(i)+plain.cols);
        }
    }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<uchar > > &features)
{
//       5539799825
// branching factor and depth levels
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;
//  VocabularyUCHAR voc(k, L, weight, score);
//  voc.create(features);
//  OrbVocabulary voc(k, L, weight, score);
//
////  // save the vocabulary to disk
//  auto statrt = std::chrono::high_resolution_clock::now();
//  voc.create(features);
//    auto end = std::chrono::high_resolution_clock::now();
//  cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - statrt).count() << endl;
////  cout << "... done!" << endl;
//
////  cout << "Vocabulary information: " << endl
////  << voc << endl << endl;
//
//  // lets do something with this vocabulary
////  cout << "Matching images against themselves (0 low, 1 high): " << endl;
//  BowVector v1, v2;
//  for(int i = 0; i < NIMAGES; i++)
//  {
//    voc.transform(features[i], v1);
//    for(int j = 0; j < NIMAGES; j++)
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

void testDatabase(const vector<vector<cv::Mat > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");
  
  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

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


