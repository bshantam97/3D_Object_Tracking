
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        // Convert to homogenous coordinates
        X = (cv::Mat_<double>(4,1) << it1->x, it1->y, it1->z, 1);

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        // Idea is to see whether a point is being held by 1 or multiple boxes
        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
}

// match list of 3D objects between current and previous frame
// Use the keypoint matches between the current and the previous frame (Make outer loop of those)
// Then figure out which of the keypoints are contained within the bounding boxes in the previous and the current frame
// Store the boxIds lets say in a multimap
// Matches mus be the ones with the highest number of keypoint correspondences
// We need to output a map with matching boxId's between previous frame and current frame

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // it-> std::vector<cv::DMatch>::iterator object
    // First we create an outer loop to iterate over the keypoint matches
    // PREVIOUS FRAME is "queryIdx" and the CURRENT FRAME is "trainIdx"

    // At each index of the vector I have my query index and the key index
    // What I want to do is to find out the associated keypoint (x,y) coordinate with the associated cv::DMatch
    // Then I know we have the cv::Rect object inside the DataFrame which provides us with the box coordinates
    // Using an if condition can determine whether the keypoint is inside the bounding box or not 
    // In the struct DataFrame we have a std::vector<BoundingBox> that provides the ROI around the detected objects
    // In the struct DataFrame we also have a std::vector<cv::KeyPoint> that gives the keypoints within the camera image
    // Now queryIdx and trainIdx basically provide the row/index of the interest point matrix 
    // If lets say our queryIdx is 95 and trainIdx is 1 we can figure out the associated keypoint coordinates
    // in the previous and current frame by prevFrame.keypoints[queryIdx].pt and currFrame.keypoints[trainIdx].pt
    // When we loop it will point to the vector indices of the "matches" which contains cv::DMatch->queryIdx and trainIdx
    // Now using this we can obtain our keypoint indices

    // Define a single vector to hold the previous and current frame keypoint matches
    // Have defined a MatchedKeypointCoordinate structure to hold the data
    
    std::vector<MatchedKeypointCoordinate> matchedKeypoints;
    int index = 0;
    
    for (auto it = matches.begin(); it != matches.end(); it++) {
        matchedKeypoints[index].prevX = prevFrame.keypoints[it->queryIdx].pt.x;
        matchedKeypoints[index].prevY = prevFrame.keypoints[it->queryIdx].pt.y;
        matchedKeypoints[index].currX = currFrame.keypoints[it->trainIdx].pt.x;
        matchedKeypoints[index].currY = currFrame.keypoints[it->trainIdx].pt.y;
        ++index;
    }
    // The idea here is to create 2 maps to store boxIds and the number of keypoints inside it
    // After we input the information we can iterate and compute the difference between number of keypoints and the one with 
    // the minimum difference will be input into the map
    // At most the time complexity should be O(numBoundingBox*maxKeypoints)
    std::map<int,int> prevFrameKeypoints;
    std::map<int,int> currFrameKeypoint;
    std::pair<int,int> difference;

    // Store the boxId's and associated number of keypoints in the map
    for (auto it1 = prevFrame.boundingBoxes.begin(); it1 != prevFrame.boundingBoxes.end(); it1++) {
        int keypointCount = 0;
        for (auto it2 = matchedKeypoints.begin(); it2 != matchedKeypoints.end(); it2++) {
            if (it2->prevX >= (it1->roi.x) && it2->prevX <= (it1->roi.x + it1->roi.width) && it2->prevY >= (it1->roi.y) && it2->prevY <= (it1->roi.y + it1->roi.height)) {
                keypointCount++;
            }
        }
        prevFrameKeypoints.insert({it1->boxID, keypointCount});
    }

    for (auto it1 = currFrame.boundingBoxes.begin(); it1 != currFrame.boundingBoxes.end(); it1++) {
        int keypointCount = 0;
        for (auto it2 = matchedKeypoints.begin(); it2 != matchedKeypoints.end(); it2++) {
            if (it2->currX >= (it1->roi.x) && it2->currX <= (it1->roi.x + it1->roi.width) && it2->currY >= (it1->roi.y) && it2->currY <= (it1->roi.y + it1->roi.height)) {
                keypointCount++;
            }
        }
        currFrameKeypoint.insert({it1->boxID, keypointCount});
    }

    
    // Compute the differences between the prevFrameKeypoints and currFrameKeypoints stored inside a map
    for (auto it1 = prevFrameKeypoints.begin(); it1 != prevFrameKeypoints.end(); it1++) {
        // Set it to a very large value
        // currDiff will be used to update the current difference value and diff will be used to store the smallest difference value
        int diff = 1e-8; 
        int currDiff = 1e-8;
        for (auto it2 = currFrameKeypoint.begin(); it2 != currFrameKeypoint.end(); it2++) {
            if (currDiff < diff) {
                diff = currDiff;
                difference.first = it2->first; // BoxId
                difference.second = it2->second; // Number of Keypoints
            } else {
                currDiff = (it1->second) - (it2->second);
            }
        }
        // Store the best bounding boxes in the map
        bbBestMatches.insert({it1->first, difference.first});
    }
}
