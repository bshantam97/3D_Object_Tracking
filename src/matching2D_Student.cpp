#include <numeric>
#include "matching2D.hpp"

void detKeypointsHarris (std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    // The image input is already grayscale
    int blockSize = 4; //  The neighborhood size of the Harris corner response
    int aperture = 3; // For the sobel operator
    double k = 0.04; // The free parameter in the formula R = eig1*eig2 - k * (eig1 + eig2)
    int minResponse = 100;
    cv::Mat dest = cv::Mat::zeros(img.size(), CV_32FC1); // The image type is decided according to the documentation
    cv::cornerHarris(img, dest, blockSize, aperture, k);
    
    // Normalize the image to increase contrast for better feature extract
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dest, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // Scales and calculates absolute value and converts the image to 8-bit (0-255)
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Now what we do is calculate the local maxima in the harris response matrix and perform an NMS in a local neighborhood around each maxima
    // The keypoint vector has been given as argument
    double nmsOverlap;
    for (size_t i = 0; i < dst_norm_scaled.rows; i++) {
        for (size_t j = 0; j < dst_norm_scaled.cols; j++) {
            int response = (int)dst_norm_scaled.at<float>(i,j);
            if (response > minResponse) {

                cv::KeyPoint harrisKeypoint;
                harrisKeypoint.pt = cv::Point2f(j,i);
                harrisKeypoint.response = response; // Actual keypoint response value
                harrisKeypoint.size = 2*aperture; // Region around keypoint used to form keypoint descriptor
                // Create boolean overlap variable
                bool overlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                    // Compute the percentage overlap
                    double kptOverlap = cv::KeyPoint::overlap(harrisKeypoint, *it);
                    if (kptOverlap > nmsOverlap) {
                        overlap = true;
                        if (harrisKeypoint.response > (*it).response) {
                            *it = harrisKeypoint;
                            break;
                        }
                    }
                }

                if (!overlap) {
                    keypoints.push_back(harrisKeypoint);
                }
            }
        }
    }
    if (bVis) {
        std::string windowName = "Harris Keypoints";
        cv::namedWindow(windowName, 5);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern (std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis) {
    
    // Feature detector objects
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
    cv::Ptr<cv::FastFeatureDetector> FastDetector;
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::SiftFeatureDetector> siftDetector;

    // The create methods
    if (detectorType == "FAST") {
        int threshold = 50;
        bool nms = true;

        FastDetector  = cv::FastFeatureDetector::create(threshold, nms, type);
    }

    if (detectorType == "BRISK") detector = cv::BRISK::create();

    if (detectorType == "SIFT") siftDetector = cv::SIFT::create();

    if (detectorType == "ORB") detector = cv::ORB::create();

    if (detectorType == "AKAZE") detector = cv::AKAZE::create();
    
    // The Detect methods
    auto startTime = std::chrono::steady_clock::now();
    
    if (detectorType == "SIFT") siftDetector->detect(img, keypoints);
    
    if (detectorType == "BRISK" || detectorType == "ORB" || detectorType == "AKAZE") detector->detect(img, keypoints);
    
    if (detectorType == "FAST") FastDetector->detect(img, keypoints);
    
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime);
    std::cout << detectorType <<" Detected " << keypoints.size() << " keypoints in " << elapsedTime.count() << " ms " << std::endl; 

    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName1 = detectorType + "Results";
        cv::namedWindow(windowName1, 1);
        cv::imshow(windowName1, visImage);
        cv::waitKey(0);
    }
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = true;

    // cv::Ptr is basically a std::shared_ptr 

    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Ptr<cv::FlannBasedMatcher> flannMatcher;
    
    if (matcherType == "MAT_BF")
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    
    if (matcherType == "MAT_FLANN")
    {
        // Convert binary descriptors to floating point due to bug in OpenCV
        if (descSource.type() != CV_32F) {
            descSource.convertTo(descSource, CV_32F);
        }
        if (descRef.type() != CV_32F) {
            descRef.convertTo(descRef, CV_32F);
        }
        flannMatcher = cv::FlannBasedMatcher::create();
        std::cout << "FLANN Based Matching";
    }

    // perform matching task
    if (selectorType == "SEL_NN")
    { // nearest neighbor (best match)
        auto startTime = std::chrono::steady_clock::now();
        if (matcherType.compare("MAT_BF") == 0)
            matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        else 
            flannMatcher->match(descSource, descRef, matches);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime);
        std::cout << " NN with " << matches.size() << " matches " << " in " << elapsedTime.count() << " ms " << std::endl;
    }
    if (selectorType == "SEL_KNN")
    { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knnMatches;
        auto startTime = std::chrono::steady_clock::now();
        if (matcherType.compare("MAT_BF") == 0)
            matcher->knnMatch(descSource, descRef, knnMatches, 2); // Finds the best match for each descriptor in desc1
        else 
            flannMatcher->knnMatch(descSource, descRef, knnMatches, 2);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime);

        const double threshold = 0.7;
        for (auto x : knnMatches) {
            if (x[0].distance < threshold * x[1].distance) matches.push_back(x[0]);
        }
        std::cout << " Removed matches after nearest neighbor ratio " << knnMatches.size() - matches.size() << std::endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType)
{
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        // select appropriate descriptor
        
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::BRISK::create(threshold, octaves, patternScale);
        // perform feature description
        auto startTime = std::chrono::steady_clock::now();
        extractor->compute(img, keypoints, descriptors);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime);
        std::cout << descriptorType << " descriptor extraction in " << elapsedTime.count() << " ms" << std::endl;
    }
    else if (descriptorType.compare("SIFT") == 0) {
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::SIFT::create();
        auto startTime = std::chrono::steady_clock::now();
        extractor->compute(img, keypoints, descriptors);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime);
        std::cout << descriptorType << " descriptor extraction in " << elapsedTime.count() << " ms" << std::endl;
    } 
    else if (descriptorType.compare("ORB") == 0) {
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
        auto startTime = std::chrono::steady_clock::now();
        extractor->compute(img, keypoints, descriptors);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime);
        std::cout << descriptorType << " descriptor extraction in " << elapsedTime.count() << " ms" << std::endl;
    }
    else if (descriptorType.compare("BRIEF") == 0) {
        cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        auto startTime = std::chrono::steady_clock::now();
        extractor->compute(img, keypoints, descriptors);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime);
        std::cout << descriptorType << " descriptor extraction in " << elapsedTime.count() << " ms " << std::endl;
    } 
    else if (descriptorType.compare("FREAK") == 0) {
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::FREAK::create();
        auto startTime = std::chrono::steady_clock::now();
        extractor->compute(img, keypoints, descriptors);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime);
        std::cout << descriptorType << " descriptor extraction in " << elapsedTime.count() << " ms " << std::endl;
    } 
    else if (descriptorType.compare("AKAZE") == 0) {
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::AKAZE::create();
        auto startTime = std::chrono::steady_clock::now();
        extractor->compute(img, keypoints, descriptors);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime);
        std::cout << descriptorType << " descriptor extraction in " << elapsedTime.count() << " ms " << std::endl;
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}