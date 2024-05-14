#include "knnclassifier.h"
#include <iostream>


void KNNClassifier::fit(TrainingSet& trainImages, std::vector<int>& trainLabels) {
    for (int i = 0; i < trainImages.getImages().size(); i ++)
        trainLabels.push_back(trainImages.getImages().at(i).at(0));
};


std::vector<int> KNNClassifier::predict(TrainingSet& trainImages) {
    std::vector<int>answer;

    for (int i = 0; i < trainImages.getImages().size(); i ++) {
        // we make a priority queue in which we keep the distance and the images
        std::priority_queue<std::pair<double, std::vector<int>>, std::vector<std::pair<double, std::vector<int>>>, std::greater<std::pair<double, std::vector<int>>>> pQueue;
        for (int ii = 0; ii < this->getTrainingSet().getImages().size(); ii ++) {
            int euclideanDist = 0;
            // we only check the image pixels
            for (int j = 1; j < trainImages.getImages().at(i).size(); j ++)
                euclideanDist += std::pow(trainImages.getImages().at(i).at(j) - this->getTrainingSet().getImages().at(ii).at(j), 2);
            // we add every image to the queue
            pQueue.push({sqrt(euclideanDist), this->getTrainingSet().getImages().at(ii)});
        }

        int digitFreq[10] = {0};
        for (int aux = 0; aux < k; aux ++) {
            // we take the first digit from each image
            int digit = pQueue.top().second.at(0);
            digitFreq[digit] ++;
            pQueue.pop();
        }

        // we find the majority
        int label = 0;
        int maximalDigit = 0;
        for (int aux = 0; aux < 10; aux ++)
            if (digitFreq[aux] > maximalDigit) {
                maximalDigit = digitFreq[aux];
                label = aux;
            }

        answer.push_back(label);
    }

    return answer;
};


double KNNClassifier::eval(TrainingSet& trainImages) {

    std::vector<int>correctLabels;
    fit(trainImages,correctLabels);

    auto predictedLabels = predict(trainImages);

    int correctPredictions = 0;
    int totalSamples = trainImages.getImages().size();
    for (int i = 0 ; i < correctLabels.size(); i ++)
        if (correctLabels[i] == predictedLabels[i])
            correctPredictions ++;

    return static_cast<double>(correctPredictions)/totalSamples;
};


bool KNNClassifier::load(std::string filepath) {
    return Classifier::load(filepath);
}


bool KNNClassifier::save(std::string filepath) {
    return Classifier::save(filepath);
}

