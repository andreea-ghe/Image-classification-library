#include "bayesclassifier.h"
#include <cmath>
#include <iostream>
#include <iomanip>


void BayesClassifier::fit(TrainingSet& trainImages, std::vector<int>& trainLabels) {
    // we fix the training set matrix
    for (int i = 0; i < this->getTrainingSet().getImages().size(); i ++)
        for (int j = 1; j < this->getTrainingSet().getImages().at(i).size(); j ++)
            if (this->getTrainingSet().getImages().at(i).at(j) != 0)
                this->getTrainingSet().changeElementAtPos(i, j, 255);

    // we compute the prior vector
    for (int i = 0; i < 10; i ++)
        freqLabel[i] = 0;
    // we find the number of occurences of a class
    for (int i = 0; i < this->getTrainingSet().getImages().size(); i ++)
        freqLabel[this->getTrainingSet().getImages().at(i).at(0)] ++;
    // this is the prior vector, the probability for every class
    for (int i = 0; i < 10; i ++)
        prior.push_back(static_cast<double>(freqLabel[i])/this->getTrainingSet().getImages().size());

    // we convert the values to 0 or 255
    for (int i = 0; i < trainImages.getImages().size(); i ++)
        for (int j = 1; j < trainImages.getImages().at(i).size(); j ++)
            if (trainImages.getImages().at(i).at(j) > 0)
                trainImages.changeElementAtPos(i, j, 255);

    for (int i = 0; i < trainImages.getImages().size(); i ++)
        trainLabels.push_back(trainImages.getImages().at(i).at(0));
};

std::vector<int> BayesClassifier::predict(TrainingSet& trainImages) {

    std::vector<int>answer;

    // for each class we compute the probability of each position to be 255
    for (int label = 0; label < 10; label ++) {
        likelihood.push_back(std::vector<double>());
        for (int i = 1; i <= 28*28; i ++) {
            int cnt255 = 0;
            for (int j = 0; j < this->getTrainingSet().getImages().size(); j ++)
                if (this->getTrainingSet().getImages().at(j).at(0) == label && this->getTrainingSet().getImages().at(j).at(i) == 255)
                    cnt255 ++;
            //std::cout << static_cast<double>(cnt255 + 1)/(freqLabel[label] + 10) << '\n';
            likelihood.at(label).push_back(static_cast<double>(cnt255 + 1)/(freqLabel[label] + 10));
        }
    }

//    for (int i = 0; i < likelihood.size(); i ++) {
//        for (int j = 0; j < likelihood.at(i).size(); j++)
//            std::cout << likelihood.at(i).at(j) << ' ';
//        std::cout << likelihood.at(i).size() << "yay\n";
//    }

    for (int i = 0; i < trainImages.getImages().size(); i ++) {
        double maxim = 999999;
        int ans = 0;
        for (int label = 0; label < 10; label ++) {
            double posterior = std::log(prior.at(label));
            if (posterior < 0.00001)
                posterior = 0.00001;

            //std::cout << prior.at(label) << '\n';
            for (int j = 1; j < trainImages.getImages().at(i).size(); j ++) {
                if (trainImages.getImages().at(i).at(j) == 255)
                    posterior += std::log(likelihood.at(label).at(j - 1));
                else
                    posterior += std::log(1 - likelihood.at(label).at(j - 1));
            }
            if (posterior * (-1) < maxim) {
                maxim = posterior * (-1);
                ans = label;
            }
        }
        answer.push_back(ans);
    }
    return answer;
};

double BayesClassifier::eval(TrainingSet& trainImages) {

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

bool BayesClassifier::load(std::string filepath) {
    return Classifier::load(filepath);
}


bool BayesClassifier::save(std::string filepath) {
    return Classifier::save(filepath);
}

