#pragma once
#include "classifier.h"

class BayesClassifier: public Classifier {
protected:
    int freqLabel[10];
    std::vector<double>prior;
    std::vector<std::vector<double>>likelihood;
public:
    BayesClassifier(TrainingSet& _trainSet): Classifier{_trainSet} {};

    void fit(TrainingSet& trainImages, std::vector<int>& trainLabels) override;
    std::vector<int> predict(TrainingSet& trainImages) override;

    bool save(std::string filepath) override;
    bool load(std::string filepath) override;

    double eval(TrainingSet& trainImages) override;
};
