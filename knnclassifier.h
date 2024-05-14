#pragma once
#include <math.h>
#include <queue>
#include "classifier.h"

class KNNClassifier: public Classifier{
protected:
    int k;
public:
    KNNClassifier(TrainingSet& _trainSet, int _k): Classifier{_trainSet}, k{_k} {};

    void fit(TrainingSet& trainImages, std::vector<int>& trainLabels) override;
    std::vector<int> predict(TrainingSet& trainImages) override;

    bool save(std::string filepath) override;
    bool load(std::string filepath) override;

    double eval(TrainingSet& trainImages) override;

    void setK(int _k) { k = _k; };
};