#pragma once
#include <iostream>
#include <functional>
#include <fstream>
#include "trainingset.h"
using std::ostream;


class Metric {
protected:
    std::vector<std::vector<int>> confusionMatrix;
    TrainingSet testImg;
    std::function<void(TrainingSet&, std::vector<int>&)> fit;
    std::function<std::vector<int>(TrainingSet&)> predict;
    double trueNegative, falsePositive, falseNegative, truePositive;
public:
    Metric(TrainingSet& _testImg, std::function<void(TrainingSet&, std::vector<int>&)> _fit,
           std::function<std::vector<int>(TrainingSet&)> _predict);
    void computeConfusionMatrix(TrainingSet& testImg);
    std::vector<std::vector<int>> getConfusionMatrix();
    virtual double computeMetric(int label) = 0;
};


class Accuracy: public Metric {
public:
    Accuracy(TrainingSet& _testImg, std::function<void(TrainingSet&, std::vector<int>&)> _fit,
             std::function<std::vector<int>(TrainingSet&)> _predict): Metric{_testImg, _fit, _predict} {};
    double computeMetric(int label) override;
};


class Precision: public Metric {
public:
    Precision(TrainingSet& _testImg, std::function<void(TrainingSet&, std::vector<int>&)> _fit,
              std::function<std::vector<int>(TrainingSet&)> _predict): Metric{_testImg, _fit, _predict} {};
    double computeMetric(int label) override;
};


class Recall: public Metric {
public:
    Recall(TrainingSet& _testImg, std::function<void(TrainingSet&, std::vector<int>&)> _fit,
           std::function<std::vector<int>(TrainingSet&)> _predict): Metric{_testImg, _fit, _predict} {};
    double computeMetric(int label) override;
};


class Prevalence: public Metric {
public:
    Prevalence(TrainingSet& _testImg, std::function<void(TrainingSet&, std::vector<int>&)> _fit,
               std::function<std::vector<int>(TrainingSet&)> _predict): Metric{_testImg, _fit, _predict} {};
    double computeMetric(int label) override;
};