#include "metric.h"

Metric::Metric(TrainingSet& _testImg, std::function<void(TrainingSet&, std::vector<int>&)> _fit,
       std::function<std::vector<int>(TrainingSet&)> _predict):
        testImg{_testImg}, fit{_fit}, predict{_predict},
        trueNegative{0}, falsePositive{0}, falseNegative{0}, truePositive{0} {};

void Metric::computeConfusionMatrix(TrainingSet& _testImg) {

    std::vector<int>correctLabels;
    fit(_testImg,correctLabels);

    auto predictedLabels = predict(_testImg);

    for (int i = 0; i < 10; i ++)
        confusionMatrix.push_back(std::vector<int>(10));
    for (int i = 0; i < confusionMatrix.size(); i ++)
        for (int j = 0; j < confusionMatrix.at(i).size(); j ++)
            confusionMatrix.at(i).at(j) = 0;


    for (int i = 0 ; i < correctLabels.size(); i ++) {
        if (correctLabels.at(i) == predictedLabels.at(i))
            confusionMatrix.at(correctLabels.at(i)).at(correctLabels.at(i)) ++;
        else if (correctLabels.at(i) != predictedLabels.at(i))
            confusionMatrix.at(correctLabels.at(i)).at(predictedLabels.at(i)) ++;
    }

}


std::vector<std::vector<int>> Metric::getConfusionMatrix() {
    return confusionMatrix;
};


double Accuracy::computeMetric(int label) {

    truePositive = confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        falsePositive += confusionMatrix.at(i).at(label);
    falsePositive -= confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        falseNegative += confusionMatrix.at(label).at(i);
    falseNegative -= confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        for (int j = 0; j < 10; j ++)
            trueNegative += confusionMatrix.at(i).at(j);
    trueNegative -= falseNegative;
    trueNegative -= falsePositive;
    truePositive -= confusionMatrix.at(label).at(label);

    return static_cast<double>(truePositive + trueNegative) / (trueNegative+ falsePositive + falseNegative + truePositive);
}


double Precision::computeMetric(int label) {

    truePositive = confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        falsePositive += confusionMatrix.at(i).at(label);
    falsePositive -= confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        falseNegative += confusionMatrix.at(label).at(i);
    falseNegative -= confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        for (int j = 0; j < 10; j ++)
            trueNegative += confusionMatrix.at(i).at(j);
    trueNegative -= falseNegative;
    trueNegative -= falsePositive;
    truePositive -= confusionMatrix.at(label).at(label);

    return static_cast<double>(truePositive) / (falsePositive + truePositive);
}


double Recall::computeMetric(int label) {

    truePositive = confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        falsePositive += confusionMatrix.at(i).at(label);
    falsePositive -= confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        falseNegative += confusionMatrix.at(label).at(i);
    falseNegative -= confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        for (int j = 0; j < 10; j ++)
            trueNegative += confusionMatrix.at(i).at(j);
    trueNegative -= falseNegative;
    trueNegative -= falsePositive;
    truePositive -= confusionMatrix.at(label).at(label);

    return static_cast<double>(truePositive) / (truePositive + falseNegative);
}


double Prevalence::computeMetric(int label) {

    truePositive = confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        falsePositive += confusionMatrix.at(i).at(label);
    falsePositive -= confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        falseNegative += confusionMatrix.at(label).at(i);
    falseNegative -= confusionMatrix.at(label).at(label);

    for (int i = 0; i < 10; i ++)
        for (int j = 0; j < 10; j ++)
            trueNegative += confusionMatrix.at(i).at(j);
    trueNegative -= falseNegative;
    trueNegative -= falsePositive;
    truePositive -= confusionMatrix.at(label).at(label);

    return static_cast<double>(truePositive) / (truePositive + falsePositive);
}
