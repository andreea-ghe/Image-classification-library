#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "trainingset.h"

class Classifier {
private:
    TrainingSet trainSet;
public:
    Classifier(TrainingSet _trainSet): trainSet{_trainSet} {};

    virtual void fit(TrainingSet& trainImages, std::vector<int>& trainLabels) = 0;
    virtual std::vector<int> predict(TrainingSet& trainImages) = 0;

    virtual bool save(std::string filepath);
    virtual bool load(std::string filepath);

    virtual double eval(TrainingSet& trainImages) = 0;
    virtual ~Classifier() = default;

    TrainingSet& getTrainingSet() { return trainSet; };
    const TrainingSet& getTrainingSet() const { return trainSet; }

};