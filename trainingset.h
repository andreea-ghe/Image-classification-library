#pragma once
#include <vector>
#include <string>
#include <map>
#include <fstream>
using std::istream;
using std::ostream;

class TrainingSet {
protected:
    std::vector<std::vector<int>>images;
public:
    TrainingSet();
    TrainingSet(TrainingSet& _images);
    std::vector<std::vector<int>> getImages() const { return images; };
    void addRow(int n) { images.push_back(std::vector<int>(n)); };
    void changeElementAtPos(int i, int j, int newVal) { images.at(i).at(j) = newVal;};
    friend ostream& operator<<(ostream& os, const TrainingSet& trainImg);
    friend istream& operator>>(istream& is, TrainingSet& trainImg);
};