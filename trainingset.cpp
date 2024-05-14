#include "trainingset.h"
#include <iostream>


TrainingSet::TrainingSet() {
    for (int i = 0; i < images.size(); i ++)
        for (int j = 0; j < images.at(i).size(); j ++)
            images.at(i).at(j) = 0;
}


TrainingSet::TrainingSet(TrainingSet &_images) {
    for (int i = 0; i < images.size(); i ++)
        for (int j = 0; j < images.at(i).size(); j ++)
            images.at(i).at(j) = _images.getImages().at(i).at(j);
}


ostream& operator<<(ostream& os, const TrainingSet& trainImg) {
    for (int i = 0; i < trainImg.getImages().size(); i ++, os << '\n')
        for (int j = 0; j < trainImg.getImages().at(i).size(); j++)
            os << trainImg.getImages().at(i).at(j) << ' ';
    return os;
}


istream& operator>>(istream& is, TrainingSet& trainImg) {
    int i = 0;
    std::string line;
    while(std::getline(is, line)) {
        if (i == 0) {
            i++;
            continue;
        }

        trainImg.addRow(785);
        // after making space, we add the elements from the line
        int j = 0;
        size_t pos = 0;
        std::string token;
        while ((pos = line.find(',')) != std::string::npos) {
            token = line.substr(0, pos);
            //std::cout << token << ' ';
            trainImg.changeElementAtPos(i - 1, j, stoi(token));
            line.erase(0, pos + 1);
            j++;
        }
        // we add the last element
        if (!line.empty()) {
            trainImg.changeElementAtPos(i - 1, j, stoi(line));
        }

        i++;
    }
    return is;
}