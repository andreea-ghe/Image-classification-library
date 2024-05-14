#include "classifier.h"
#include <iostream>


bool Classifier::load(std::string filepath) {
    std::ifstream fin(filepath);
    if (!fin.is_open())
        return false;
    fin >> trainSet;
    fin.close();
    return true;
}


bool Classifier::save(std::string filepath) {
    std::ofstream fout(filepath);
    if (!fout.is_open())
        return false;
    fout << trainSet;
    fout.close();
    return true;
}