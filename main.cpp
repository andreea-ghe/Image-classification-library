#include <QApplication>
#include <QPushButton>
#include <QLabel>
#include <QMenu>
#include <QAction>
#include <QLineEdit>
#include <QIntValidator>
#include <QString>
#include <QtConcurrent>
#include <QFuture>
#include <QImage>
#include <QPixmap>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QVBoxLayout>
#include "knnclassifier.h"
#include "bayesclassifier.h"
#include "metric.h"
#include <functional>



int main(int argc, char *argv[]) {

    // we can only select a classifier after the selection of training and testing sets
    // in order to do that, we call the checkSelection function
    QApplication a(argc, argv);
    QWidget window;
    window.setWindowTitle("UI");
    window.resize(1500, 1000);

    std::string trainingPath = "";
    std::string testingPath = "";

    // we create the menus
    auto *menu1 = new QMenu("The options are:", &window);
    auto *opt1 = new QAction("train1.csv", menu1);
    auto *opt2 = new QAction("train2.csv", menu1);
    menu1->addAction(opt1);
    menu1->addAction(opt2);

    auto *trainingSet = new QLabel("Please select a training set :)\n", &window);
    trainingSet->move(50, 50);

    auto *menuButton1 = new QPushButton("Options:", &window);
    menuButton1->move(50, 80);
    menuButton1->resize(100, 30);

    auto *menu2 = new QMenu("The options are:", &window);
    auto *opt3 = new QAction("test1.csv", menu2);
    auto *opt4 = new QAction("test2.csv", menu2);
    menu2->addAction(opt3);
    menu2->addAction(opt4);

    auto *testingSet = new QLabel("Please select a testing set :)\n", &window);
    testingSet->move(50, 120);

    auto *menuButton2 = new QPushButton("Options:", &window);
    menuButton2->move(50, 150);
    menuButton2->resize(100, 30);



    // we show the options for classifier
    auto *classifier = new QLabel("Select the desired classifier", &window);
    classifier->move(50, 200);
    classifier->hide();

    // after selecting the training and testing sets we create a button to move forward
    auto *displayClassifiers = new QPushButton("NEXT", &window);
    displayClassifiers->move(50, 200);
    displayClassifiers->hide();

    // these are the buttons that are displayed after clicking on displayClassifiers
    auto *knnButton = new QPushButton("KNN Classifier", &window);
    knnButton->move(50, 230);
    knnButton->resize(150, 50);
    knnButton->hide();

    auto *bayesButton = new QPushButton("Naive Bayes Classifier", &window);
    bayesButton->move(210, 230);
    bayesButton->resize(150, 50);
    bayesButton->hide();

    auto checkSelection = [&]() {
        if (!trainingPath.empty() && !testingPath.empty()) {
            displayClassifiers->show();
            QObject::connect(displayClassifiers, &QPushButton::clicked, [&]() {
                displayClassifiers->hide();
                classifier->show();
                knnButton->show();
                bayesButton->show();
                menuButton1->setDisabled(true);
                menuButton2->setDisabled(true);
            });
        }
    };



    // load function
    auto *load = new QPushButton("LOAD", &window);
    load->move(50, 350);
    load->hide();



    // save function
    auto *save = new QPushButton("SAVE", &window);
    save->move(50, 380);
    save->hide();



    auto *classifyImg = new QPushButton("CLASSIFY", &window);
    classifyImg->move(50, 440);
    classifyImg->hide();


    auto *eval = new QPushButton("EVALUATE", &window);
    eval->move(50, 500);
    eval->hide();

    // if we click the knn button, bayes disappears
    int k = 0; // the k hyperparameter
    auto *whatK = new QLabel("&Select k: ", &window);
    whatK->move(50, 310);
    whatK->hide();
    auto *selectK = new QLineEdit(&window);
    selectK->move(100, 300);
    selectK->resize(50, 30);
    whatK->setBuddy(selectK);
    selectK->hide();
    auto *setK = new QPushButton("GO", &window);
    setK->move(160, 300);
    setK->resize(30, 30);
    setK->hide();

    auto *validator = new QIntValidator(1, 100, &window);
    selectK->setValidator(validator);

    QObject::connect(selectK, &QLineEdit::textChanged, [setK](const QString &text) {
        setK->setEnabled(!text.isEmpty());
    });

    // loading label to see progress for load
    auto *loadingLabel1 = new QLabel("Loading, please wait...", &window);
    loadingLabel1->move(50, 380);
    loadingLabel1->hide();

    // loading label to see progress for save
    auto *loadingLabel2 = new QLabel("Loading, please wait...", &window);
    loadingLabel2->move(50, 410);
    loadingLabel2->hide();



    // predicted number label
    QLabel *predictedNumber = new QLabel(&window);
    predictedNumber->move(50, 470);
    predictedNumber->hide();

    TrainingSet* trset1;
    KNNClassifier knn(*trset1, k);
    TrainingSet tstset1;
    QObject::connect(knnButton, &QPushButton::clicked, [&](){
        bayesButton->hide();
        whatK->show();
        selectK->show();
        setK->show();
        setK->setDisabled(true);
        knnButton->setDisabled(true);

        QObject::connect(setK, &QPushButton::clicked, [&]() {
            QString kValue = selectK->text();
            k = kValue.toInt();
            knn.setK(k);
            selectK->setEnabled(false);
            setK->setDisabled(true);
            load->show();

            QPushButton::connect(load, &QPushButton::clicked, [&](){
                loadingLabel1->show();
                QApplication::processEvents();

                QtConcurrent::run([&]() {
                    QThread::sleep(2);

                    knn.load(trainingPath);

                    std::ifstream fin(testingPath);
                    if (!fin.is_open()) {
                        qDebug() << "Failed to open test file";
                        return;
                    }
                    fin >> tstset1;
                    fin.close();

                    QMetaObject::invokeMethod(&window, [&]() {
                        loadingLabel1->hide();
                        load->setText("LOADED");
                        load->setDisabled(true);
                        save->show();
                        classifyImg->show();
                        eval->show();
                    }, Qt::QueuedConnection);

                });
            });

            QObject::connect(save, &QPushButton::clicked, [&]() {
                loadingLabel2->show();
                QApplication::processEvents();

                QtConcurrent::run([&]() {
                    QThread::sleep(2);
                    knn.save("train_out");
                    QMetaObject::invokeMethod(&window, [&]() {
                        loadingLabel2->hide();
                        save->setText("SAVED");
                    }, Qt::QueuedConnection);
                });
            });

            QObject::connect(classifyImg, &QPushButton::clicked, [&]() {
                srand(time(NULL));
                int n = knn.getTrainingSet().getImages().size();
                int random = rand() % n;
                // we pick a random image
                auto randImg = knn.getTrainingSet().getImages().at(random);
                unsigned char data[28*28];
                for (int i = 0; i < 28 * 28; i ++)
                    data[i] = randImg.at(i);
                QImage image(data, 28, 28, QImage::Format_Grayscale8);

                QLabel *imageLabel = new QLabel(&window);
                imageLabel->setFixedSize(28, 28);
                imageLabel->move(140, 439);
                imageLabel->setAlignment(Qt::AlignCenter);
                imageLabel->setFrameStyle(QFrame::Box | QFrame::Raised);

                imageLabel->setPixmap(QPixmap::fromImage(image));
                imageLabel->show();

                // we actually classify the image
                TrainingSet chosenImage;
                chosenImage.addRow(785);
                for (int i = 0; i < randImg.size(); i ++)
                    chosenImage.changeElementAtPos(0, i, randImg.at(i));


                std::vector<int>correctLabels;
                knn.fit(chosenImage,correctLabels);
                auto predictedLabels = knn.predict(chosenImage);

                QString predictedLabel = QString::fromStdString("Predicted number: " + std::to_string(predictedLabels.at(0)));
                predictedNumber->setText(predictedLabel);
                predictedNumber->show();
            });

            QObject::connect(eval, &QPushButton::clicked, [&]() {
                QString statistics1;
                QString statistics2;
                QString statistics3;
                std::string data1;
                std::string data2;
                std::string data3;

                Accuracy accuracy(tstset1,
                                  [&knn](TrainingSet& tstset1, std::vector<int>& aux){ knn.fit(tstset1, aux); },
                                  [&knn](TrainingSet& tstset1) -> std::vector<int> { return knn.predict(tstset1); });
                accuracy.computeConfusionMatrix(tstset1);

                Precision precision(tstset1,
                                    [&knn](TrainingSet& tstset1, std::vector<int>& aux){ knn.fit(tstset1, aux); },
                                    [&knn](TrainingSet& tstset1) -> std::vector<int> { return knn.predict(tstset1); });
                precision.computeConfusionMatrix(tstset1);

                Recall recall(tstset1,
                              [&knn](TrainingSet& tstset1, std::vector<int>& aux){ knn.fit(tstset1, aux); },
                              [&knn](TrainingSet& tstset1) -> std::vector<int> { return knn.predict(tstset1); });
                recall.computeConfusionMatrix(tstset1);

                Prevalence prevalence(tstset1,
                                      [&knn](TrainingSet& tstset1, std::vector<int>& aux){ knn.fit(tstset1, aux); },
                                      [&knn](TrainingSet& tstset1) -> std::vector<int> { return knn.predict(tstset1); });
                prevalence.computeConfusionMatrix(tstset1);

                for (int i = 0; i < 10; i ++) {

                    double acc = accuracy.computeMetric(i);
                    double prec = precision.computeMetric(i);
                    double rec = recall.computeMetric(i);
                    double prev = prevalence.computeMetric(i);
                    if (i < 5) {
                        data1 += "The accuracy for class " + std::to_string(i) + " is: " + std::to_string(acc) + '\n';
                        data1 += "The precision for class " + std::to_string(i) + " is: " + std::to_string(prec) + '\n';
                        data1 += "The reacall for class " + std::to_string(i) + " is: " + std::to_string(rec) + '\n';
                        data1 += "The prevalence for class " + std::to_string(i) + " is: " + std::to_string(prev) + '\n';
                        data1 += '\n';
                    }
                    else {
                        data2 += "The accuracy for class " + std::to_string(i) + " is: " + std::to_string(acc) + '\n';
                        data2 += "The precision for class " + std::to_string(i) + " is: " + std::to_string(prec) + '\n';
                        data2 += "The reacall for class " + std::to_string(i) + " is: " + std::to_string(rec) + '\n';
                        data2 += "The prevalence for class " + std::to_string(i) + " is: " + std::to_string(prev) + '\n';
                        data2 += '\n';
                    }
                }

                data3 += "Overall accuracy of this classifier algorithm is: " + std::to_string(knn.eval(tstset1)) + '\n';
                data3 += "This is the confusion matrix: ";

                auto confusionMatrix = accuracy.getConfusionMatrix();
                QTableWidget *table = new QTableWidget(confusionMatrix.size(), confusionMatrix.at(0).size(), &window);
                for (int i = 0; i < confusionMatrix.size(); i ++) {
                    for (int j = 0; j < confusionMatrix.at(i).size(); j ++) {
                        QTableWidgetItem *item = new QTableWidgetItem(QString::number(confusionMatrix.at(i).at(j)));
                        item->setTextAlignment(Qt::AlignCenter);

                        int value = confusionMatrix.at(i).at(j);
                        int intensity = std::min(255, value * 10);
                        item->setData(Qt::BackgroundRole, QColor(255, 200 - intensity, 200 - intensity));

                        table->setItem(i, j, item);
                    }
                }

                table->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
                table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
                table->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);


                statistics1 = QString::fromStdString(data1);
                statistics2 = QString::fromStdString(data2);
                statistics3 = QString::fromStdString(data3);

                QLabel *metric1 = new QLabel(statistics1, &window);
                metric1->move(250, 50);
                metric1->show();

                QLabel *metric2 = new QLabel(statistics2, &window);
                metric2->move(500, 50);
                metric2->show();

                QLabel *metric3 = new QLabel(statistics3, &window);
                metric3->move(750, 50);
                metric3->show();

                table->move(750, 100);
                table->show();
            });

        });
    });

    // we click on bayes button, knn disappears
    TrainingSet *trset2;
    BayesClassifier bayes(*trset2);
    TrainingSet tstset2;
    QObject::connect(bayesButton, &QPushButton::clicked, [&]() {
        knnButton->hide();
        bayesButton->move(50, 230);
        bayesButton->setDisabled(true);
        load->show();

        QPushButton::connect(load, &QPushButton::clicked, [&]() {
            loadingLabel1->show();
            QApplication::processEvents();

            QtConcurrent::run([&]() {
                QThread::sleep(2);

                bayes.load(trainingPath);

                std::ifstream fin(testingPath);
                if (!fin.is_open()) {
                    qDebug() << "Failed to open test file";
                    return;
                }

                fin >> tstset2;
                fin.close();

                QMetaObject::invokeMethod(&window, [&]() {
                    loadingLabel1->hide();
                    load->setText("LOADED");
                    load->setDisabled(true);
                    save->show();
                    classifyImg->show();
                    eval->show();
                }, Qt::QueuedConnection);
            });
        });

        QObject::connect(save, &QPushButton::clicked, [&]() {
            loadingLabel2->show();
            QApplication::processEvents();

            QtConcurrent::run([&]() {
                QThread::sleep(2);
                bayes.save("train_out");
                QMetaObject::invokeMethod(&window, [&]() {
                    loadingLabel2->hide();
                    save->setText("SAVED");
                }, Qt::QueuedConnection);
            });
        });

        QObject::connect(classifyImg, &QPushButton::clicked, [&]() {

            srand(time(NULL));
            int n = bayes.getTrainingSet().getImages().size();
            int random = rand() % n;
            // we pick a random image
            auto randImg = bayes.getTrainingSet().getImages().at(random);
            unsigned char data[28*28];
            for (int i = 0; i < 28 * 28; i ++)
                data[i] = randImg.at(i);
            QImage image(data, 28, 28, QImage::Format_Grayscale8);

            QLabel *imageLabel = new QLabel(&window);
            imageLabel->setFixedSize(28, 28);
            imageLabel->move(140, 439);
            imageLabel->setAlignment(Qt::AlignCenter);
            imageLabel->setFrameStyle(QFrame::Box | QFrame::Raised);

            imageLabel->setPixmap(QPixmap::fromImage(image));
            imageLabel->show();

            // we actually classify the image
            TrainingSet chosenImage;
            chosenImage.addRow(785);
            for (int i = 0; i < randImg.size(); i ++)
                chosenImage.changeElementAtPos(0, i, randImg.at(i));

            std::vector<int>correctLabels;
            bayes.fit(chosenImage,correctLabels);
            auto predictedLabels = bayes.predict(chosenImage);

            QString predictedLabel = QString::fromStdString("Predicted number: " + std::to_string(predictedLabels.at(0)));
            predictedNumber->setText(predictedLabel);
            predictedNumber->show();

        });

        QObject::connect(eval, &QPushButton::clicked, [&]() {

            QString statistics1;
            QString statistics2;
            QString statistics3;
            std::string data1;
            std::string data2;
            std::string data3;

            Accuracy accuracy(tstset2,
                              [&bayes](TrainingSet& tstset2, std::vector<int>& aux){ bayes.fit(tstset2, aux); },
                              [&bayes](TrainingSet& tstset2) -> std::vector<int> { return bayes.predict(tstset2); });
            accuracy.computeConfusionMatrix(tstset2);

            Precision precision(tstset2,
                                [&bayes](TrainingSet& tstset2, std::vector<int>& aux){ bayes.fit(tstset2, aux); },
                                [&bayes](TrainingSet& tstset2) -> std::vector<int> { return bayes.predict(tstset2); });
            precision.computeConfusionMatrix(tstset2);

            Recall recall(tstset2,
                          [&bayes](TrainingSet& tstset2, std::vector<int>& aux){ bayes.fit(tstset2, aux); },
                          [&bayes](TrainingSet& tstset2) -> std::vector<int> { return bayes.predict(tstset2); });
            recall.computeConfusionMatrix(tstset2);

            Prevalence prevalence(tstset2,
                                  [&bayes](TrainingSet& tstset2, std::vector<int>& aux){ bayes.fit(tstset2, aux); },
                                  [&bayes](TrainingSet& tstset2) -> std::vector<int> { return bayes.predict(tstset2); });
            prevalence.computeConfusionMatrix(tstset2);

            for (int i = 0; i < 10; i ++) {

                double acc = accuracy.computeMetric(i);
                double prec = precision.computeMetric(i);
                double rec = recall.computeMetric(i);
                double prev = prevalence.computeMetric(i);
                if (i < 5) {
                    data1 += "The accuracy for class " + std::to_string(i) + " is: " + std::to_string(acc) + '\n';
                    data1 += "The precision for class " + std::to_string(i) + " is: " + std::to_string(prec) + '\n';
                    data1 += "The recall for class " + std::to_string(i) + " is: " + std::to_string(rec) + '\n';
                    data1 += "The prevalence for class " + std::to_string(i) + " is: " + std::to_string(prev) + '\n';
                    data1 += '\n';
                }
                else {
                    data2 += "The accuracy for class " + std::to_string(i) + " is: " + std::to_string(acc) + '\n';
                    data2 += "The precision for class " + std::to_string(i) + " is: " + std::to_string(prec) + '\n';
                    data2 += "The recall for class " + std::to_string(i) + " is: " + std::to_string(rec) + '\n';
                    data2 += "The prevalence for class " + std::to_string(i) + " is: " + std::to_string(prev) + '\n';
                    data2 += '\n';
                }
            }

            data3 += "Overall accuracy of this classifier algorithm is: " + std::to_string(bayes.eval(tstset2)) + '\n';
            data3 += "This is the confusion matrix: ";

            auto confusionMatrix = accuracy.getConfusionMatrix();
            auto *table = new QTableWidget(confusionMatrix.size(), confusionMatrix.at(0).size(), &window);
            for (int i = 0; i < confusionMatrix.size(); i ++) {
                for (int j = 0; j < confusionMatrix.at(i).size(); j ++) {
                    auto *item = new QTableWidgetItem(QString::number(confusionMatrix.at(i).at(j)));
                    item->setTextAlignment(Qt::AlignCenter);

                    int value = confusionMatrix.at(i).at(j);
                    int intensity = std::min(255, value * 10);
                    item->setData(Qt::BackgroundRole, QColor(255, 200 - intensity, 200 - intensity));

                    table->setItem(i, j, item);
                }
            }

            table->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
            table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
            table->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);


            statistics1 = QString::fromStdString(data1);
            statistics2 = QString::fromStdString(data2);
            statistics3 = QString::fromStdString(data3);

            QLabel *metric1 = new QLabel(statistics1, &window);
            metric1->move(250, 50);
            metric1->show();

            QLabel *metric2 = new QLabel(statistics2, &window);
            metric2->move(500, 50);
            metric2->show();

            QLabel *metric3 = new QLabel(statistics3, &window);
            metric3->move(750, 50);
            metric3->show();

            table->move(750, 100);
            table->show();
        });

    });



    // select training set
    QObject::connect(menuButton1, &QPushButton::clicked, [&](){
        menu1->popup(window.mapToGlobal(menuButton1->pos()));
    });

    QObject::connect(opt1, &QAction::triggered, [&](){
        trainingPath = "train1.csv";
        menuButton1->setText("train1.csv");
        checkSelection();
    });

    QObject::connect(opt2, &QAction::triggered, [&]() {
        trainingPath = "train2.csv";
        menuButton1->setText("train2.csv");
        checkSelection();
    });



    // select testing set
    QObject::connect(menuButton2, &QPushButton::clicked, [&](){
        menu2->popup(window.mapToGlobal(menuButton2->pos()));
    });

    QObject::connect(opt3, &QAction::triggered, [&](){
        testingPath = "test1.csv";
        menuButton2->setText("test1.csv");
        checkSelection();
    });

    QObject::connect(opt4, &QAction::triggered, [&]() {
        testingPath = "test2.csv";
        menuButton2->setText("test2.csv");
        checkSelection();
    });

    window.show();
    return QApplication::exec();
}
