#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_LoadImage_clicked();

    void on_SetParameters_clicked();

    void on_GenerateShares_clicked();

    void on_ChooseScheme_clicked();

    void on_CombineShares_clicked();

    void on_SaveToFile_clicked();

    void on_nextButton_clicked();

    void on_previousButton_clicked();

    void on_combineSharesCheck_clicked();

    void on_fixImageCheck_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
