from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QDialog, QMessageBox
from PyQt5.QtGui import QIntValidator
import script_classificationRF



class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        """MainWindow constructor."""
        super().__init__()
        self.resize(529, 772)
        self.setWindowTitle("Random Forest Classifier")

        #menubar
        self.menubar = QtWidgets.QMenuBar()
        self.menubar.setGeometry(QtCore.QRect(0, 0, 504, 20))

        self.About = QtWidgets.QMenu(self.menubar)
        self.About.setTitle("About")
        self.setMenuBar(self.menubar)
        self.actionsub = QtWidgets.QAction(self)
        self.actionsub.setText("Random Forest Documentation")
        self.actionsub2 = QtWidgets.QAction(self)
        self.actionsub2.setText("Outputs")
        self.About.addAction(self.actionsub)
        self.About.addAction(self.actionsub2)
        self.menubar.addAction(self.About.menuAction())
        self.actionsub.triggered.connect(self.on_about)
        self.actionsub2.triggered.connect(self.on_aboutoutputs)
        

        #centralwidget
        self.centralwidget = QtWidgets.QWidget()
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.gridLayout_5 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        


        #groupboxes
        self.groupBox_inputs = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_inputs.setTitle("Inputs")
        self.gridLayout_1 = QtWidgets.QGridLayout(self.groupBox_inputs)

        self.groupBox_clfMode = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_clfMode.setTitle("Select Classification Mode")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_clfMode)
        
        #grids
        
        
        
        #self.gridLayout_3 = QtWidgets.QGridLayout()
        #self.gridLayout_3.setContentsMargins(0, 0, 0, 0)

        
        #labels
        self.label_progTitle = QtWidgets.QLabel(self.layoutWidget,text = "Random Forest Classifier\n v1.0", font=QtGui.QFont('Sans',25))
        self.label_progTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.label_inpImg = QtWidgets.QLabel(self.groupBox_inputs,text = "Input Image")
        self.label_inpBlocks = QtWidgets.QLabel(self.groupBox_inputs,text = "Number of Blocks ( >1)")
        self.label_outImgC = QtWidgets.QLabel(self.groupBox_inputs,text = "Output Image (Class)")
        self.label_outImgP = QtWidgets.QLabel(self.groupBox_inputs,text = "Ouput Image (Probabilities)")
        self.label_importTrain = QtWidgets.QLabel(self.groupBox_clfMode,text = "Import training data")
        self.label_trainVar = QtWidgets.QLabel(self.groupBox_clfMode,text = "Training Variables")
        self.label_trainLabel = QtWidgets.QLabel(self.groupBox_clfMode,text = "Training Labels")
        self.label_importClf = QtWidgets.QLabel(self.groupBox_clfMode,text = "Import Classifier")
        
        #buttons
        self.browseInpImg = QtWidgets.QPushButton(self.groupBox_inputs,text = "Browse")
        self.saveOutImgC = QtWidgets.QPushButton(self.groupBox_inputs,text = "Save")
        self.saveOutImgP = QtWidgets.QPushButton(self.groupBox_inputs,text = "Save")
        self.browseTrain = QtWidgets.QPushButton(self.groupBox_clfMode,text = "Browse")
        self.browseClf = QtWidgets.QPushButton(self.groupBox_clfMode,text = "Browse")
        self.selectVars = QtWidgets.QPushButton(self.groupBox_clfMode,text = "Select From List")
        self.setClfParameters = QtWidgets.QPushButton(text = "Set Random Forest Parameters")
        self.runClf = QtWidgets.QPushButton(text = "Run")

        #lineEdits
        self.lineEdit_inpImg = QtWidgets.QLineEdit(self.groupBox_inputs)
        self.lineEdit_inpBlocks = QtWidgets.QLineEdit(self.groupBox_inputs)
        self.lineEdit_outImgC = QtWidgets.QLineEdit(self.groupBox_inputs)
        self.lineEdit_outImgP = QtWidgets.QLineEdit(self.groupBox_inputs)
        self.lineEdit_importTrain = QtWidgets.QLineEdit(self.groupBox_clfMode)
        self.lineEdit_trainVar = QtWidgets.QLineEdit(self.groupBox_clfMode)
        self.lineEdit_trainLabel = QtWidgets.QLineEdit(self.groupBox_clfMode)
        self.lineEdit_importClf = QtWidgets.QLineEdit(self.groupBox_clfMode)

        #radiobuttons
        self.radioButton_train = QtWidgets.QRadioButton(self.groupBox_clfMode)
        self.radioButton_train.setText("Train Classifier")
        self.radioButton_importClf = QtWidgets.QRadioButton(self.groupBox_clfMode)
        self.radioButton_importClf.setText("Import Classifier")        

    


        #layout

        self.gridLayout_5.addWidget(self.label_progTitle, 0, 0, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox_inputs, 1, 0, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox_clfMode, 2, 0, 1, 1)
        self.gridLayout_5.addWidget(self.setClfParameters, 3, 0, 1, 1)
        self.gridLayout_5.addWidget(self.runClf, 4, 0, 1, 1)
        
        self.verticalLayout.addWidget(self.splitter)
        self.setCentralWidget(self.centralwidget)
        
        
        
        
        

        ##start of group box input##
      
        self.gridLayout_1.addWidget(self.label_inpImg,0,0,1,1)
        self.gridLayout_1.addWidget(self.lineEdit_inpImg,0,1,1,1)
        self.gridLayout_1.addWidget(self.browseInpImg,0,2,1,1)
        
        self.gridLayout_1.addWidget(self.label_inpBlocks,1,0,1,1)
        self.gridLayout_1.addWidget(self.lineEdit_inpBlocks,1,1,1,1)
        
        self.gridLayout_1.addWidget(self.label_outImgC,2,0,1,1)
        self.gridLayout_1.addWidget(self.lineEdit_outImgC,2,1,1,1)
        self.gridLayout_1.addWidget(self.saveOutImgC,2,2,1,1)      
       
        self.gridLayout_1.addWidget(self.label_outImgP,3,0,1,1)
        self.gridLayout_1.addWidget(self.lineEdit_outImgP,3,1,1,1)
        self.gridLayout_1.addWidget(self.saveOutImgP,3,2,1,1)

       ##end of groupbox input##

       ##start of groupbox clfmode##

        self.gridLayout_2.addWidget(self.radioButton_train,0,0,1,1)
        
        self.gridLayout_2.addWidget(self.label_importTrain,1,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_importTrain,1,1,1,1)
        self.gridLayout_2.addWidget(self.browseTrain,1,2,1,1)
        
        self.gridLayout_2.addWidget(self.label_trainVar,2,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_trainVar,2,1,1,1)
        self.gridLayout_2.addWidget(self.selectVars,2,2,1,1)
        
        self.gridLayout_2.addWidget(self.label_trainLabel,3,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_trainLabel,3,1,1,1)

        self.gridLayout_2.addWidget(self.radioButton_importClf,4,0,1,1)
        
        self.gridLayout_2.addWidget(self.label_importClf,5,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_importClf,5,1,1,1)
        self.gridLayout_2.addWidget(self.browseClf,5,2,1,1)

        ##end of groupbox clfmode##
       



        #other stuff
        self.variables = ''
        self.labels = ''

        #limit nblocs to only integers
        self.onlyInt = QIntValidator(2, 100000)
        self.lineEdit_inpBlocks.setValidator(self.onlyInt)
        self.lineEdit_importTrain.setEnabled(False)
        self.lineEdit_trainVar.setEnabled(False)
        self.lineEdit_trainLabel.setEnabled(False)
        self.lineEdit_importClf.setEnabled(False)
        self.browseTrain.setEnabled(False)
        self.selectVars.setEnabled(False)
        self.browseClf.setEnabled(False)
        self.runClf.setEnabled(False)

        #click and connect
        self.radioButton_train.toggled.connect(self.importTraining)
        self.radioButton_importClf.toggled.connect(self.importClassifier)
        self.browseInpImg.clicked.connect(self.selectFile)
        self.saveOutImgC.clicked.connect(self.saveFileC)
        self.saveOutImgP.clicked.connect(self.saveFileP)
        self.browseTrain.clicked.connect(self.selectFileTrain)
        self.browseClf.clicked.connect(self.selectFileClf)
        self.selectVars.clicked.connect(self.selectFromList)
        self.setClfParameters.clicked.connect(self.pressedSelectClfPar)
        self.runClf.clicked.connect(self.pressedRun)


        #initialize clf parameters as empty, in case user wants to run with default
        script_classificationRF.estimators_user=''
        script_classificationRF.maxfeatures_user=''
        script_classificationRF.njobs_user=''
        script_classificationRF.bootstrap_user=''
        script_classificationRF.criterion_user=''
        script_classificationRF.max_depth_user=''
        script_classificationRF.minsamplessplit_user=''
        script_classificationRF.minsamplesleaf_user=''
        script_classificationRF.minweightleaf_user=''
        script_classificationRF.maxleafnodes_user=''
        script_classificationRF.minimpdecrease_user=''
        script_classificationRF.minimpsplit_user=''
        script_classificationRF.oobscore_user=''
        #script_classificationRF.randomstate_user=self.lineEdit_randomstate.text() #to be implemented
        script_classificationRF.verbose_user=''
        script_classificationRF.warmstart_user=''
        #script_classificationRF.classweight_user=self.lineEdit_classweight.text() # to be implemented
        script_classificationRF.ccpalpha_user=''
        script_classificationRF.maxsamples_user=''
        
    def importTraining(self, enabled):
        if enabled:
            self.lineEdit_importTrain.setEnabled(True)
            self.lineEdit_trainVar.setEnabled(True)
            self.lineEdit_trainLabel.setEnabled(True)
            self.lineEdit_importClf.setEnabled(False)
            self.browseTrain.setEnabled(True)
            self.selectVars.setEnabled(True)
            self.browseClf.setEnabled(False)
            self.runClf.setEnabled(True)
            
    def importClassifier(self, enabled):
        if enabled:
            self.lineEdit_importClf.setEnabled(True)
            self.lineEdit_importTrain.setEnabled(False)
            self.lineEdit_trainVar.setEnabled(False)
            self.lineEdit_trainLabel.setEnabled(False)
            self.browseTrain.setEnabled(False)
            self.selectVars.setEnabled(False)
            self.browseClf.setEnabled(True)
            self.runClf.setEnabled(True)

    def selectFile(self):

        filename = QFileDialog.getOpenFileName(None, "Browse input image", "", "Image (*.tif *.png *.jpg *.bmp *.jpeg)")
        self.lineEdit_inpImg.setText(filename[0])            
        
    def saveFileC(self):

        filename = QFileDialog.getSaveFileName(None, "Save output image (class)", "", "Image (*.tif *.png *.jpg *.bmp *.jpeg)")
        self.lineEdit_outImgC.setText(filename[0])
        
    def saveFileP(self):

        filename = QFileDialog.getSaveFileName(None, "Save output image (prob.)", "", "Image (*.tif *.png *.jpg *.bmp *.jpeg)")
        self.lineEdit_outImgP.setText(filename[0])

    def selectFileTrain(self):

        filename = QFileDialog.getOpenFileName(None, "Browse train file", "", "CSV (*.csv)")
        self.lineEdit_importTrain.setText(filename[0])

    def selectFileClf(self):

        filename = QFileDialog.getOpenFileName(None, "Browse classifier", "")
        self.lineEdit_importClf.setText(filename[0])

    @QtCore.pyqtSlot(str,str)
    def setVarLabels(self, variables, labels):

        self.variables = variables
        self.labels = labels
        self.lineEdit_trainVar.setText(self.variables)
        self.lineEdit_trainLabel.setText(self.labels)

    def selectFromList(self):
        #'C:/Users/Daniel/Desktop/NOVA IMS/Thesis/DGT/Sampling/treino_75.csv'        
        script_classificationRF.treino=self.lineEdit_importTrain.text()
        #print(script_classificationRF.getColumns())

        
        self.dialog = DialogWindow(script_classificationRF.getColumns())
        self.dialog.submitted.connect(self.setVarLabels)
        self.dialog.show()


    def pressedSelectClfPar(self):
        self.dialog2 = DialogWindow2()
        #self.dialog2
        self.dialog2.show()


        

    def pressedRun(self):

        script_classificationRF.fn=self.lineEdit_inpImg.text()
        script_classificationRF.blocks=int(self.lineEdit_inpBlocks.text())
        script_classificationRF.outRaster=self.lineEdit_outImgC.text()
        script_classificationRF.outRaster2=self.lineEdit_outImgP.text()

        #pass classification mode
        if self.radioButton_train.isChecked():
            script_classificationRF.clfMode = 1
            script_classificationRF.variables = self.lineEdit_trainVar.text().split(sep=',')
            script_classificationRF.labelCod = self.lineEdit_trainLabel.text()
        else:
            script_classificationRF.clfMode = 2
            script_classificationRF.clfPickle = self.lineEdit_importClf.text()

        

        script_classificationRF.runScript()

        
    def on_about(self):
        msg = QMessageBox()
        msg.setWindowTitle('About')
        msg.setText('This Random Forest implementation is based on Python Scikit-learn. Detailed information can be found at the following link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html')
        msg.setTextFormat(QtCore.Qt.MarkdownText)
        msg.setIcon(QMessageBox.Information)
        x = msg.exec_()
        
    def on_aboutoutputs(self):
        self.dialog3 = DialogWindow3()
        self.dialog3.show()
    
    
        

class DialogWindow(QtWidgets.QWidget):

    submitted = QtCore.pyqtSignal(str, str)

    def __init__(self, listitems):
        super().__init__()
        self.resize(777, 566)
        self.setWindowTitle("Select Variables and Labels")
        self.list1 = QtWidgets.QListWidget()
        self.list1.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
       
        self.list2 = QtWidgets.QListWidget()

        i = 0
        for elem in listitems:
            
            item = QtWidgets.QListWidgetItem()
            self.list1.addItem(item)
            item2 = QtWidgets.QListWidgetItem()
            self.list2.addItem(item2)
            item = self.list1.item(i)
            item2 = self.list2.item(i)
            item.setText(str(elem))
            item2.setText(str(elem))
            i = i+1
        i = None
                          
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.submit_button = QtWidgets.QPushButton('Submit')

        self.setLayout(QtWidgets.QFormLayout())
        self.layout().addRow('Select Variables', self.list1)
        self.layout().addRow('Select Label', self.list2)
        buttons = QtWidgets.QWidget()
        buttons.setLayout(QtWidgets.QHBoxLayout())
        buttons.layout().addWidget(self.cancel_button)
        buttons.layout().addWidget(self.submit_button)
        self.layout().addRow('', buttons)

        self.submit_button.clicked.connect(self.on_submit)
        self.cancel_button.clicked.connect(self.close)


    def on_submit(self):
        
        #convert to string
        variables_array = [x.data() for x in self.list1.selectedIndexes()]
        variables_str = ''
        for elem in variables_array:
            variables_str = variables_str + str(elem)+ ','

        variables_str = variables_str[:-1]

        labels_array = [x.data() for x in self.list2.selectedIndexes()]
        labels_str = str(labels_array[0])
        
        
        self.submitted.emit(variables_str,labels_str)
        self.close()


class DialogWindow2(QtWidgets.QWidget):

    #submitted = QtCore.pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.resize(300, 566)
        self.setWindowTitle("Classification Parameters")

        self.groupboxMain = QtWidgets.QGroupBox()
        self.groupboxMain.setTitle('Main')
        self.groupboxSecondary = QtWidgets.QGroupBox()
        self.groupboxSecondary.setTitle('Secondary')
        self.groupboxObs = QtWidgets.QGroupBox()
        self.groupboxObs.setTitle('Observations')

        self.gridLayout_1 = QtWidgets.QGridLayout(self.groupboxMain)
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupboxSecondary)
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupboxObs)

        self.n_estimators_label = QtWidgets.QLabel(self.groupboxMain,text="Number of trees")
        self.lineEdit_estimators = QtWidgets.QLineEdit(self.groupboxMain)
        self.max_features_label = QtWidgets.QLabel(self.groupboxMain,text="Max features")
        self.lineEdit_maxfeatures = QtWidgets.QLineEdit(self.groupboxMain)
        self.njobs_label = QtWidgets.QLabel(self.groupboxMain,text="Number of jobs (processor usage)")
        self.lineEdit_njobs = QtWidgets.QLineEdit(self.groupboxMain)
        
        
        self.bootstrap_label = QtWidgets.QLabel(self.groupboxSecondary,text="Bootstrap")
        self.lineEdit_bootstrap = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.criterion_label = QtWidgets.QLabel(self.groupboxSecondary,text="Criterion")
        self.lineEdit_criterion = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.maxdepth_label = QtWidgets.QLabel(self.groupboxSecondary,text="Max. depth")
        self.lineEdit_maxdepth = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.minsamplessplit_label = QtWidgets.QLabel(self.groupboxSecondary,text="Min. samples split")
        self.lineEdit_minsamplessplit = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.minsamplesleaf_label = QtWidgets.QLabel(self.groupboxSecondary,text="Min. samples leaf")
        self.lineEdit_minsamplesleaf = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.minweightfleaf_label = QtWidgets.QLabel(self.groupboxSecondary,text="Min. weight fraction leaf")
        self.lineEdit_minweightfleaf = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.maxleafnodes_label = QtWidgets.QLabel(self.groupboxSecondary,text="Max. leaf nodes")
        self.lineEdit_maxleafnodes = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.minimpdecrease_label = QtWidgets.QLabel(self.groupboxSecondary,text="Min. impurity decrease")
        self.lineEdit_minimpdecrease = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.minimpsplit_label = QtWidgets.QLabel(self.groupboxSecondary,text="Min. impurity split")
        self.lineEdit_minimpsplit = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.oobscore_label = QtWidgets.QLabel(self.groupboxSecondary,text="Out of bag score")
        self.lineEdit_oobscore = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.randomstate_label = QtWidgets.QLabel(self.groupboxSecondary,text="Random state*")
        self.lineEdit_randomstate = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.verbose_label = QtWidgets.QLabel(self.groupboxSecondary,text="Verbose")
        self.lineEdit_verbose = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.warmstart_label = QtWidgets.QLabel(self.groupboxSecondary,text="Warm start")
        self.lineEdit_warmstart = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.classweight_label = QtWidgets.QLabel(self.groupboxSecondary,text="Class weight*")
        self.lineEdit_classweight = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.ccpalpha_label = QtWidgets.QLabel(self.groupboxSecondary,text="CCP alpha")
        self.lineEdit_ccpalpha = QtWidgets.QLineEdit(self.groupboxSecondary)
        self.maxsamples_label = QtWidgets.QLabel(self.groupboxSecondary,text="Max. samples")
        self.lineEdit_maxsamples = QtWidgets.QLineEdit(self.groupboxSecondary)

        self.obs1_label = QtWidgets.QLabel(self.groupboxObs,text="1) Leave empty to use default")
        self.obs2_label = QtWidgets.QLabel(self.groupboxObs,text="2) Use Number of jobs = -1 to use all CPUs")
        self.obs3_label = QtWidgets.QLabel(self.groupboxObs,text="\n*This version does not support setting such parameters")
        

        self.gridLayout_1.addWidget(self.n_estimators_label,0,0,1,1)
        self.gridLayout_1.addWidget(self.lineEdit_estimators,0,1,1,1)
        self.gridLayout_1.addWidget(self.max_features_label,1,0,1,1)
        self.gridLayout_1.addWidget(self.lineEdit_maxfeatures,1,1,1,1)
        self.gridLayout_1.addWidget(self.njobs_label,2,0,1,1)
        self.gridLayout_1.addWidget(self.lineEdit_njobs,2,1,1,1)

        self.gridLayout_2.addWidget(self.bootstrap_label,0,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_bootstrap,0,1,1,1)
        self.gridLayout_2.addWidget(self.criterion_label,1,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_criterion,1,1,1,1)
        self.gridLayout_2.addWidget(self.maxdepth_label,2,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_maxdepth,2,1,1,1)
        self.gridLayout_2.addWidget(self.minsamplessplit_label,3,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_minsamplessplit,3,1,1,1)
        self.gridLayout_2.addWidget(self.minsamplesleaf_label,4,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_minsamplesleaf,4,1,1,1)
        self.gridLayout_2.addWidget(self.minweightfleaf_label,5,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_minweightfleaf,5,1,1,1)
        self.gridLayout_2.addWidget(self.maxleafnodes_label,6,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_maxleafnodes,6,1,1,1)
        self.gridLayout_2.addWidget(self.minimpdecrease_label,7,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_minimpdecrease,7,1,1,1)
        self.gridLayout_2.addWidget(self.minimpsplit_label,8,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_minimpsplit,8,1,1,1)
        self.gridLayout_2.addWidget(self.oobscore_label,9,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_oobscore,9,1,1,1)
        self.gridLayout_2.addWidget(self.randomstate_label,10,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_randomstate,10,1,1,1)
        self.gridLayout_2.addWidget(self.verbose_label,11,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_verbose,11,1,1,1)
        self.gridLayout_2.addWidget(self.warmstart_label,12,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_warmstart,12,1,1,1)
        self.gridLayout_2.addWidget(self.classweight_label,13,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_classweight,13,1,1,1)
        self.gridLayout_2.addWidget(self.ccpalpha_label,14,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_ccpalpha,14,1,1,1)
        self.gridLayout_2.addWidget(self.maxsamples_label,15,0,1,1)
        self.gridLayout_2.addWidget(self.lineEdit_maxsamples,15,1,1,1)

        self.gridLayout_3.addWidget(self.obs1_label,0,0,1,1)
        self.gridLayout_3.addWidget(self.obs2_label,1,0,1,1)
        self.gridLayout_3.addWidget(self.obs3_label,3,0,1,1)

                          
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.submit_button = QtWidgets.QPushButton('Submit')
        self.help_button = QtWidgets.QPushButton('Help')

        self.setLayout(QtWidgets.QFormLayout())
        self.layout().addRow(self.groupboxMain)
        self.layout().addRow(self.groupboxSecondary)
        self.layout().addRow(self.groupboxObs)
        
        
        #self.layout().addRow(self.n_estimators_label)
        #self.layout().addRow(self.lineEdit_estimators)
        #self.layout().addRow(self.max_features_label)
        #self.layout().addRow(self.lineEdit_maxfeatures)
        #self.layout().addRow(self.bootstrap_label)
        #self.layout().addRow(self.lineEdit_bootstrap)
        
        buttons = QtWidgets.QWidget()
        buttons.setLayout(QtWidgets.QHBoxLayout())
        buttons.layout().addWidget(self.cancel_button)
        buttons.layout().addWidget(self.submit_button)
        buttons.layout().addWidget(self.help_button)
        self.layout().addRow('', buttons)

        self.submit_button.clicked.connect(self.on_submit)
        self.cancel_button.clicked.connect(self.close)
        self.help_button.clicked.connect(self.on_help)

        #other configs
        self.lineEdit_randomstate.setEnabled(False)
        self.lineEdit_classweight.setEnabled(False)
        self.onlyInt = QIntValidator(1, 100000)
        self.onlyIntjobs = QIntValidator(-1, 100000)
        self.lineEdit_estimators.setValidator(self.onlyInt)
        self.lineEdit_maxdepth.setValidator(self.onlyInt)
        self.lineEdit_maxleafnodes.setValidator(self.onlyInt)
        self.lineEdit_njobs.setValidator(self.onlyIntjobs)
        self.lineEdit_verbose.setValidator(self.onlyInt)
        


    def on_submit(self):
        
        script_classificationRF.estimators_user=self.lineEdit_estimators.text()
        script_classificationRF.maxfeatures_user=self.lineEdit_maxfeatures.text()
        script_classificationRF.njobs_user=self.lineEdit_njobs.text()
        script_classificationRF.bootstrap_user=self.lineEdit_bootstrap.text()
        script_classificationRF.criterion_user=self.lineEdit_criterion.text()
        script_classificationRF.max_depth_user=self.lineEdit_maxdepth.text()
        script_classificationRF.minsamplessplit_user=self.lineEdit_minsamplessplit.text()
        script_classificationRF.minsamplesleaf_user=self.lineEdit_minsamplesleaf.text()
        script_classificationRF.minweightleaf_user=self.lineEdit_minweightfleaf.text()
        script_classificationRF.maxleafnodes_user=self.lineEdit_maxleafnodes.text()
        script_classificationRF.minimpdecrease_user=self.lineEdit_minimpdecrease.text()
        script_classificationRF.minimpsplit_user=self.lineEdit_minimpsplit.text()
        script_classificationRF.oobscore_user=self.lineEdit_oobscore.text()
        #script_classificationRF.randomstate_user=self.lineEdit_randomstate.text() #to be implemented
        script_classificationRF.verbose_user=self.lineEdit_verbose.text()
        script_classificationRF.warmstart_user=self.lineEdit_warmstart.text()
        #script_classificationRF.classweight_user=self.lineEdit_classweight.text() # to be implemented
        script_classificationRF.ccpalpha_user=self.lineEdit_ccpalpha.text()
        script_classificationRF.maxsamples_user=self.lineEdit_maxsamples.text()
        
        self.close()

    def on_help(self):
        msg = QMessageBox()
        msg.setWindowTitle('Help')
        msg.setText('This Random Forest implementation is based on Python Scikit-learn. Detailed information can be found at the following link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html')
        msg.setTextFormat(QtCore.Qt.MarkdownText)
        msg.setIcon(QMessageBox.Information)
        x = msg.exec_()
        
class DialogWindow3(QtWidgets.QWidget):

    #submitted = QtCore.pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.resize(300, 150)
        self.setWindowTitle("About outputs")

        self.text_label = QtWidgets.QLabel(text="1) Output Class: The program will generate a raster file with 2 bands, where band 1 is the predicted class (most tree votes) and band 2 is the second most voted class\n")
        self.text_label.setWordWrap(True)
        self.text2_label = QtWidgets.QLabel(text="2) Output Probabilities: The program will output a one-band raster containing the difference between the highest and second highest class probabilites\n")
        self.text2_label.setWordWrap(True)
        self.text_label.setAlignment(QtCore.Qt.AlignJustify)
        self.text2_label.setAlignment(QtCore.Qt.AlignJustify)
        self.Ok_button = QtWidgets.QPushButton('Ok')
        
        self.setLayout(QtWidgets.QFormLayout())
        self.layout().addRow(self.text_label)
        self.layout().addRow(self.text2_label)
        self.layout().addRow(self.Ok_button)

        self.Ok_button.clicked.connect(self.close)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    #ui = Ui_MainWindow()
    #ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
