# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_VentanaPrincipal(object):
    def setupUi(self, VentanaPrincipal):
        VentanaPrincipal.setObjectName("VentanaPrincipal")
        VentanaPrincipal.resize(1273, 862)
        font = QtGui.QFont()
        font.setPointSize(10)
        VentanaPrincipal.setFont(font)
        VentanaPrincipal.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.centralwidget = QtWidgets.QWidget(VentanaPrincipal)
        self.centralwidget.setObjectName("centralwidget")
        self.TextoModelo = QtWidgets.QLabel(self.centralwidget)
        self.TextoModelo.setGeometry(QtCore.QRect(10, 10, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.TextoModelo.setFont(font)
        self.TextoModelo.setObjectName("TextoModelo")
        self.TextoMetrica = QtWidgets.QLabel(self.centralwidget)
        self.TextoMetrica.setGeometry(QtCore.QRect(10, 230, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.TextoMetrica.setFont(font)
        self.TextoMetrica.setObjectName("TextoMetrica")
        self.groupBoxModelo = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxModelo.setGeometry(QtCore.QRect(10, 30, 181, 191))
        self.groupBoxModelo.setTitle("")
        self.groupBoxModelo.setObjectName("groupBoxModelo")
        self.radioButtonVGG16 = QtWidgets.QRadioButton(self.groupBoxModelo)
        self.radioButtonVGG16.setGeometry(QtCore.QRect(10, 130, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonVGG16.setFont(font)
        self.radioButtonVGG16.setObjectName("radioButtonVGG16")
        self.radioButtonVGG19 = QtWidgets.QRadioButton(self.groupBoxModelo)
        self.radioButtonVGG19.setGeometry(QtCore.QRect(10, 100, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonVGG19.setFont(font)
        self.radioButtonVGG19.setObjectName("radioButtonVGG19")
        self.radioButtonResNet = QtWidgets.QRadioButton(self.groupBoxModelo)
        self.radioButtonResNet.setGeometry(QtCore.QRect(10, 70, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonResNet.setFont(font)
        self.radioButtonResNet.setObjectName("radioButtonResNet")
        self.radioButtonInceptionV4 = QtWidgets.QRadioButton(self.groupBoxModelo)
        self.radioButtonInceptionV4.setGeometry(QtCore.QRect(10, 40, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonInceptionV4.setFont(font)
        self.radioButtonInceptionV4.setObjectName("radioButtonInceptionV4")
        self.radioButtonAlexNet = QtWidgets.QRadioButton(self.groupBoxModelo)
        self.radioButtonAlexNet.setGeometry(QtCore.QRect(10, 160, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonAlexNet.setFont(font)
        self.radioButtonAlexNet.setObjectName("radioButtonAlexNet")
        self.groupBoxMetrica = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxMetrica.setGeometry(QtCore.QRect(10, 240, 181, 191))
        font = QtGui.QFont()
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.groupBoxMetrica.setFont(font)
        self.groupBoxMetrica.setTitle("")
        self.groupBoxMetrica.setObjectName("groupBoxMetrica")
        self.radioButtonExactitud = QtWidgets.QRadioButton(self.groupBoxMetrica)
        self.radioButtonExactitud.setGeometry(QtCore.QRect(10, 40, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonExactitud.setFont(font)
        self.radioButtonExactitud.setObjectName("radioButtonExactitud")
        self.radioButtonAUC = QtWidgets.QRadioButton(self.groupBoxMetrica)
        self.radioButtonAUC.setGeometry(QtCore.QRect(10, 160, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonAUC.setFont(font)
        self.radioButtonAUC.setObjectName("radioButtonAUC")
        self.radioButtonPrecision = QtWidgets.QRadioButton(self.groupBoxMetrica)
        self.radioButtonPrecision.setGeometry(QtCore.QRect(10, 70, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonPrecision.setFont(font)
        self.radioButtonPrecision.setObjectName("radioButtonPrecision")
        self.radioButtonRecall = QtWidgets.QRadioButton(self.groupBoxMetrica)
        self.radioButtonRecall.setGeometry(QtCore.QRect(10, 100, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonRecall.setFont(font)
        self.radioButtonRecall.setObjectName("radioButtonRecall")
        self.radioButtonHinge = QtWidgets.QRadioButton(self.groupBoxMetrica)
        self.radioButtonHinge.setGeometry(QtCore.QRect(10, 130, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonHinge.setFont(font)
        self.radioButtonHinge.setObjectName("radioButtonHinge")
        self.groupBoxOA = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxOA.setGeometry(QtCore.QRect(10, 450, 411, 361))
        font = QtGui.QFont()
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.groupBoxOA.setFont(font)
        self.groupBoxOA.setTitle("")
        self.groupBoxOA.setObjectName("groupBoxOA")
        self.spinBoxEpocas = QtWidgets.QSpinBox(self.groupBoxOA)
        self.spinBoxEpocas.setGeometry(QtCore.QRect(120, 120, 81, 24))
        self.spinBoxEpocas.setObjectName("spinBoxEpocas")
        self.TextoEpocas = QtWidgets.QLabel(self.groupBoxOA)
        self.TextoEpocas.setGeometry(QtCore.QRect(10, 120, 59, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.TextoEpocas.setFont(font)
        self.TextoEpocas.setObjectName("TextoEpocas")
        self.TextoParoEpocas = QtWidgets.QLabel(self.groupBoxOA)
        self.TextoParoEpocas.setGeometry(QtCore.QRect(10, 80, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.TextoParoEpocas.setFont(font)
        self.TextoParoEpocas.setObjectName("TextoParoEpocas")
        self.spinBoxParoepocas = QtWidgets.QSpinBox(self.groupBoxOA)
        self.spinBoxParoepocas.setGeometry(QtCore.QRect(120, 80, 81, 24))
        self.spinBoxParoepocas.setObjectName("spinBoxParoepocas")
        self.TextoTamI = QtWidgets.QLabel(self.groupBoxOA)
        self.TextoTamI.setGeometry(QtCore.QRect(10, 40, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.TextoTamI.setFont(font)
        self.TextoTamI.setObjectName("TextoTamI")
        self.spinBoxTamI = QtWidgets.QSpinBox(self.groupBoxOA)
        self.spinBoxTamI.setGeometry(QtCore.QRect(120, 40, 51, 24))
        self.spinBoxTamI.setObjectName("spinBoxTamI")
        self.radioButtonDecrementoLR = QtWidgets.QRadioButton(self.groupBoxOA)
        self.radioButtonDecrementoLR.setGeometry(QtCore.QRect(10, 200, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonDecrementoLR.setFont(font)
        self.radioButtonDecrementoLR.setObjectName("radioButtonDecrementoLR")
        self.groupBoxDecrementoLR = QtWidgets.QGroupBox(self.groupBoxOA)
        self.groupBoxDecrementoLR.setGeometry(QtCore.QRect(10, 200, 191, 131))
        self.groupBoxDecrementoLR.setTitle("")
        self.groupBoxDecrementoLR.setObjectName("groupBoxDecrementoLR")
        self.TextoPaciencia = QtWidgets.QLabel(self.groupBoxDecrementoLR)
        self.TextoPaciencia.setGeometry(QtCore.QRect(10, 30, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.TextoPaciencia.setFont(font)
        self.TextoPaciencia.setObjectName("TextoPaciencia")
        self.spinBoxPaciencia = QtWidgets.QSpinBox(self.groupBoxDecrementoLR)
        self.spinBoxPaciencia.setGeometry(QtCore.QRect(100, 30, 81, 24))
        self.spinBoxPaciencia.setObjectName("spinBoxPaciencia")
        self.TextoFactor = QtWidgets.QLabel(self.groupBoxDecrementoLR)
        self.TextoFactor.setGeometry(QtCore.QRect(10, 60, 59, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.TextoFactor.setFont(font)
        self.TextoFactor.setObjectName("TextoFactor")
        self.spinBoxFactor = QtWidgets.QSpinBox(self.groupBoxDecrementoLR)
        self.spinBoxFactor.setGeometry(QtCore.QRect(100, 60, 81, 24))
        self.spinBoxFactor.setObjectName("spinBoxFactor")
        self.TextoValorm = QtWidgets.QLabel(self.groupBoxDecrementoLR)
        self.TextoValorm.setGeometry(QtCore.QRect(10, 90, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.TextoValorm.setFont(font)
        self.TextoValorm.setObjectName("TextoValorm")
        self.spinBoxValorm = QtWidgets.QSpinBox(self.groupBoxDecrementoLR)
        self.spinBoxValorm.setGeometry(QtCore.QRect(100, 90, 81, 24))
        self.spinBoxValorm.setObjectName("spinBoxValorm")
        self.groupBox = QtWidgets.QGroupBox(self.groupBoxOA)
        self.groupBox.setGeometry(QtCore.QRect(230, 10, 171, 341))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.comboBoxCapaL2 = QtWidgets.QComboBox(self.groupBox)
        self.comboBoxCapaL2.setGeometry(QtCore.QRect(80, 230, 79, 23))
        self.comboBoxCapaL2.setObjectName("comboBoxCapaL2")
        self.line_5 = QtWidgets.QFrame(self.groupBox)
        self.line_5.setGeometry(QtCore.QRect(0, 180, 171, 16))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.TextoEpocasCAPA2 = QtWidgets.QLabel(self.groupBox)
        self.TextoEpocasCAPA2.setGeometry(QtCore.QRect(10, 230, 59, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.TextoEpocasCAPA2.setFont(font)
        self.TextoEpocasCAPA2.setObjectName("TextoEpocasCAPA2")
        self.TextoEpocasVALOR = QtWidgets.QLabel(self.groupBox)
        self.TextoEpocasVALOR.setGeometry(QtCore.QRect(10, 270, 59, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.TextoEpocasVALOR.setFont(font)
        self.TextoEpocasVALOR.setObjectName("TextoEpocasVALOR")
        self.spinBoxVL2 = QtWidgets.QSpinBox(self.groupBox)
        self.spinBoxVL2.setGeometry(QtCore.QRect(80, 270, 81, 24))
        self.spinBoxVL2.setObjectName("spinBoxVL2")
        self.TextoEpocasCAPA = QtWidgets.QLabel(self.groupBox)
        self.TextoEpocasCAPA.setGeometry(QtCore.QRect(10, 60, 59, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.TextoEpocasCAPA.setFont(font)
        self.TextoEpocasCAPA.setObjectName("TextoEpocasCAPA")
        self.TextoEpocasVALOR1 = QtWidgets.QLabel(self.groupBox)
        self.TextoEpocasVALOR1.setGeometry(QtCore.QRect(10, 100, 59, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.TextoEpocasVALOR1.setFont(font)
        self.TextoEpocasVALOR1.setObjectName("TextoEpocasVALOR1")
        self.comboBoxCapaL1 = QtWidgets.QComboBox(self.groupBox)
        self.comboBoxCapaL1.setGeometry(QtCore.QRect(80, 60, 79, 23))
        self.comboBoxCapaL1.setObjectName("comboBoxCapaL1")
        self.spinBoxVL1 = QtWidgets.QSpinBox(self.groupBox)
        self.spinBoxVL1.setGeometry(QtCore.QRect(80, 100, 81, 24))
        self.spinBoxVL1.setObjectName("spinBoxVL1")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 200, 59, 15))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(10, 30, 59, 15))
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(40, 150, 81, 21))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 310, 81, 21))
        self.pushButton_2.setObjectName("pushButton_2")
        self.spinBoxTamI_2 = QtWidgets.QSpinBox(self.groupBoxOA)
        self.spinBoxTamI_2.setGeometry(QtCore.QRect(170, 40, 51, 24))
        self.spinBoxTamI_2.setObjectName("spinBoxTamI_2")
        self.TextoOA = QtWidgets.QLabel(self.centralwidget)
        self.TextoOA.setGeometry(QtCore.QRect(10, 440, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.TextoOA.setFont(font)
        self.TextoOA.setObjectName("TextoOA")
        self.groupBoxOptimizador = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxOptimizador.setGeometry(QtCore.QRect(210, 30, 201, 191))
        font = QtGui.QFont()
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.groupBoxOptimizador.setFont(font)
        self.groupBoxOptimizador.setTitle("")
        self.groupBoxOptimizador.setObjectName("groupBoxOptimizador")
        self.radioButtonRMSprop = QtWidgets.QRadioButton(self.groupBoxOptimizador)
        self.radioButtonRMSprop.setGeometry(QtCore.QRect(10, 40, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonRMSprop.setFont(font)
        self.radioButtonRMSprop.setObjectName("radioButtonRMSprop")
        self.radioButtonSGD = QtWidgets.QRadioButton(self.groupBoxOptimizador)
        self.radioButtonSGD.setGeometry(QtCore.QRect(10, 100, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonSGD.setFont(font)
        self.radioButtonSGD.setObjectName("radioButtonSGD")
        self.radioButtonAdam = QtWidgets.QRadioButton(self.groupBoxOptimizador)
        self.radioButtonAdam.setGeometry(QtCore.QRect(10, 70, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButtonAdam.setFont(font)
        self.radioButtonAdam.setObjectName("radioButtonAdam")
        self.label = QtWidgets.QLabel(self.groupBoxOptimizador)
        self.label.setGeometry(QtCore.QRect(10, 150, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.spinBoxLearningR = QtWidgets.QSpinBox(self.groupBoxOptimizador)
        self.spinBoxLearningR.setGeometry(QtCore.QRect(110, 150, 81, 21))
        self.spinBoxLearningR.setObjectName("spinBoxLearningR")
        self.TextoOptimizador = QtWidgets.QLabel(self.centralwidget)
        self.TextoOptimizador.setGeometry(QtCore.QRect(210, 10, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.TextoOptimizador.setFont(font)
        self.TextoOptimizador.setObjectName("TextoOptimizador")
        self.pushButtonDatosE = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonDatosE.setGeometry(QtCore.QRect(600, 10, 151, 21))
        self.pushButtonDatosE.setObjectName("pushButtonDatosE")
        self.pushButtonDValidacion = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonDValidacion.setGeometry(QtCore.QRect(600, 40, 151, 21))
        self.pushButtonDValidacion.setObjectName("pushButtonDValidacion")
        self.pushButtonLimpiar = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonLimpiar.setGeometry(QtCore.QRect(440, 20, 151, 31))
        self.pushButtonLimpiar.setObjectName("pushButtonLimpiar")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(430, 60, 841, 21))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.pushButtonEntrenar = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonEntrenar.setGeometry(QtCore.QRect(810, 10, 121, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButtonEntrenar.setFont(font)
        self.pushButtonEntrenar.setObjectName("pushButtonEntrenar")
        self.pushButtonDetener = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonDetener.setGeometry(QtCore.QRect(810, 40, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButtonDetener.setFont(font)
        self.pushButtonDetener.setObjectName("pushButtonDetener")
        self.pushButtonGuardar = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonGuardar.setGeometry(QtCore.QRect(990, 10, 91, 21))
        self.pushButtonGuardar.setObjectName("pushButtonGuardar")
        self.pushButtonCargar = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonCargar.setGeometry(QtCore.QRect(990, 40, 91, 21))
        self.pushButtonCargar.setObjectName("pushButtonCargar")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(770, 0, 16, 71))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.progressBarEnt = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBarEnt.setGeometry(QtCore.QRect(440, 160, 821, 21))
        self.progressBarEnt.setProperty("value", 24)
        self.progressBarEnt.setObjectName("progressBarEnt")
        self.labelGuardar = QtWidgets.QLabel(self.centralwidget)
        self.labelGuardar.setGeometry(QtCore.QRect(1100, 10, 151, 20))
        self.labelGuardar.setText("")
        self.labelGuardar.setObjectName("labelGuardar")
        self.labelCargar = QtWidgets.QLabel(self.centralwidget)
        self.labelCargar.setGeometry(QtCore.QRect(1100, 40, 161, 20))
        self.labelCargar.setText("")
        self.labelCargar.setObjectName("labelCargar")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(420, 0, 20, 811))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.labelPerdida = QtWidgets.QLabel(self.centralwidget)
        self.labelPerdida.setGeometry(QtCore.QRect(570, 90, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelPerdida.setFont(font)
        self.labelPerdida.setObjectName("labelPerdida")
        self.labelPerdidaV = QtWidgets.QLabel(self.centralwidget)
        self.labelPerdidaV.setGeometry(QtCore.QRect(570, 120, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelPerdidaV.setFont(font)
        self.labelPerdidaV.setText("")
        self.labelPerdidaV.setObjectName("labelPerdidaV")
        self.labelExactitud = QtWidgets.QLabel(self.centralwidget)
        self.labelExactitud.setGeometry(QtCore.QRect(690, 90, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelExactitud.setFont(font)
        self.labelExactitud.setObjectName("labelExactitud")
        self.labelExactitudV = QtWidgets.QLabel(self.centralwidget)
        self.labelExactitudV.setGeometry(QtCore.QRect(690, 120, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelExactitudV.setFont(font)
        self.labelExactitudV.setText("")
        self.labelExactitudV.setObjectName("labelExactitudV")
        self.labelRecallV = QtWidgets.QLabel(self.centralwidget)
        self.labelRecallV.setGeometry(QtCore.QRect(960, 120, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelRecallV.setFont(font)
        self.labelRecallV.setText("")
        self.labelRecallV.setObjectName("labelRecallV")
        self.labelRecall = QtWidgets.QLabel(self.centralwidget)
        self.labelRecall.setGeometry(QtCore.QRect(960, 90, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelRecall.setFont(font)
        self.labelRecall.setObjectName("labelRecall")
        self.labelPrecisionV = QtWidgets.QLabel(self.centralwidget)
        self.labelPrecisionV.setGeometry(QtCore.QRect(830, 120, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelPrecisionV.setFont(font)
        self.labelPrecisionV.setText("")
        self.labelPrecisionV.setObjectName("labelPrecisionV")
        self.labelPresicion = QtWidgets.QLabel(self.centralwidget)
        self.labelPresicion.setGeometry(QtCore.QRect(830, 90, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelPresicion.setFont(font)
        self.labelPresicion.setObjectName("labelPresicion")
        self.labelPrecisionV_2 = QtWidgets.QLabel(self.centralwidget)
        self.labelPrecisionV_2.setGeometry(QtCore.QRect(1190, 120, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelPrecisionV_2.setFont(font)
        self.labelPrecisionV_2.setText("")
        self.labelPrecisionV_2.setObjectName("labelPrecisionV_2")
        self.labelAUC = QtWidgets.QLabel(self.centralwidget)
        self.labelAUC.setGeometry(QtCore.QRect(1190, 90, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelAUC.setFont(font)
        self.labelAUC.setObjectName("labelAUC")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(950, 0, 16, 71))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.TextoFuncionP = QtWidgets.QLabel(self.centralwidget)
        self.TextoFuncionP.setGeometry(QtCore.QRect(210, 230, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.TextoFuncionP.setFont(font)
        self.TextoFuncionP.setObjectName("TextoFuncionP")
        self.groupFuncionP = QtWidgets.QGroupBox(self.centralwidget)
        self.groupFuncionP.setGeometry(QtCore.QRect(210, 240, 201, 191))
        self.groupFuncionP.setTitle("")
        self.groupFuncionP.setObjectName("groupFuncionP")
        self.radioButton = QtWidgets.QRadioButton(self.groupFuncionP)
        self.radioButton.setGeometry(QtCore.QRect(10, 70, 171, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButton.setFont(font)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupFuncionP)
        self.radioButton_2.setGeometry(QtCore.QRect(10, 100, 171, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupFuncionP)
        self.radioButton_3.setGeometry(QtCore.QRect(10, 130, 171, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.groupFuncionP)
        self.radioButton_4.setGeometry(QtCore.QRect(10, 40, 181, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.radioButton_4.setFont(font)
        self.radioButton_4.setObjectName("radioButton_4")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(450, 240, 381, 331))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(880, 240, 381, 331))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.labelAUC_2 = QtWidgets.QLabel(self.centralwidget)
        self.labelAUC_2.setGeometry(QtCore.QRect(1080, 90, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelAUC_2.setFont(font)
        self.labelAUC_2.setObjectName("labelAUC_2")
        self.labelPrecisionV_3 = QtWidgets.QLabel(self.centralwidget)
        self.labelPrecisionV_3.setGeometry(QtCore.QRect(1080, 120, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelPrecisionV_3.setFont(font)
        self.labelPrecisionV_3.setText("")
        self.labelPrecisionV_3.setObjectName("labelPrecisionV_3")
        self.labelPerdidaV_2 = QtWidgets.QLabel(self.centralwidget)
        self.labelPerdidaV_2.setGeometry(QtCore.QRect(450, 120, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelPerdidaV_2.setFont(font)
        self.labelPerdidaV_2.setText("")
        self.labelPerdidaV_2.setObjectName("labelPerdidaV_2")
        self.labelPerdida_2 = QtWidgets.QLabel(self.centralwidget)
        self.labelPerdida_2.setGeometry(QtCore.QRect(450, 90, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.labelPerdida_2.setFont(font)
        self.labelPerdida_2.setObjectName("labelPerdida_2")
        VentanaPrincipal.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(VentanaPrincipal)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1273, 22))
        self.menubar.setObjectName("menubar")
        self.menuArchivos = QtWidgets.QMenu(self.menubar)
        self.menuArchivos.setObjectName("menuArchivos")
        self.menuInfo = QtWidgets.QMenu(self.menubar)
        self.menuInfo.setObjectName("menuInfo")
        self.menuAvanzado = QtWidgets.QMenu(self.menubar)
        self.menuAvanzado.setObjectName("menuAvanzado")
        VentanaPrincipal.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(VentanaPrincipal)
        self.statusbar.setObjectName("statusbar")
        VentanaPrincipal.setStatusBar(self.statusbar)
        self.actionSalir = QtWidgets.QAction(VentanaPrincipal)
        self.actionSalir.setObjectName("actionSalir")
        self.actionRegularizadores = QtWidgets.QAction(VentanaPrincipal)
        self.actionRegularizadores.setObjectName("actionRegularizadores")
        self.actionNormalizaci_n = QtWidgets.QAction(VentanaPrincipal)
        self.actionNormalizaci_n.setObjectName("actionNormalizaci_n")
        self.menuArchivos.addAction(self.actionSalir)
        self.menuAvanzado.addAction(self.actionRegularizadores)
        self.menubar.addAction(self.menuArchivos.menuAction())
        self.menubar.addAction(self.menuAvanzado.menuAction())
        self.menubar.addAction(self.menuInfo.menuAction())

        self.retranslateUi(VentanaPrincipal)
        QtCore.QMetaObject.connectSlotsByName(VentanaPrincipal)

    def retranslateUi(self, VentanaPrincipal):
        _translate = QtCore.QCoreApplication.translate
        VentanaPrincipal.setWindowTitle(_translate("VentanaPrincipal", "Proyecto Valquiria"))
        self.TextoModelo.setText(_translate("VentanaPrincipal", "Modelo"))
        self.TextoMetrica.setText(_translate("VentanaPrincipal", "M??tricas"))
        self.radioButtonVGG16.setText(_translate("VentanaPrincipal", "VGG-16"))
        self.radioButtonVGG19.setText(_translate("VentanaPrincipal", "VGG-19"))
        self.radioButtonResNet.setText(_translate("VentanaPrincipal", "ResNet-50"))
        self.radioButtonInceptionV4.setText(_translate("VentanaPrincipal", "Inception-V4"))
        self.radioButtonAlexNet.setText(_translate("VentanaPrincipal", "AlexNet"))
        self.radioButtonExactitud.setText(_translate("VentanaPrincipal", "Exactitud"))
        self.radioButtonAUC.setText(_translate("VentanaPrincipal", "AUC"))
        self.radioButtonPrecision.setText(_translate("VentanaPrincipal", "Precisi??n"))
        self.radioButtonRecall.setText(_translate("VentanaPrincipal", "Recall"))
        self.radioButtonHinge.setText(_translate("VentanaPrincipal", "Hinge"))
        self.TextoEpocas.setText(_translate("VentanaPrincipal", "Epocas"))
        self.TextoParoEpocas.setText(_translate("VentanaPrincipal", "Paro epocas"))
        self.TextoTamI.setText(_translate("VentanaPrincipal", "Tama??o Img"))
        self.radioButtonDecrementoLR.setText(_translate("VentanaPrincipal", "Decremento LR"))
        self.TextoPaciencia.setText(_translate("VentanaPrincipal", "Paciencia"))
        self.TextoFactor.setText(_translate("VentanaPrincipal", "Factor"))
        self.TextoValorm.setText(_translate("VentanaPrincipal", "Valor min"))
        self.TextoEpocasCAPA2.setText(_translate("VentanaPrincipal", "Capa"))
        self.TextoEpocasVALOR.setText(_translate("VentanaPrincipal", "Valor"))
        self.TextoEpocasCAPA.setText(_translate("VentanaPrincipal", "Capa"))
        self.TextoEpocasVALOR1.setText(_translate("VentanaPrincipal", "Valor"))
        self.label_2.setText(_translate("VentanaPrincipal", "L2"))
        self.label_3.setText(_translate("VentanaPrincipal", "L1"))
        self.pushButton.setText(_translate("VentanaPrincipal", "Agregar"))
        self.pushButton_2.setText(_translate("VentanaPrincipal", "Agregar"))
        self.TextoOA.setText(_translate("VentanaPrincipal", "Opciones adicionales"))
        self.radioButtonRMSprop.setText(_translate("VentanaPrincipal", "RMSprop"))
        self.radioButtonSGD.setText(_translate("VentanaPrincipal", "SGD"))
        self.radioButtonAdam.setText(_translate("VentanaPrincipal", "Adam"))
        self.label.setText(_translate("VentanaPrincipal", "Learning R:"))
        self.TextoOptimizador.setText(_translate("VentanaPrincipal", "Optimizador"))
        self.pushButtonDatosE.setText(_translate("VentanaPrincipal", "Datos para entrenar"))
        self.pushButtonDValidacion.setText(_translate("VentanaPrincipal", "Datos para validaci??n"))
        self.pushButtonLimpiar.setText(_translate("VentanaPrincipal", "Limpiar"))
        self.pushButtonEntrenar.setText(_translate("VentanaPrincipal", "Entrenar"))
        self.pushButtonDetener.setText(_translate("VentanaPrincipal", "Detener"))
        self.pushButtonGuardar.setText(_translate("VentanaPrincipal", "Guardar"))
        self.pushButtonCargar.setText(_translate("VentanaPrincipal", "Cargar"))
        self.labelPerdida.setText(_translate("VentanaPrincipal", "Perdida"))
        self.labelExactitud.setText(_translate("VentanaPrincipal", "Exactitud"))
        self.labelRecall.setText(_translate("VentanaPrincipal", "Recall"))
        self.labelPresicion.setText(_translate("VentanaPrincipal", "Precisi??n"))
        self.labelAUC.setText(_translate("VentanaPrincipal", "AUC"))
        self.TextoFuncionP.setText(_translate("VentanaPrincipal", "Funci??n perdida"))
        self.radioButton.setText(_translate("VentanaPrincipal", "Sparse catagorical"))
        self.radioButton_2.setText(_translate("VentanaPrincipal", "Categorical"))
        self.radioButton_3.setText(_translate("VentanaPrincipal", "Binary"))
        self.radioButton_4.setText(_translate("VentanaPrincipal", "Mean squared error"))
        self.labelAUC_2.setText(_translate("VentanaPrincipal", "Hinge"))
        self.labelPerdida_2.setText(_translate("VentanaPrincipal", "Epoca"))
        self.menuArchivos.setTitle(_translate("VentanaPrincipal", "Archivos"))
        self.menuInfo.setTitle(_translate("VentanaPrincipal", "Info"))
        self.menuAvanzado.setTitle(_translate("VentanaPrincipal", "Avanzado"))
        self.actionSalir.setText(_translate("VentanaPrincipal", "Salir"))
        self.actionRegularizadores.setText(_translate("VentanaPrincipal", "Regularizadores"))
        self.actionNormalizaci_n.setText(_translate("VentanaPrincipal", "Normalizaci??n"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    VentanaPrincipal = QtWidgets.QMainWindow()
    ui = Ui_VentanaPrincipal()
    ui.setupUi(VentanaPrincipal)
    VentanaPrincipal.show()
    sys.exit(app.exec_())

