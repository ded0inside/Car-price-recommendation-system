import sys
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt6 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.use('QtAgg')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

global age_rel
global fuel_relation
global seats_relation
global df_num
global df_nor
global seats_count
global rentdata
global regr


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # This subplot divides the canvas in 10 parts
        # and I only use 9 of them for the plot, leaving some space to
        # clearly see the axes.
        # Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html
        self.axes = fig.add_subplot(10, 1, (1, 9))
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        font = QtGui.QFont("Sanserif", 11)
        QtWidgets.QApplication.setFont(QtGui.QFont(font))

        # we don't start analysis before doing a data-import
        self.data_import = False
        self.plot_canvas = MplCanvas(self, width=10, height=4, dpi=100)
        self.toolbar = NavigationToolbar(self.plot_canvas, self)
        self.plot_canvas.axes.text(0.03, 0.5, "Please click on \'Start Analysis\'\n"
                                              "to view data analysis and predictions\n"
                                              "\n"
                                              "Default values are visible under \'Current Inputs\'\n"
                                              "You can change them anytime after clicking \'Start Analysis\'")

        canvas_layout = QtWidgets.QGridLayout()
        canvas_layout.addWidget(self.plot_canvas, 1, 1)

        trend_button_layout = QtWidgets.QGridLayout()

        self.trends_explainer = QtWidgets.QLabel()
        self.trends_explainer.setText("Click on any of the buttons below to\n"
                                      "see the trends in data\n")

        # buttons for viewing trends in data (EDA)
        age_price_trend = QtWidgets.QPushButton()
        age_price_trend.setText("Car Age Vs Price")

        fuel_price_trend = QtWidgets.QPushButton()
        fuel_price_trend.setText("Fuel Type Vs Mileage")

        seat_num_trend = QtWidgets.QPushButton()
        seat_num_trend.setText("Number Of Seats Vs Price")

        seats_distribution = QtWidgets.QPushButton()
        seats_distribution.setText("Seats number distribution")

        transmission_distribution = QtWidgets.QPushButton()
        transmission_distribution.setText("Transmission distribution")

        owner_trend = QtWidgets.QPushButton()
        owner_trend.setText("Owner Type vs Price")

        back_to_prediction = QtWidgets.QPushButton()
        back_to_prediction.setText("Back To Analysis")

        age_price_trend.setMinimumSize(50, 25)
        fuel_price_trend.setMinimumSize(50, 25)
        seat_num_trend.setMinimumSize(50, 25)
        seats_distribution.setMinimumSize(50, 25)
        transmission_distribution.setMinimumSize(50, 25)
        owner_trend.setMinimumSize(50, 25)
        back_to_prediction.setMinimumSize(50, 25)

        age_price_trend.clicked.connect(self.age_relation)
        fuel_price_trend.clicked.connect(self.fuel_type_relation)
        seat_num_trend.clicked.connect(self.seat_num_relation)
        seats_distribution.clicked.connect(self.seat_num_distribution)
        transmission_distribution.clicked.connect(self.transmission_distribution)
        owner_trend.clicked.connect(self.owner_relation)
        back_to_prediction.clicked.connect(self.show_prediction)

        trend_button_layout.addWidget(self.trends_explainer, 0, 1)
        trend_button_layout.addWidget(age_price_trend, 1, 1)
        trend_button_layout.addWidget(fuel_price_trend, 2, 1)
        trend_button_layout.addWidget(seat_num_trend, 3, 1)
        trend_button_layout.addWidget(owner_trend, 4, 1)
        trend_button_layout.addWidget(transmission_distribution, 5, 1)
        trend_button_layout.addWidget(seats_distribution, 6, 1)
        trend_button_layout.addWidget(back_to_prediction, 7, 1)
        trend_button_layout.setContentsMargins(10, 0, 10, 0)

        canvas_trends_layout = QtWidgets.QGridLayout()
        canvas_trends_layout.addLayout(canvas_layout, 1, 1)

        # customizable user input
        user_input_layout = QtWidgets.QGridLayout()

        self.tr_num = 0
        transmission_type = QtWidgets.QComboBox()
        transmission_type.addItems(['Manual', 'Automatic'])
        self.tr_list = ["Manual", 'Automatic']
        transmission_type.currentIndexChanged.connect(self.transmission_input)
        transmission_type.setMinimumSize(100, 40)

        self.ft = 0
        fuel_type = QtWidgets.QComboBox()
        fuel_type.addItems(['CNG', 'Diesel', 'Petrol', 'LPG', 'Electric'])
        self.fuel_list = ['CNG', 'Diesel', 'Petrol', 'LPG', 'Electric']
        fuel_type.currentIndexChanged.connect(self.fueltype_input)
        fuel_type.setMinimumSize(100, 40)

        self.brand_class = 0
        brand = QtWidgets.QComboBox()
        brand.addItems(['Budget Car', 'Expensive Car'])
        self.brand_list = ["Budget Car", 'Expensive Car']
        brand.currentIndexChanged.connect(self.brand_input)
        brand.setMinimumSize(100, 40)

        self.power = 40
        power = QtWidgets.QSpinBox()
        power.setRange(34, 522)
        power.setPrefix(f"Power: ")
        power.setSuffix(" bhp")
        power.setSingleStep(20)
        power.valueChanged.connect(self.power_change)
        power.setMinimumSize(100, 40)

        self.kmd = 1000
        km_driven = QtWidgets.QSpinBox()
        km_driven.setMinimum(1000)
        km_driven.setMaximum(6500000)
        km_driven.setPrefix(f"Km Driven: ")
        km_driven.setSingleStep(1000)
        km_driven.valueChanged.connect(self.km_driven)
        km_driven.setMinimumSize(50, 40)

        age_slider = QtWidgets.QSlider()
        self.ac = 2
        self.age_widget = QtWidgets.QLabel()
        self.age_widget.setText(f'Car Age: {self.ac} year(s)')
        age_slider.setMinimum(2)
        age_slider.setMaximum(23)
        age_slider.setSingleStep(1)
        age_slider.valueChanged.connect(self.change_age)
        age_slider.setOrientation(QtCore.Qt.Orientation.Vertical)

        self.seat_no = 2
        self.seat_slider_display = QtWidgets.QLabel()
        self.seat_slider_display.setText(f"Number of Seats: {self.seat_no}")
        seat_slider = QtWidgets.QSlider()
        seat_slider.setRange(2, 10)
        seat_slider.setSingleStep(1)
        seat_slider.valueChanged.connect(self.seat_change)

        self.engine = 100
        engine_input = QtWidgets.QSpinBox()
        engine_input.setPrefix('Engine Power: ')
        engine_input.setSuffix(' cc')
        engine_input.setRange(100, 6000)
        engine_input.setSingleStep(100)
        engine_input.valueChanged.connect(self.engine_change)
        engine_input.setMinimumSize(100, 40)

        user_input_vertical = QtWidgets.QVBoxLayout()
        user_input_vertical.addWidget(transmission_type)
        user_input_vertical.addWidget(km_driven)
        user_input_vertical.addWidget(fuel_type)
        user_input_vertical.addWidget(power)
        user_input_vertical.addWidget(brand)
        user_input_vertical.addWidget(engine_input)

        # creating separate layout for sliders, for better positioning
        user_age_layout = QtWidgets.QVBoxLayout()
        user_age_layout.addWidget(self.age_widget)
        user_age_layout.addWidget(age_slider)

        user_dial_layout = QtWidgets.QVBoxLayout()
        user_dial_layout.addWidget(self.seat_slider_display)
        user_dial_layout.addWidget(seat_slider)

        user_input_layout.addLayout(user_input_vertical, 0, 0)
        user_input_layout.addLayout(user_age_layout, 0, 2)
        user_input_layout.addLayout(user_dial_layout, 0, 1)

        # output widgets
        self.predicted_price = 0
        self.predicted_price_display = QtWidgets.QLabel()
        self.predicted_price_display.setText(
            f"Current Inputs:\n"
            f"Transmission: {self.tr_list[self.tr_num]}\n"
            f"Km Driven: {self.kmd}\n"
            f"Fuel-Type: {self.fuel_list[self.ft]}\n"
            f"Power: {self.power} bhp\n"
            f"Brand Class: {self.brand_list[self.brand_class]}\n"
            f"Number Of Seats: {self.seat_no}\n"
            f"Engine: {self.engine} cc\n"
            f"Car Age: {self.ac}\n"
            f"Predicted price is: ₹{self.predicted_price * 100000}!")
        self.predicted_price_display.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # application buttons
        self.start_analysis_button = QtWidgets.QPushButton()
        self.start_analysis_button.setText("Start Analysis")
        self.start_analysis_button.clicked.connect(self.show_prediction)
        self.start_analysis_button.setMinimumSize(100, 40)

        # we want to show a prediction everytime the input changes
        self.input_changed = False

        self.saved_plot = False
        save_button = QtWidgets.QPushButton()
        save_button.setMinimumSize(100, 40)
        save_button.clicked.connect(self.save_plot)
        save_button.setText("Save Results")

        quit_button = QtWidgets.QPushButton()
        quit_button.setMinimumSize(100, 40)
        quit_button.setText("Quit")
        quit_button.clicked.connect(self.closeEvent)

        app_buttons_layout = QtWidgets.QVBoxLayout()
        app_buttons_layout.addWidget(self.predicted_price_display)
        app_buttons_layout.addWidget(self.start_analysis_button)
        app_buttons_layout.addWidget(save_button)
        app_buttons_layout.addWidget(quit_button)

        # final layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.toolbar, 0, 1)
        layout.addLayout(canvas_trends_layout, 1, 1)
        layout.addLayout(trend_button_layout, 1, 2)
        layout.addLayout(user_input_layout, 2, 1)
        layout.addLayout(app_buttons_layout, 2, 2)
        layout.setContentsMargins(20, 0, 20, 20)

        # setting the main layout
        main_wid = QtWidgets.QWidget()
        main_wid.setLayout(layout)

        self.setWindowTitle("Used Car Selling Price Predictor")
        self.setMinimumSize(1000, 700)
        self.setCentralWidget(main_wid)
        self.center()
        self.show()

    def center(self):
        # centering the layout of the application.
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def update_predict_label(self):
        self.predicted_price_display.setText(
            f"Current Inputs:\n"
            f"Transmission: {self.tr_list[self.tr_num]}\n"
            f"Km Driven: {self.kmd}\n"
            f"Fuel-Type: {self.fuel_list[self.ft]}\n"
            f"Power: {self.power} bhp\n"
            f"Brand Class: {self.brand_list[self.brand_class]}\n"
            f"Number Of Seats: {self.seat_no}\n"
            f"Engine: {self.engine} cc\n"
            f"Car Age: {self.ac}\n"
            f"Predicted price is: ₹{self.predicted_price}!")

    # functions that describe the trends in data:
    def age_relation(self):
        if self.data_import:
            self.trends_explainer.setText('As the age of the car increases\n'
                                          'the selling price decreases\n'
                                          )
            self.plot_canvas.axes.cla()
            self.plot_canvas.axes.plot(age_rel['Price'].mean())
            self.plot_canvas.axes.set_xlabel(' <- Car Age (Years)->')
            self.plot_canvas.axes.set_ylabel(' <- Avg Selling Price (lakhs) ->')
            self.plot_canvas.draw()
        else:
            self.trends_explainer.setText('Please Import Data Before Viewing Analysis')

    def fuel_type_relation(self):
        if self.data_import:
            self.trends_explainer.setText('CNG has the highest power\n'
                                          'Diesel and Electric are almost equal\n'
                                          'Petrol is the least\n'
                                          'NOTE: minimum power is 18km and max is 25km!'
                                          )
            self.plot_canvas.axes.cla()
            self.plot_canvas.axes.bar(['CNG', 'Diesel', 'Electric', 'LPG', 'Petrol'],
                                      fuel_relation['Mileage'].mean())
            self.plot_canvas.axes.set_xlabel(' <- Type Of Fuel->')
            self.plot_canvas.axes.set_ylabel(' <- Avg Mileage (km) ->')
            self.plot_canvas.draw()
        else:
            self.trends_explainer.setText('Please Import Data Before Viewing Analysis')

    def seat_num_relation(self):
        if self.data_import:
            self.trends_explainer.setText('Cars with 2 seats like sports cars sell for more\n'
                                          'but after that, there is a overall decrease\n'
                                          'But SUVs also sell for more than usual')

            self.plot_canvas.axes.cla()
            self.plot_canvas.axes.plot(seats_relation['Price'].mean())
            self.plot_canvas.axes.set_xlabel(' <- Number of Seats ->')
            self.plot_canvas.axes.set_ylabel(' <- Avg Selling Price (lakhs) ->')
            self.plot_canvas.draw()
        else:
            self.trends_explainer.setText('Please Import Data Before Viewing Analysis')

    def seat_num_distribution(self):
        if self.data_import:
            self.trends_explainer.setText('Cars with 2 seats are least popular\n'
                                          'Cars with 5 seats are sold the most,\n'
                                          'this also explains why their selling price is lower'
                                          )
            self.plot_canvas.axes.cla()
            self.plot_canvas.axes.bar(['2', '4', '5', '6', '7', '8', '9', '10'],
                                      seats_count)
            self.plot_canvas.axes.set_xlabel(' <- Number of Seats ->')
            self.plot_canvas.axes.set_ylabel(' <- Amount of sold cars ->')
            self.plot_canvas.draw()
        else:
            self.trends_explainer.setText('Please Import Data Before Viewing Analysis')

    def transmission_distribution(self):
        if self.data_import:
            self.trends_explainer.setText('Cars with automatic transmission are more popular'
                                          )
            self.plot_canvas.axes.cla()
            self.plot_canvas.axes.bar(df_nor['Transmission'].unique(),
                                      df_nor.groupby('Transmission')['S.No.'].count())
            self.plot_canvas.axes.set_xlabel(' <- Transmission type ->')
            self.plot_canvas.axes.set_ylabel(' <- Amount of sold cars ->')
            self.plot_canvas.draw()
        else:
            self.trends_explainer.setText('Please Import Data Before Viewing Analysis')

    def owner_relation(self):
        if self.data_import:
            self.trends_explainer.setText('As the smount of owners increases\n'
                                          'the selling price decreases\n'
                                          )
            self.plot_canvas.axes.cla()
            self.plot_canvas.axes.plot(df_nor.groupby('Owner_Type')['Price'].mean())
            self.plot_canvas.axes.set_xlabel(' <- Amount of owners ->')
            self.plot_canvas.axes.set_ylabel(' <- Avg Selling Price (lakhs) ->')
            self.plot_canvas.draw()
        else:
            self.trends_explainer.setText('Please Import Data Before Viewing Analysis')

    # functions which map the user input to the data for prediction:
    def brand_input(self, i):
        self.brand_class = i
        self.input_changed = True
        self.update_predict_label()
        # whenever data is changed, we update the graph instantly
        if self.data_import:
            self.show_prediction()

    def change_age(self, i):
        self.ac = i
        self.age_widget.setText(f'Car Age: {self.ac} year(s)')
        self.input_changed = True
        self.update_predict_label()
        if self.data_import:
            self.show_prediction()

    def engine_change(self, i):
        self.engine = i
        self.input_changed = True
        self.update_predict_label()
        if self.data_import:
            self.show_prediction()

    def km_driven(self, i):
        self.kmd = i
        self.input_changed = True
        self.update_predict_label()
        if self.data_import:
            self.show_prediction()

    def power_change(self, i):
        self.input_changed = True
        self.power = i
        self.update_predict_label()
        if self.data_import:
            self.show_prediction()

    def transmission_input(self, i):
        self.input_changed = True
        self.tr_num = i
        self.update_predict_label()
        if self.data_import:
            self.show_prediction()

    def seat_change(self, i):
        self.input_changed = True
        self.seat_no = i
        self.seat_slider_display.setText(f"Number of Seats: {self.seat_no}")
        self.update_predict_label()
        if self.data_import:
            self.show_prediction()

    def fueltype_input(self, i):
        self.input_changed = True
        self.ft = i
        self.update_predict_label()
        if self.data_import:
            self.show_prediction()

    def plot_price(self, year=None, price=None):
        self.plot_canvas.axes.cla()

        global rentdata
        df = rentdata.loc[:, ('Price_log', 'Ageofcar')]

        df['Year'] = 2023 - df['Ageofcar']
        df['Price'] = np.exp(df['Price_log'])

        df = df[['Price', 'Year']]
        df.sort_values(by=['Year'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        x = df['Year'].values
        _y = df["Price"].values
        euro_price = self.predicted_price * 1000
        euro_price = "{:.2f}".format(euro_price)
        self.predicted_price = "{:.2f}".format(self.predicted_price)
        self.plot_canvas.axes.scatter(x, _y)
        self.plot_canvas.axes.plot(year, price, marker="*", markersize=10, c='red')
        self.plot_canvas.axes.annotate(xy=(year, price),
                                       text=f"Predicted Price: ₹{self.predicted_price} lakhs\n"
                                            f"Or €{euro_price}",
                                       xytext=(1999.5, 130))
        self.plot_canvas.axes.set_ylabel("<- Price in lakhs ->")
        self.plot_canvas.axes.set_xlabel("<- Year ->")
        self.plot_canvas.draw()

    def show_prediction(self):
        self.generate_model()
        if self.input_changed:
            self.saved_plot = False
        self.trends_explainer.setText("Click on any of the buttons below to\n"
                                      "see the trends in data")
        # Brand Class
        bc = self.brand_class
        # Owner_Type_num
        ot = 1
        # Transmission_num
        tr = self.tr_num
        # Fuel_Type_num
        ft = self.ft
        # Kilometers_Driven_log
        kdl = np.log(self.kmd)
        # mileage
        m = rentdata['Mileage'].mean()
        # Seats
        s = self.seat_no
        # Ageofcar
        ac = self.ac
        # Power
        p = self.power
        # Engine
        e = self.engine

        _X_test = [[bc, ot, tr, ft, kdl, m, s, ac, p, e]]
        predicted_price = regr.predict(_X_test)[0][0]
        self.predicted_price = np.exp(predicted_price)

        self.plot_price(year=2023 - ac, price=self.predicted_price)
        self.predicted_price_display.setText(
            f"Current Inputs:\n"
            f"Transmission: {self.tr_list[self.tr_num]}\n"
            f"Km Driven: {self.kmd}\n"
            f"Fuel-Type: {self.fuel_list[self.ft]}\n"
            f"Power: {self.power} bhp\n"
            f"Brand Class: {self.brand_list[self.brand_class]}\n"
            f"Number Of Seats: {self.seat_no}\n"
            f"Engine: {self.engine} cc\n"
            f"Car Age: {self.ac}\n"
            f"Predicted price is: ₹{self.predicted_price}!")
        self.start_analysis_button.setText("Interactive Analysis Started")
        self.input_changed = False

    def save_plot(self):
        self.toolbar.save_figure(self.plot_canvas.figure)
        self.saved_plot = True
        self.input_changed = False

    def closeEvent(self, event) -> None:
        if self.saved_plot:
            self.close()
        else:
            dlg = QtWidgets.QMessageBox(self)
            dlg.setWindowTitle("Unsaved Plot Alert!")
            dlg.setText("You have not saved the plot, are you sure you want to exit?")
            dlg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No |
                                   QtWidgets.QMessageBox.StandardButton.Save)
            button = dlg.exec()

            if button == QtWidgets.QMessageBox.StandardButton.Yes:
                sys.exit()
            elif button == QtWidgets.QMessageBox.StandardButton.Save:
                self.save_plot()
            else:
                if not type(event) == bool:
                    event.ignore()

    def generate_model(self):
        self.data_import = True
        self.start_analysis_button.setEnabled(False)
        global df_num
        global df_nor
        global seats_count
        global rentdata
        global regr
        df_num = pd.read_csv(r"cardata_numerized.csv")
        df_nor = pd.read_csv(r"used_cars_data_cleaned.csv")

        df_num.sort_values('Ageofcar')

        global age_rel
        age_rel = df_nor.groupby('Ageofcar')

        global fuel_relation
        fuel_relation = df_nor.groupby('Fuel_Type')

        global seats_relation
        seats_relation = df_nor.groupby('Seats')
        seats_count = df_nor.groupby('Seats')['S.No.'].count()

        rentdata = pd.read_csv(r"cardata_numerized.csv")

        X = rentdata.iloc[:, 1:-1]
        y = rentdata[['Price_log']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        regr = LinearRegression()

        regr.fit(X_train, y_train)

        # Make predictions using the testing set
        Y_pred = regr.predict(X_test)
        print("Coefficients: \n", regr.coef_)

        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(y_test, Y_pred))

        # The coefficient of determination: 1 is perfect prediction
        print("Coefficient of determination: %.2f" % r2_score(y_test, Y_pred))


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec()
