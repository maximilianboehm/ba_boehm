from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import os


def eval_with_linear_reg(X_train, X_test, y_train, y_test, file_name, times_to_aug, additional_info):

    # scale input data
    trans_in = MinMaxScaler()
    trans_in.fit(X_train)

    # scale output data
    trans_out = MinMaxScaler()
    trans_out.fit(y_train)

    # create and train linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    predictions = lin_reg.predict(X_test)

    print('R2: ', r2_score(y_test, predictions))
    print('MAE: ', mean_absolute_error(y_test, predictions))

    # write evalutation results to desired files
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    with open(file_name, 'a+') as f:
        line1 = '\n'
        line2 = ' '
        line3 = '\n'
        line4 = additional_info
        line5 = '\n'
        line6 = 'TIMES_TO_AUGMENT='+str(times_to_aug)
        line7 = '\n'
        line8 = 'TSTR:'
        line9 = '\n'
        line10 = '   Rsquared: ' + str(r2_score(y_test, predictions))
        line11 = '\n'
        line12 = '   MAE: ' + str(mean_absolute_error(y_test, predictions))
        f.writelines([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12])

