from sklearn.externals import joblib
import pickle


def main():
   # model = joblib.load("../models/classifier.pkl")
    model = pickled_model = pickle.load(open('../models/classifier.pkl', 'rb'))
    model.get_params()


if __name__ == '__main__':
    main()