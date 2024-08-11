import pandas as pd

class PreprocessData:
    def __init__(self, in_file, out_file):
        self.in_file = in_file
        self.out_file = out_file

    def read_data(self):
        self.data = pd.read_csv(self.in_file)
        return self.data

    def clean_data(self):
        self.data.drop(columns= ["Short YouTube ", "Depth", "Unnamed: 31"], inplace=True)
        
        for i in self.data.columns:
            d = self.data[i].isnull().sum()
            if d >= self.data.shape[0] * 0.30:
                self.data.drop(columns=[i], inplace=True)
        self.data.dropna(inplace=True)
        self.data.drop(columns=["Laterality", "Mechanics", "Plane Of Motion #1"], inplace=True)
        #self.data.drop(columns=["Unnamed: 0"], inplace=True)
        self.data.rename(columns=lambda x: x.strip(), inplace=True)
        self.data.rename(columns={'Movement Pattern #1': 'Movement Pattern'}, inplace=True, errors='raise')
        print(self.data.columns)

    def save_cleaned_data(self):
        self.data.to_csv(self.out_file, index=False)

