import pandas as pd

class CreateDocument:
    def __init__(self, in_file):
        self.in_file = in_file
        self.data = None
        self.documents = []

    def read_data(self):
        self.data = pd.read_csv(self.in_file)
        return self.data

    def define_difficulty(self, difficulty):
        if difficulty in ["Beginner", "Novice"]:
            return "It means this exercise is best for people who have never exercised before or have very little experience."
        elif difficulty == "Intermediate":
            return "It means this exercise is best for people who have exercised before or have 1 or 2 years of experience."
        elif difficulty in ["Advanced", "Expert"]:
            return "It means this exercise is best for people who have exercised for 3 or more years."
        elif difficulty in ["Master", "Grand Master", "Legendary"]:
            return "It means this exercise is best for people who have exercised for 5 or more years."
        return "UNKNOWN"

    def create_document(self):
        for i in range(self.data.shape[0]):
            difficulty = self.data["Difficulty Level"][i]
            suitability = self.define_difficulty(difficulty)
            template = f"""1. The name of the exercise is {self.data["Exercise"][i]}
2. Difficulty level of the exercise is {self.data["Difficulty Level"][i]}. {suitability}.
3. It targets {self.data['Body Region'][i]} region of the body.
4. This exercise targets {self.data['Target Muscle Group'][i]} muscle group.
5. This exercise requires a {self.data['Force Type'][i]} force to perform. If the force required is Other that means it is neither a push nor pull force.
6. Prime Mover Muscle for this exercise is {self.data["Prime Mover Muscle"][i]}.
7. The equipment required to perform this exercise is {self.data["Primary Equipment"][i]}. It requires {self.data["# Primary Items"][i]} items. This is the number of items/equipment required to perform the exercise.
8. It is performed while having a {self.data["Posture"][i]} posture. If any person is not allowed to take this posture then they should not perform this exercise.
9. It requires {self.data["Single or Double Arm"][i]} movements and also {self.data['Continuous or Alternating Arms'][i]} arm movements.
10. It is performed by keeping a {self.data["Grip"][i]} grip.
11. The weight or load ends up in this position - {self.data['Load Position (Ending)'][i]} at the completion of an exercise repetition. If the position is Other then don't give importance to this position.
12. The movement pattern of this exercise is {self.data['Movement Pattern'][i]}."""

            self.documents.append(template)

    def convert_document_to_df(self):
        if len(self.documents) != 1389:
            raise ValueError("The number of documents should be 1389.")
        df = pd.DataFrame(self.documents, columns=['Summary'])
        return df

    def save_pickle(self, df, filename='./data/summaries.pkl'):
        df.to_pickle(filename)

