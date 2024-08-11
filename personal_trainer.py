import pandas as pd
from datetime import datetime
from data_preprocessing import PreprocessData
from document_creator import CreateDocument
from RAG_retriever import RAGRetriever
import os
import pickle as pkl
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class PersonalTrainer:
    def __init__(self, retriever, qf_retriever, index_path, data_path, rag_data_path):
        self.retriever = retriever
        self.index_path = index_path
        self.data_path = data_path
        self.history = []
        self.qf_retriever = qf_retriever
        self.rag_data_path = rag_data_path
        self.client = OpenAI()


    def retrieve_exercises(self, query, retriever, show_exercise=False):
        with open(self.index_path, 'rb') as f:
            index = pkl.load(f)
        with open(self.rag_data_path, 'rb') as f:
            rag_data = pkl.load(f)
        data = pd.read_csv(self.data_path)

        if retriever == "qf":
            retriever = self.qf_retriever
            nodes_with_scores = retriever.retrieve(query)
            q = retriever._get_queries(query)
            additional_info = [k.query_str for k in q]
            print(additional_info)
        else:
            retriever = self.retriever
            nodes_with_scores = retriever.retrieve(query)
            additional_info = []
            print(additional_info)

        node_id = [node.node_id for node in nodes_with_scores]
        exercise_info = []
        exercise = []
        print("Retrieving exerises...")
        for i in node_id:
            ex = index.docstore.get_document(i).metadata["exercise"]
            idx = data[data['Exercise'] == ex].index.tolist()
            exercise.append(data["Exercise"][idx[0]])
            e_info = rag_data["Summary"][idx[0]]
            exercise_info.append(ex+"\n"+e_info)
            if show_exercise:
                print(ex)
        return exercise, exercise_info, additional_info

    def history_generation(self, query):
        today = datetime.today().date()
        history_prompt = f"""You are a physiotherapist/personal trainer and as every professional you keep notes of your clients progress and history.
        Given the user QUERY extract information regarding the user that you think is important and can help you in the future to decide best workout routines for them.

        For example for the following QUERY:
        I am a dancer and I recently broke my wrist. I want to improve my wrist strength now and increase its mobility.

        You could extract information like:
        - Noted on date January 10th 2024.
        - Intermediate level of experience exercising as she is a dancer. <write experience level from beginner to experienced>
        - Had a wrist injury.
        - Performed exercises to improve wrist strength and mobility.
        - Future exercise should not put a lot of pressure on the wrist.
        Give only these points as the output and nothing else. If there is no information that you think is useful give <nothing usefull> as the answer.
        For the following query extract such points in the same format:
        {query}
        """

        response = self.client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": "You are a professional physiotherapist."},
        {"role":"assistant", "content": history_prompt}
      ],
      temperature = 0.2
    )

        return response.choices[0].message.content

    def recommend_with_history(self, query, exercise):
        today = datetime.today().date()

        generationPrompt = f"""Given the user QUERY, user HISTORY and EXERCISE CONTEXT retrieved from the RAG system recommend the user at most 5 exercises that they can perform to meet their needs.
        Make sure that you consider the history of the user before making suggestions. Be mindful of the date the history was noted.
        Also use every information you have on the user to recommend them exercises from the EXERCISE CONTEXT. Todays date is {today}. Check the history date and make recommendations.

        USER QUERY:
        {query}

        USER HISTORY:
        {self.history}

        EXERCISE CONTEXT:
        {exercise}
        Make sure you consider the experience level of the user and the equipments they have. Also pay attention to any injuries or issues they are facing.
        Keep your answer concise and recommend only from the EXERCISE CONTEXT provided.
        Provide answer in the following format:
        <sr no> <Exercise Name>
        Difficulty level: <Difficulty level>
        Equipment: <Equipment Required>
        Target Muscle: <Target Muscle Group>
        Movement: <Movement>"""
        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional physiotherapist."},
            {"role":"assistant", "content": generationPrompt}
        ],
        temperature = 0.2
        )
        return response.choices[0].message.content

    def recommend_without_history(self, query, exercises):
        generation_prompt = f"""Given the user QUERY and INFORMATION about the exercises retrieved from RAG system. your task is to understand the exercises and provide top 5 exercises that you feel would be the best suited to the user according to their QUERY.
        Only provide exercises which are suitable for the user. Try using additional information such as injuries or issues the user is facing to make recommendations.
        The user QUERY is as follows:
        {query}

        The List of exercises are as follows:
        {exercises}
        Make sure you consider the experience level of the user and the equipments they have.
        Keep your answer concise and recommend only from the EXERCISE CONTEXT provided.
        Provide answer in the following format:
        <sr no> <Exercise Name>
        Difficulty level: <Difficulty level>
        Equipment: <Equipment Required>
        Target Muscle: <Target Muscle Group>
        Movement: <Movement>
        Reason: <Why you chose this exercise in one or two lines>"""
        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional physiotherapist."},
            {"role": "user", "content": generation_prompt},
        ],
        temperature = 0.2
        )
        return response.choices[0].message.content

    def run_trainer(self, query, retriever="qf", new_user=False):
        if new_user:
            self.history = []
        
        e, exercise, additional_info = self.retrieve_exercises(query + " \n\n" + str(self.history), retriever=retriever)
        query = query + " " + " ".join(additional_info)

        res_without_hist = self.recommend_without_history(query, exercise)
        print("\n\n")
        print("THE FOLLOWING ARE RECOMMENDATIONS WITHOUT HISTORY:\n\n")
        print(res_without_hist)
        print("\n\n")
        return e, self.history

    def recommendations(self, query):
        print("\n\n The following is result from FIASS")
        e_faiss, self.history = self.run_trainer(query, self.retriever, new_user = True)
        print("\n\n The following is result from QFR")
        e_multi ,self. history = self.run_trainer(query, retriever = "qf", new_user = True)
        df = pd.DataFrame({"faiss":e_faiss,"multi":e_multi})
        print(df)

if __name__ == "__main__":
    query = str(input("Enter your query: "))

    rag_retriever = RAGRetriever("./data/index_colab.pkl", "./data/nodes_colab.pkl")
    rag_retriever.read_data()
    retriever = rag_retriever.faiss_retriever()
    bm25_retriever = rag_retriever.bm25Retriever()
    qf_retriever = rag_retriever.query_fusion_retriever(query)
    personal_trainer = PersonalTrainer(retriever, qf_retriever, "./data/index_colab.pkl", "./data/cleaned_ex_data.csv" , "./data/summaries.pkl")
    personal_trainer.recommendations(query)
    
