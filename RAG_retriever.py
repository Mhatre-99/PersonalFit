import pickle as pkl
from datetime import datetime
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.openai import OpenAI
from rank_bm25 import BM25Okapi
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
import nest_asyncio
from openai import OpenAI

class RAGRetriever:
    def __init__(self, index_path, nodes_path):
        self.index_path = index_path
        self.nodes_path = nodes_path
        self.retriever = None
        self.bm25_retriever = None
        self.qf_retriever = None
        self.openai_client = OpenAI()

    def read_data(self):
        with open(self.index_path, 'rb') as f:
            self.index = pkl.load(f)
        with open(self.nodes_path, 'rb') as f:
            self.nodes = pkl.load(f)

    def faiss_retriever(self):
        self.index.storage_context.persist()
        self.retriever = self.index.as_retriever()
        self.retriever.similarity_top_k = 20
        return self.retriever

    def bm25Retriever(self):
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.nodes,
            similarity_top_k=20,
            stemmer=Stemmer.Stemmer("english"),
            language="english"
        )
        self.bm25_retriever.persist("./bm25_retriever")

        return self.bm25_retriever

    def generate_prompt(self, query, num_queries):
        prompt = f"""Generate sub queries based on the QUERY provided to you. Answer the {num_queries} following appropriate questions based on the QUERY.
                     1. What target muscle should be targeted by the user? answer from this list ['Abdominals', 'Chest', 'Adductors', 'Back', 'Shoulders', 'Biceps','Glutes', 'Hamstrings', 'Abductors', 'Calves', 'Quadriceps','Trapezius', 'Forearms', 'Triceps', 'Trapezius ']
                     2. Which muscle would be the Prime Muscle Mover? Answer from the following list: ['Rectus Abdominis', 'Obliques', 'Pectoralis Major','Adductor Magnus', 'Latissimus Dorsi', 'Posterior Deltoids','Biceps Brachii', 'Gluteus Maximus', 'Biceps Femoris','Gluteus Medius', 'Gastrocnemius', 'Quadriceps Femoris','Medial Deltoids', 'Anterior Deltoids', 'Upper Trapezius','Brachioradialis', 'Triceps Brachii ', 'Triceps Brachii','Infraspinatus']
                     3. Which body region should the exercise target?
                     4. What injuries does the user have?
                     5. Which equipments does the user has or is it bodyweight?
                     6. How experienced is the user?
                     7. Which top 5 posture is suitable for the user to exercise? Answer from the folloiwng list: ['Supine', 'Quadruped', 'Seated', 'Hanging', 'Prone','Knee Hover Quadruped', 'Side Plank', 'Kneeling', 'V Sit Seated','L Sit', 'Standing', 'Half Kneeling', 'Inverted ', 'Side Lying','Bridge', 'Walking', 'Wall Sit', 'Inverted', 'Seated Floor','Single Leg Standing', 'Split Stance Standing','Single Leg Supported', 'Tall Kneeling','Knee Over Toes Split Stance Standing', 'Half Knee Hover','Staggered Stance Standing', 'Foot Elevated Standing','Single Leg Standing Bent Knee', 'Toe Balance Standing','Bent Over Standing', 'Horse Stance Standing', 'March']
                     8. Does user has any restrictions in movement.
                     9. Which top 5 movement pattern is suitable or should be done by the user? Answer from the following list: ['Anti-Extension', 'Anti-Rotational', 'Rotational','Spinal Flexion', 'Horizontal Push', 'Hip Flexion','Lateral Flexion', 'Anti-Lateral Flexion', 'Isometric Hold','Horizontal Pull', 'Vertical Pull', 'Hip External Rotation','Hip Hinge', 'Hip Abduction', 'Hip Extension','Ankle Plantar Flexion', 'Hip Adduction', 'Shoulder Abduction','Shoulder External Rotation', 'Vertical Push', 'Loaded Carry','Scapular Elevation', 'Elbow Flexion', 'Elbow Extension','Wrist Flexion', 'Knee Dominant', 'Hip Dominant', 'Hip Dominant ','Knee Dominant ', 'Shoulder Flexion', 'Wrist Extension']

                     For example, if the QUERY is about creating a workout plan for someone with knee pain, generate sub-queries like this in the same format and only the answers:
                     - The target muscles should include the 'Glutes', 'Hamstrings','Calves', and 'Quadriceps' .
                     - The Quadriceps Femoris, Gastrocnemius, Gluteus Maximus and Adductor Magnus are often the prime movers in exercises designed to support the knee joint.
                     - The exercises should target the lower body
                     - The user has knee pain. Specific injuries could include patellofemoral pain syndrome, osteoarthritis, meniscus tears, ligament injuries, or general knee inflammation.
                     - Since nothing is mentioned we will assume the user has no equipments and would do bodyweight exercises.
                     - Since nothing is mentioned we will assume the user is a beginner.
                     - Postures suitable for knee pain: 'Supine', 'Quadruped', 'Side Plank', 'Bridge', 'Standing'.
                     - The user likely has restrictions in movements that place high stress on the knee joint. These might include high-impact activities.
                     - Suitable movement patterns for knee pain: 'Hip Hinge', 'Hip Flexion', 'Hip Extension', 'Anti-Extension', 'Loaded Carry'.
                     Use simple and easy vocabulary
                     For the following query generate {num_queries} subqueries:
                     {query}"""
        return prompt

    def query_fusion_retriever(self, query):
        prompt = self.generate_prompt(query, 10)
        self.qf_retriever = QueryFusionRetriever(
            [self.retriever, self.bm25_retriever],
            similarity_top_k=20,
            num_queries=10,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=False,
            query_gen_prompt=prompt
        )

        nest_asyncio.apply()
        return self.qf_retriever
