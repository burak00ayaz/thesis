from transformers import AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
sentence = "Paris is the capital of France."

def tokenize(sentence: str):
    return tokenizer.tokenize(sentence)


# 1. Get token strings
# tokens = tokenizer.tokenize(sentence)
# print(tokens)

# 2. Get token IDs
# token_ids = tokenizer.encode(sentence)
# print(token_ids)

# 3. Recommended: get model-ready inputs
# inputs = tokenizer(sentence, return_tensors="pt")
# print(inputs)



context_1 = """
Mistral-7B-Instruct-v0.2 is an instruction-tuned version of Mistral-7B-v0.2.
Compared to v0.1, Mistral-7B-v0.2 has a 32k context window.
"""
question_1 = "What is the context window of Mistral-7B-Instruct-v0.2?"
expected_answer_1 = "32k."

context_2 = """
Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary.
Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend 'Venite Ad Me Omnes'.
Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.
It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Ahmet Burak Ayaz in 1858.
At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.
"""
question_2 = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
expected_answer_2 = "Ahmet Burak Ayaz."

context_3 = "The Arven Research Lab opened its new robotics wing in Zurich in March 2024. The wing contains three testing rooms, each designed for a different robot type: indoor drones, warehouse robots, and medical assistance robots. The indoor drone room is called FlightCell-7 and includes motion-capture cameras mounted on the ceiling. The warehouse robot room is called GridBay-2 and uses movable shelves. The medical assistance robot room is called CareSim-4 and contains hospital beds and patient mannequins."
question_3 = "Which testing room is used for indoor drones?"
expected_answer_3 = "FlightCell-7."

context_4 = "MiraTech released two versions of its lightweight language model in 2025. The first version, MiraLite-1.3B, was designed for mobile devices and required only 2.8 GB of memory during inference. The second version, MiraLite-3.2B, was intended for edge servers and required 6.5 GB of memory. Although the larger model performed better on reasoning tasks, the smaller model was preferred for offline applications because it could run on phones without cloud access."
question_4 = "How much memory did MiraLite-1.3B require during inference?"
expected_answer_4 = "2.8 GB."

context_5 = "In the city archive of Bremen, three historical letters were digitized under the identifiers BRM-1821-A, BRM-1821-B, and BRM-1821-C. The first letter was written by merchant Anton Keller, the second by teacher Elise Brandt, and the third by shipbuilder Johann Meier. Researchers were especially interested in BRM-1821-B because it described daily school routines in early nineteenth-century Bremen. The archive made all three letters publicly available in January 2023."
question_5 = "Which identifier belongs to the letter written by Elise Brandt?"
expected_answer_5 = "BRM-1821-B."

context_6 = "A small medical study compared three sleep-tracking devices: SomnoBand X2, RestPatch Mini, and NightLoop 5. SomnoBand X2 measured heart rate and movement using a wrist sensor. RestPatch Mini was attached to the chest and recorded breathing patterns. NightLoop 5 was placed under the mattress and estimated sleep stages from vibration data. The researchers found that RestPatch Mini produced the most reliable breathing measurements, while NightLoop 5 was the easiest device for participants to use."
question_6 = "Which device was attached to the chest?"
expected_answer_6 = "RestPatch Mini."

context_7 = "The university library moved several computer science books to a new shelf system in April. Books about databases were placed on shelf CS-D42, books about computer architecture on CS-A17, and books about machine learning on CS-M88. The old shelf labels were removed after the move, which confused some students during the first week. A printed map near the entrance listed the updated shelf codes and included examples of popular books in each category."
question_7 = "Where were the computer architecture books placed?"
expected_answer_7 = "CS-A17."

context_8 = "During the Mars simulation mission, the crew tested three emergency protocols. Protocol Red-14 covered oxygen leaks, Protocol Blue-09 covered communication loss, and Protocol Green-31 covered power failure. On the fifth day, the team simulated a communication blackout lasting forty minutes. The crew successfully followed Protocol Blue-09 by switching to local decision-making and logging all actions manually. The mission supervisor later praised the team for reacting calmly under pressure."
question_8 = "Which protocol covered communication loss?"
expected_answer_8 = "Protocol Blue-09."

context_9 = "A bakery in Munich introduced three seasonal breads for autumn. Walnut rye was sold only on Mondays and Wednesdays. Pumpkin seed sourdough was available every Friday. Apple cinnamon brioche was baked on Saturdays and usually sold out before noon. The owner said pumpkin seed sourdough had the highest profit margin because it used inexpensive ingredients and sold at a premium price. Customers, however, most often asked for the apple cinnamon brioche."
question_9 = "Which bread had the highest profit margin?"
expected_answer_9 = "Pumpkin seed sourdough."

context_10 = "The software team used three internal tools during the migration project. LogTrace collected runtime errors from production services. SchemaGuard checked whether database changes were backward compatible. DeployPilot managed staged rollouts across test, staging, and production environments. The team relied most heavily on SchemaGuard because the migration involved many table modifications. DeployPilot was only used after all schema checks passed successfully."
question_10 = "What was SchemaGuard used for?"
expected_answer_10 = "It checked whether database changes were backward compatible."

context_11 = "At the annual student engineering fair, four teams presented embedded systems projects. Team Nova built a smart greenhouse controller. Team Orbit designed a low-power satellite beacon. Team Flux created a wearable posture sensor. Team Atlas demonstrated an autonomous model car. The judges awarded first prize to Team Orbit because their beacon operated for six months on a small battery. Team Flux received the audience award for its live demonstration."
question_11 = "Which team won first prize?"
expected_answer_11 = "Team Orbit."

context_12 = "A logistics company tested three delivery routes for packages leaving its Leipzig warehouse. Route L-18 passed through Halle and was the shortest in distance. Route L-27 passed through Jena and had the lowest fuel consumption. Route L-35 passed through Erfurt and was the fastest during morning traffic. After two weeks of testing, the company selected Route L-27 for regular operation because fuel cost was more important than delivery speed."
question_12 = "Why did the company select Route L-27?"
expected_answer_12 = "Because it had the lowest fuel consumption, and fuel cost was more important than delivery speed."



print('Number of tokens in context_1:', len(tokenize(context_1)))
print('Number of tokens in context_2:', len(tokenize(context_2)))
print('Number of tokens in context_3:', len(tokenize(context_3)))
print('Number of tokens in context_4:', len(tokenize(context_4)))
print('Number of tokens in context_5:', len(tokenize(context_5)))
print('Number of tokens in context_6:', len(tokenize(context_6)))
print('Number of tokens in context_7:', len(tokenize(context_7)))
print('Number of tokens in context_8:', len(tokenize(context_8)))
print('Number of tokens in context_9:', len(tokenize(context_9)))
print('Number of tokens in context_10:', len(tokenize(context_10)))
print('Number of tokens in context_11:', len(tokenize(context_11)))
print('Number of tokens in context_12:', len(tokenize(context_12)))


print('Tokens in context_1:', tokenize(context_1))
print('Tokens in context_2:', tokenize(context_2))
print('Tokens in context_3:', tokenize(context_3))
print('Tokens in context_4:', tokenize(context_4))
print('Tokens in context_5:', tokenize(context_5))
print('Tokens in context_6:', tokenize(context_6))
print('Tokens in context_7:', tokenize(context_7))
print('Tokens in context_8:', tokenize(context_8))
print('Tokens in context_9:', tokenize(context_9))
print('Tokens in context_10:', tokenize(context_10))
print('Tokens in context_11:', tokenize(context_11))
print('Tokens in context_12:', tokenize(context_12))