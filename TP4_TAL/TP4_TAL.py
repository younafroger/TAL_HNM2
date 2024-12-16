import spacy
from spacy.training.example import Example
from spacy.lookups import Lookups
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


def process(txt, liste):
    tokens = txt.split("\n")
    start = 0
    entities = list()
    phrase = list()
    for t in tokens:
        if len(t) > 0 and t[0] != "#":
            word, _, _, tag = t.split(" ")
            end = len(word) + start
            entities.append((start, end, tag))
            phrase.append(word)
            start = end + 1
    liste.append((" ".join(phrase), {"entities": entities}))
    return liste

def load_conll(file_path, max_samples=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        txt = file.read()
        phrases = txt.split("\n\n")
        for i, phrase in enumerate(phrases):
            if max_samples and i >= max_samples:
                break
            data = process(phrase, data)
    return data


train_file = "fr-train.conll"
test_file = "fr_test_withtags.conll"
dev_file = "fr-dev.conll"

fr_train = load_conll(train_file, max_samples=500)  # Limit training data
fr_test = load_conll(test_file, max_samples=200)   # Limit test data
fr_dev = load_conll(dev_file, max_samples=200)     # Limit dev data


nlp = spacy.blank("fr")
lookups = Lookups()
lookups.add_table("lemma_lookup", {})
lookups.add_table("lemma_rules", {})
lookups.add_table("lemma_exc", {})
lookups.add_table("lemma_index", {})
nlp.vocab.lookups = lookups

ner = nlp.add_pipe("ner", last=True)
new_labels = ["ArtWork", "Artist", "VisualWork", "WrittenWork"]
for label in new_labels:
    ner.add_label(label)


train_data = []
for text, annotations in fr_train:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    train_data.append(example)


optimizer = nlp.begin_training()
epochs = 40 
batch_size = 32

for i in range(epochs):
    losses = {}
    batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
    for batch in batches:
        nlp.update(batch, drop=0.5, losses=losses)
    print(f"Epoch {i + 1}/{epochs}, Losses: {losses}")


nlp.to_disk("ner_model")


def evaluate_ner(model, test_data):
    true_labels = []
    predicted_labels = []
    all_labels = set()

    for text, annotations in test_data:
        doc = model(text)
        true_entities = [(start, end, label) for start, end, label in annotations["entities"]]
        predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        for true_entity in true_entities:
            start, end, label = true_entity
            true_labels.append(label)
            all_labels.add(label)

            matching_predicted = [ent[2] for ent in predicted_entities if ent[:2] == (start, end)]
            if matching_predicted:
                predicted_labels.append(matching_predicted[0])
            else:
                predicted_labels.append("NONE")

        for predicted_entity in predicted_entities:
            if predicted_entity[:2] not in [(ent[0], ent[1]) for ent in true_entities]:
                predicted_labels.append(predicted_entity[2])
                true_labels.append("NONE")
                all_labels.add(predicted_entity[2])

    all_labels = sorted(all_labels | {"NONE"})
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=all_labels, yticklabels=all_labels, fmt="d")
    plt.title("Confusion Matrix for NER")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig("confusion_matrix.png")


    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, labels=all_labels))

# Load the saved model and evaluate
nlp_finetuned = spacy.load("ner_model")
evaluate_ner(nlp_finetuned, fr_test)

