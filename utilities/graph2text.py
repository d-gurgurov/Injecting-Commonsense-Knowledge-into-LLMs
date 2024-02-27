from sklearn.model_selection import train_test_split
import json
import pandas as pd

# setting a language code
language = "bg"

# loading the data extracted from ConceptNet
with open(f"/Users/macbookpro/Desktop/dfki/inject_kn/data/cn_data_{language}.json") as f:
    data = json.load(f)

def load_conceptnet_data(data):
    entities = []

    for word in data:
        for relationship in data[word]:
            start = relationship["start"].split('/')[-1].replace("_", " ")
            end = relationship["end"].split('/')[-1].replace("_", " ")
            rel = relationship["rel"]

            if start != end and rel != 'ExternalURL':
                entities.append((start, rel, end))

    return entities

# tranforming the row data into a list of tuples
cn = load_conceptnet_data(data)
print(f"Number of triples extracted from ConceptNet: {len(cn)}")

# preparing relationship types for constructing natural language sentences
relationships = []
for i in range(len(cn)):
    relationships.append(cn[i][1])

relationships = set(relationships)

print('Here are all possible relationship types:')
print(relationships)

relationship_mapping = {
    'Antonym': 'is the opposite of',
    'DerivedFrom': 'is derived from',
    'EtymologicallyDerivedFrom': 'is etymologically derived from',
    'EtymologicallyRelatedTo': 'is etymologically related to',
    'FormOf': 'is a form of',
    'HasContext': 'has context of',
    'IsA': 'is a type of',
    'RelatedTo': 'is related to',
    'SimilarTo': 'is similar to',
    'Synonym': 'is a synonym of',
    'SymbolOf': 'is a symbol of',
    'DistinctFrom': 'is distinct from',
}

# constructing natural language sentences
cn_sents = []

for triple in cn:
    subject, relationship, obj = triple
    if relationship in relationship_mapping:
        cn_sents.append(f"{subject} {relationship_mapping[relationship]} {obj}.")

print(f"Number of sentences constructed from ConceptNet: {len(cn_sents)}")

# splitting the data into training and validation sets
train_sents, validation_sents = train_test_split(cn_sents, test_size=0.1, random_state=42)

train_df = pd.DataFrame(train_sents, columns=["text"])
val_df = pd.DataFrame(validation_sents, columns=["text"])

# saving the data into csv files
train_df.to_csv(f"train_cn_{language}.csv", index=False)
val_df.to_csv(f"val_cn_{language}.csv", index=False)