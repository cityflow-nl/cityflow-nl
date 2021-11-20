import json

import spacy


def lang_augmentation(path_to_data_json):
    with open(path_to_data_json) as f:
        data = json.load(f)
    nlp = spacy.load("en_core_web_sm")
    for track_id in data:
        new_text = ""
        for i, text in enumerate(data[track_id]):
            doc = nlp(text)

            for chunk in doc.noun_chunks:
                nb = chunk.text
                break
            data[track_id][i] = nb+". "+data[track_id][i]
            new_text += nb+"."
            if i < 2:
                new_text += " "
        data[track_id].append(new_text)
    with open(path_to_data_json.split(".")[-2]+"_nlpaug.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    lang_augmentation("data/test-queries.json")
