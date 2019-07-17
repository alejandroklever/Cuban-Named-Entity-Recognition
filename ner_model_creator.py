import json
import logging
import pickle
import random
import spacy
from pathlib import Path
from spacy.util import compounding, minibatch
import tempfile
from fire import Fire

"""Global"""
LABELS = []
TRAIN_DATA = []
MODEL = None
ID = 'es'


def train_model(output_dir, model=None, language='es', n_iter=100):
    """Load the MODEL, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank(f"{language}")  # create blank Language class
        print(f"Created blank '{language}' MODEL")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new MODEL
        if MODEL is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # save MODEL to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved MODEL to", output_dir)

        # test the saved MODEL
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Org", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


def create_train_data(input_file=None, output_file=None):
    try:
        training_data = []
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1, label))

            training_data.append((text, {"entities": entities}))

        print(training_data)

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None


def create_json(input_path, output_path, unknown_label):
    ans = []
    try:
        f = open(input_path, 'r')  # input file
        fp = open(output_path, 'w')  # output file
        data_dict = {}
        annotations = []
        label_dict = {}
        s = ''
        start = 0
        for line in f:
            if line[0:len(line) - 1] != '.\tO':
                word, entity = line.split('\t')
                s += word + " "
                entity = entity[:len(entity) - 1]
                if entity != unknown_label and len(entity) != 1:
                    d = {'text': word, 'start': start, 'end': start + len(word) - 1}
                    try:
                        label_dict[entity].append(d)
                    except Exception as e:
                        label_dict[entity] = []
                        label_dict[entity].append(d)
                start += len(word) + 1
            else:
                data_dict['content'] = s
                print(s)
                s = ''
                label_list = []
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if label_dict[ents][i]['text'] != '':
                            l = [ents, label_dict[ents][i]]
                            for j in range(i + 1, len(label_dict[ents])):
                                if label_dict[ents][i]['text'] == label_dict[ents][j]['text']:
                                    di = {'start': label_dict[ents][j]['start'],
                                          'end': label_dict[ents][j]['end'],
                                          'text': label_dict[ents][i]['text']}
                                    l.append(di)
                                    label_dict[ents][j]['text'] = ''
                            label_list.append(l)

                for entities in label_list:
                    label = {'label': [entities[0]], 'points': entities[1:]}
                    annotations.append(label)
                data_dict['annotation'] = annotations
                annotations = []
                json.dump(data_dict, fp)
                fp.write('\n')
                ans.append({'annotation' : [ a for a in annotations], 'content': s})
                data_dict = {}
                start = 0
                label_dict = {}
    except Exception as e:
        logging.exception("Unable to process file" + "\n" + "error = " + str(e))
        return None

    return ans


def load_train_data(data_path):
    global TRAIN_DATA
    with open(data_path, 'rb') as fp:
        TRAIN_DATA = pickle.load(fp)


def set_existing_model(model):
    MODEL = model


def set_blank_model(id):
    MODEL = None
    ID = spacy.blank(id)


def create_model(source_tsv, model_output, language='es'):

    tempjson = tempfile.NamedTemporaryFile()
    td = tempfile.NamedTemporaryFile()
    n = tempjson.name
    create_json(source_tsv,n,'abc')
    create_train_data(n, td.name)
    load_train_data(td.name)
    train_model(model_output, language=language)

def update_model(source_tsv, model_output, source_model):

    tempjson = tempfile.NamedTemporaryFile()
    td = tempfile.NamedTemporaryFile()
    n = tempjson.name
    create_json(source_tsv,n,'abc')
    create_train_data(n, td.name)
    load_train_data(td.name)
    train_model(model_output, model=source_model )

if __name__ == "__main__":
    Fire({'create_model': create_model, 'update_model': update_model})