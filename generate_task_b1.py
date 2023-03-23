import sys
import csv
import openai
openai.api_key = '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'
from nltk.tokenize import word_tokenize

dialog_dict = {}
with open(sys.argv[1]) as input_file:
    reader = csv.DictReader(input_file)
    for row in reader:
        dialog_dict[row['encounter_id']] = row['dialogue']


def shorten_dialog(dialog_str, max_token_length):
    utterances = dialog_str.split('\n')
    good_utterances = []
    total_length = 0
    for utterance in utterances:
        utterance_length = len(word_tokenize(utterance))
        if total_length + utterance_length > max_token_length:
            break
        total_length += utterance_length
        good_utterances.append(utterance)
    return '\n'.join(good_utterances)


def predict_note(dialog_str):
    dialog_str = shorten_dialog(dialog_str, 1300)
    response = openai.Completion.create(
        model='davinci:ft-personal-2023-03-23-03-12-14',
        prompt=dialog_str + '\n\n###\n\n'
    )
    print(response['choices'])
    text = response['choices'][0]['text']
    return text

predicted_notes = {}
for ID, dialog in dialog_dict.items():
    category = predict_note(dialog)
    predicted_notes[ID] = category
    print(f"{ID} => {category}")

with open('taskB_gersteinlab_run1.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['TestID', 'SystemOutput'])
    for ID, dialog in dialog_dict.items():
        writer.writerow([ID, predicted_notes[ID]])
