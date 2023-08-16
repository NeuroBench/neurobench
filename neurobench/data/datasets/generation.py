from typing import Union, Optional, List, Tuple
from pathlib import Path
import os
import csv

from torch.utils.data import Dataset

import json
import numpy as np
import matplotlib.pyplot as plt



voice_commands = [
    "play", "pause", "stop", "next", "previous", "volume", "mute", "shuffle", "repeat",
    "alarm", "timer", "text", "call", "navigate", "weather", "notifications", "remind",
    "search", "open", "lights", "joke", "story", "translate", "add", "calendar", "define",
    "picture", "email", "flight", "music", "news", "home", "game", "taxi", "balance",
    "temperature", "heart", "message", "fact", "capital", "recipes", "clock", "bank",
    "workout", "coffee", "date", "ride", "restaurant", "track", "phone", "bedtime",
    "exchange", "stock", "rate", "market", "status", "trail", "place", "deal", "channel",
    "price", "store", "show", "sound", "app", "video", "quote", "cleaning",
    "go", "stop", "listen", "look", "read", "write", "speak", "think", "jump", "run",
    "walk", "eat", "drink", "sit", "stand", "sleep", "wake", "work", "relax", "study",
    "learn", "create", "draw", "paint", "sing", "dance", "play", "stop", "start", "finish",
    "complete", "pause", "resume", "open", "close", "turn", "twist", "pull", "push",
    "lift", "drop", "throw", "catch", "hit", "kick", "scream", "whisper", "smile", "laugh",
    "cry", "breathe", "blink", "wink", "shout", "whistle", "hug", "kiss", "swim", "dive",
    "float", "sink", "run", "crawl", "climb", "hang", "slide", "swing", "jump", "skip",
    "hop", "stand", "balance", "bend", "stretch", "reach", "twirl", "spin", "shake", "stir",
    "mix", "pour", "slice", "chop", "grate", "bake", "boil", "fry", "roast", "saute", "steam",
    "stir-fry", "season", "taste", "assemble", "arrange", "clean", "dust", "wipe", "scrub",
    "sweep", "mop", "vacuum", "polish", "organize", "tidy", "sort", "fold", "hang", "store",
    "unpack", "pack", "label", "sew", "stitch", "knit", "crochet", "trim", "cut", "glue",
    "stick", "tape", "tie", "knot", "fasten", "untie", "loosen", "tighten", "braid", "comb",
    "brush", "curl", "straighten", "trim", "shave", "massage", "lather", "rinse", "moisturize",
    "hydrate", "exfoliate", "apply", "remove", "dry", "pat", "blow", "brush", "curl", "straighten",
    "trim", "color", "dye", "shampoo", "condition", "treat", "protect", "style", "cover",
    "uncover", "conceal", "reveal", "wrap", "unwrap", "package", "unwrap", "unwrap", "peek",
    "watch", "observe", "notice", "see", "hear", "listen", "taste", "smell", "touch", "feel",
    "test", "examine", "inspect", "analyze", "evaluate", "assess", "measure", "weigh",
    "count", "record", "document", "photograph", "capture", "save", "discard", "keep",
    "preserve", "share", "show", "demonstrate", "teach", "learn", "understand",
    "forget", "remember", "study", "research", "explore", "discover", "uncover", "cover",
    "develop", "create", "invent", "design", "imagine", "innovate", "plan", "organize",
    "prepare", "ready", "assemble", "arrange", "set", "position", "align", "configure",
    "adjust", "adapt", "customize", "install", "uninstall", "download", "upload", "upgrade",
    "update", "sync", "backup", "restore", "copy", "paste", "cut", "undo", "redo", "save",
    "print", "scan", "fax", "send", "receive", "reply", "forward", "delete", "edit", "format",
    "resize", "crop", "rotate", "filter", "adjust", "enhance", "retouch", "convert", "export",
    "import", "embed", "link", "align", "justify", "center", "highlight", "annotate", "comment",
    "underline", "italicize", "bold", "change", "replace", "modify", "correct", "revise", "review",
    "proofread", "spellcheck", "accept", "reject", "submit", "request", "approve", "disapprove",
    "confirm", "verify", "validate", "decline", "cancel", "retry", "abort", "ignore", "accept",
    "refuse", "grant", "deny", "authorize", "prohibit", "permit", "allow", "forbid", "apply",
    "appeal", "challenge", "contest", "complain", "object", "question", "acknowledge", "thank",
    "apologize", "pardon", "greet", "introduce", "meet", "farewell", "smile", "laugh", "cry",
    "celebrate", "cheer", "toast", "encourage", "inspire", "motivate", "persuade", "advise",
    "recommend", "suggest", "instruct", "guide", "teach", "direct", "warn", "caution",
    "inform", "notify", "announce", "declare", "proclaim", "order", "request", "ask", "invite",
    "call", "shout", "whisper", "converse", "chat", "talk", "discuss", "debate", "argue",
    "share", "write", "read", "listen", "communicate", "express", "relay", "deliver", "present",
    "report", "convey", "conclude", "converse", "chat", "debate", "discuss", "describe",
    "define", "explain", "illustrate", "summarize", "outline", "organize", "compare", "contrast",
    "analyze", "evaluate", "synthesize", "interpret", "conclude", "inquire", "explore",
    "research", "examine", "study", "discover", "seek", "gather", "collect", "compile",
    "document", "record", "jot", "jot", "scribe", "transcribe", "log", "chart", "diagram",
    "illustrate", "depict", "map", "plot", "track", "trace", "monitor", "observe", "measure",
    "quantify", "analyze", "calculate", "compute", "estimate", "determine", "predict",
    "forecast", "project", "schedule", "plan", "arrange", "design", "prepare", "set", "develop",
    "construct", "build", "assemble", "create", "organize", "implement", "execute", "perform", "find"]

google_commands_list = [
    "Backward", "Bed", "Bird", "Cat", "Dog", "Down", "Eight", "Five",
    "Follow", "Forward", "Four", "Go", "Happy", "House", "Learn", "Left",
    "Marvin", "Nine", "No", "Off", "On", "One", "Right", "Seven", "Sheila",
    "Six", "Stop", "Three", "Tree", "Two", "Up", "Visual", "Wow", "Yes", "Zero"
]


VOICE_COMMANDS = voice_commands + google_commands_list


SAMPLE_RATE = 48000
ALL_LANGUAGES = ["en"] #, "es"]
FOLDER_AUDIO = "clips"

PRE_TRAINING_TAGS = []
EVALUATION_TAGS = []


def generate_mswc_fscil_splits(root: Union[str, Path], 
                               languages: List[str] = None, 
                               visualize: Optional[bool] = False):
    """
    Generate new MSWC split for a few-shot class-incremental (FSCIL) learning scenario with the following split.
    100 base classes with 500 train, 100 validation and 100 test samples each.
    100 evaluation classes with 200 samples each (to use in a 10 sessions of 10 way set-up with N shots support to train on per class and the rest as a query to evaluate performance).
    The 200 classes are arbitrarily chosen as common voice command words.
    The base ones are then the 100 of these with the most clips (at least 700) per sample and the evaluation ones as the 100 following ones.

    Returns: base_keywords, evaluation keywords (dictionarries)
    They represent the number of available samples per respective keyword in the original MSWC dataset (although the number is then clipped as detailed above).
    """

    base_keywords, evaluation_keywords = get_command_keywords(root, visualize=visualize)

    if languages is None:
        languages = ['en']

    print(languages)
    if languages  != ['en']:
        print('Other languages than english are not supported yet.')

    base_train_count = dict.fromkeys(base_keywords, 0) #{'train':0, 'val':0, 'test':0})
    base_test_count = dict.fromkeys(base_keywords, 0)
    base_val_count = dict.fromkeys(base_keywords, 0)
    evaluation_count = dict.fromkeys(evaluation_keywords, 0)

    for lang in languages:
        base_train_f = open(os.path.join(root, lang,  f'{lang}_{"base_train"}.csv'), 'w')
        base_val_f = open(os.path.join(root, lang,  f'{lang}_{"base_val"}.csv'), 'w')
        base_test_f = open(os.path.join(root, lang,  f'{lang}_{"base_test"}.csv'), 'w')
        evaluation_f = open(os.path.join(root, lang,  f'{lang}_{"evaluation"}.csv'), 'w')
        writer_base_train = csv.writer(base_train_f)
        writer_base_val = csv.writer(base_val_f)
        writer_base_test = csv.writer(base_test_f)
        writer_evaluation = csv.writer(evaluation_f)
        header = ['LINK', 'WORD', 'VALID', 'SPEAKER', 'GENDER']
        writer_base_train.writerow(header)
        writer_base_val.writerow(header)
        writer_base_test.writerow(header)
        writer_evaluation.writerow(header)

        with open(os.path.join(root, lang,  f'{lang}_splits.csv'), 'r') as f:
            for line in f:
                set, path, word, valid, speaker, gender = line.strip().split(',')
                
                # Skip header
                if set == "SET":
                    continue  

                ### Successively assign samples to train (500), validation (100) and test (100) set
                if word in base_keywords:
                    if base_train_count[word] <500:
                        writer_base_train.writerow([path, word, valid, speaker, gender])
                        base_train_count[word] +=1
                    if base_val_count[word] <100:
                        writer_base_val.writerow([path, word, valid, speaker, gender])
                        base_val_count[word] +=1
                    if base_test_count[word] <100:
                        writer_base_test.writerow([path, word, valid, speaker, gender])
                        base_test_count[word] +=1

                elif word in evaluation_keywords:
                    if evaluation_count[word] <200:
                        writer_evaluation.writerow([path, word, valid, speaker, gender])
                        evaluation_count[word] +=1


    base_train_f.close()
    base_val_f.close()
    base_test_f.close()
    evaluation_f.close()

    return base_keywords, evaluation_keywords



def get_command_keywords(root: Union[str, Path], visualize: bool = False):

    with open(os.path.join(root,"metadata.json"), 'r') as f:
        data = json.load(f)
        clips_counts = data['en']['wordcounts']

        mswc_commands = {}

        for keyword in VOICE_COMMANDS:
            
            if keyword in clips_counts:
                mswc_commands[keyword] = clips_counts[keyword]

        # Sort keywords based on available clips
        sorted_commands = sorted(mswc_commands.items(), key=lambda x: x[1], reverse=True)

        # Extract the top 200 keywords with the most clips
        pre_train_commands = dict(sorted_commands[:100])
        evaluation_commands = dict(sorted_commands[100:200])

    if visualize:
        from wordcloud import WordCloud

        # Plot word clouds of the base and evaluation words based on their total number of clips in the original MSWC dataset
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(pre_train_commands)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Top 100 Command Keywords with the Most Clips')

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(evaluation_commands)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Evaluation Command Keywords')

        plt.show()  


    return pre_train_commands, evaluation_commands