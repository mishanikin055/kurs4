"""CHAIR lexical rules, ported from authors BSD-2-Clause implementation."""

from pathlib import Path

import nltk

from annotation.evaluation.singular import singularize


class CHAIR:
    def __init__(self, imids, coco_path):
        self.imid_to_objects = {imid: [] for imid in imids}

        self.coco_path = coco_path

        # read in synonyms
        synonyms = open(Path(__file__).with_name("synonyms.txt")).readlines()
        synonyms = [s.strip().split(", ") for s in synonyms]
        self.mscoco_objects = []  # mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # Some hard coded rules for implementing CHAIR metrics on MSCOCO

        # common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = [
            "motor bike",
            "motor cycle",
            "air plane",
            "traffic light",
            "street light",
            "traffic signal",
            "stop light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "suit case",
            "sports ball",
            "baseball bat",
            "baseball glove",
            "tennis racket",
            "wine glass",
            "hot dog",
            "cell phone",
            "mobile phone",
            "teddy bear",
            "hair drier",
            "potted plant",
            "bow tie",
            "laptop computer",
            "stove top oven",
            "hot dog",
            "teddy bear",
            "home plate",
            "train track",
        ]

        # Hard code some rules for special cases in MSCOCO
        # qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = [
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "animal",
            "cub",
        ]
        # qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ["jet", "train"]

        # double_word_dict will map double words to the word they should be treated as in our analysis

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict["baby %s" % animal_word] = animal_word
            self.double_word_dict["adult %s" % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict["passenger %s" % vehicle_word] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

    def caption_to_words(self, caption):
        """
        Input: caption
        Output: MSCOCO words in the caption
        """

        # standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        words = [singularize(w) for w in words]

        # replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i : i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ("toilet" in words) & ("seat" in words):
            words = [word for word in words if word != "seat"]

        # get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        # return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words
