import csv
import json
import pdb

from probing_norms.data import DIR_THINGS


SPECIAL_CATEGORIES = {
    "bat2": "sports equipment, used for cricket",
    "baton3": "sports equipment, used for twirling",
    "baton4": "sports equipment, used for relay races",
    "bracelet1": "watch strap",
    "camera1": "device used for photography",
    "camera2": "device used for video recording",
    "chicken1": "meat",
    "chicken2": "bird, animal",
    "clipper2": "tool used for finger- and toenails",
    "crystal1": "mineral",
    "crystal2": "rock",
    "hook1": "door fastener",
    "juicer1": "kitchen tool, manual squeezer",
    "pepper1": "spice",
    "pepper2": "vegetable",
    "screen1": "projection surface",
    "shell3": "nut covering",
    "stove1": "kitchen appliance",
    "stove2": "heating device",
}


SPECIAL_DEFINITIONS = {
    "bat2": "a piece of equipment used by batters in the sport of cricket to hit the ball",
}


def parse_line(elems):
    id_ = elems[1]
    concept = elems[0]
    category = SPECIAL_CATEGORIES.get(id_, elems[22])
    definition = SPECIAL_DEFINITIONS.get(id_, elems[7])
    return {
        "id": id_,
        "concept": concept,
        "category": category,
        "definition": definition,
    }


if __name__ == "__main__":
    path1 = DIR_THINGS / "_concepts-metadata_things.tsv"
    with open(path1) as csvfile:
        data = [
            parse_line(elems)
            for i, elems in enumerate(csv.reader(csvfile, delimiter="\t"))
            if i > 0
        ]

    path2 = "data/things/concepts-and-categories.json"
    with open(path2, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)