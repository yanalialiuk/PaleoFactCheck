import os
import pickle

import wikipedia

wikipedia.set_lang("en")

CACHE_DIR = "wiki_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Wikipedia article titles to fetch (English)
WIKI_ARTICLES = [
    "Dinosaur",
    "Paleontology",
    "Mesozoic",
    "Cretaceous–Paleogene extinction event",
    "Feathered dinosaur",
    "Theropoda",
    "Archosaur",
    "Triassic–Jurassic extinction event",
    "Cretaceous",
    "Fossil",
    "Archaeopteryx",
    "Tyrannosaurus",
    "Triceratops",
    "Velociraptor",
    "Brontosaurus",
    "Pterosaur",
    "Marine reptile",
    "Ammonite",
    "Coprolite",
    "Evolution of dinosaurs",
    "Cenozoic",
    "Dinosaur paleontology",
    "Spinosaurus",
    "Iguanodon",
    "Scelidosaurus",
    "Allosaurus",
    "Seismosaurus",
    "Pachycephalosaurus",
    "Ankylosaurus",
    "Diplodocus",
    "Ceratopsia",
    "Ornithopoda",
    "Styracosaurus",
    "Sarcosuchus",
    "Dinosaur morphology",
    "Paleoecology",
    "Paleoanthropology",
    "Dinosaur fossil",
    "Plant fossil",
    "Paleogeography",
    "Geochronology",
    "Paleoclimatology",
    "Paleozoology",
    "Paleobotany",
    "Protoceratops",
    "Oviraptor",
    "Megalosaurus",
    "Paleontological excavation",
    "Ichthyosaur",
    "Plesiosaur",
    "Mosasaur",
    "Edmontosaurus",
    "Deinonychus",
    "Pachycephalosauria",
    "Paleozoic",
    "Carnotaurus",
    "Giganotosaurus",
    "Coelophysis",
    "Brachiosaurus",
    "Pachyrhinosaurus",
    "Saltasaurus",
    "Microraptor",
    "Apatosaurus",
    "Ceratosaurus",
    "Styracosauridae",
    "Hadrosauridae",
    "Avemetatarsalia",
    "Titanosaur",
    "Compsognathus",
    "Paleoinvertebrate",
    "Marine fossil",
    "Paleoceanography",
    "Mesozoic flora",
    "Mesozoic fauna",
    "Plesiosauroidea",
    "Paleontology methods",
    "Herbivorous dinosaur",
    "Carnivorous dinosaur",
    "Fossilization",
    "Paleocontinent",
    "Cladistics",
    "Evolutionary biology",
]


def load_wiki_articles():
    import_path = os.environ.get("WIKI_CACHE_IMPORT")
    if import_path and os.path.isfile(import_path):
        with open(import_path, "rb") as f:
            return pickle.load(f)

    texts = {}
    for title in WIKI_ARTICLES:
        cache_file = os.path.join(CACHE_DIR, f"{title}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                texts[title] = pickle.load(f)
            continue

        try:
            text = wikipedia.page(title).content
        except wikipedia.exceptions.DisambiguationError as e:
            text = wikipedia.page(e.options[0]).content
        except wikipedia.exceptions.PageError:
            print(f"Article not found: '{title}'")
            continue

        texts[title] = text
        with open(cache_file, "wb") as f:
            pickle.dump(text, f)

    return texts
