from functools import lru_cache


SYNONYMS = {

    # ─────────────────────────────────────────────
    # SPICES & HERBS
    # ─────────────────────────────────────────────

    # Ginger
    "adrak": "ginger",
    "adrakh": "ginger",
    "sonth": "ginger",
    "saunth": "ginger",
    "dry ginger": "ginger",
    "fresh ginger": "ginger",
    "ground ginger": "ginger",
    "ginger root": "ginger",
    "gingerroot": "ginger",
    "ginger powder": "ginger",
    "young ginger": "ginger",
    "green ginger": "ginger",
    "ginger piece": "ginger",
    "ginger knob": "ginger",
    "ginger cube": "ginger",

    # Turmeric
    "haldi": "turmeric",
    "haldi powder": "turmeric",
    "ground turmeric": "turmeric",
    "turmeric powder": "turmeric",
    "turmeric root": "turmeric",
    "manjal": "turmeric",
    "pasupu": "turmeric",

    # Cumin
    "jeera": "cumin",
    "zeera": "cumin",
    "jira": "cumin",
    "safed jeera": "cumin",
    "ground cumin": "cumin",
    "cumin powder": "cumin",
    "jeera powder": "cumin",
    "cumin seeds": "cumin seed",
    "jeera seeds": "cumin seed",
    "white cumin": "cumin seed",
    "shah jeera": "black cumin seed",
    "kala jeera": "black cumin seed",
    "black cumin": "black cumin seed",
    "black jeera": "black cumin seed",


# Green Chili variants — "green chilies" is the correct DB canonical
    "green mirch":   "green chilies",
    "hari mirch":    "green chilies",
    "green chilli":  "green chilies",   # double-l variant not in DB
    "green chilly":  "green chilies",
    "green birch":   "green chilies",   # Whisper mishear
    "green merch":   "green chilies",   # Whisper mishear
    "green nirch":   "green chilies",   # Whisper mishear
    "green mitch":   "green chilies",



    # Coriander
    "dhania": "coriander",
    "dhanya": "coriander",
    "dhaniya": "coriander",
    "cilantro": "coriander",
    "chinese parsley": "coriander",
    "fresh coriander": "coriander",
    "coriander leaves": "coriander leaf",
    "cilantro leaves": "coriander leaf",
    "cilantro leaf": "coriander leaf",
    "dhania leaves": "coriander leaf",
    "green coriander": "coriander",
    "coriander powder": "coriander powder",
    "dhaniya powder": "coriander powder",
    "dhania powder": "coriander powder",
    "ground coriander": "coriander powder",
    "coriander seeds": "coriander seed",
    "dhania seeds": "coriander seed",

    # Red Chili / Chili Powder
    "lal mirch": "red chili pepper",
    "lal mirchi": "red chili pepper",
    "mirchi": "chili",
    "mirch": "chili",
    "red chilli": "red chili pepper",
    "red chile": "red chili pepper",
    "red chili": "red chili pepper",
    "red chili powder": "red chili powder",
    "lal mirch powder": "red chili powder",
    "cayenne": "cayenne pepper",
    "cayenne chili": "cayenne pepper",
    "ground cayenne": "cayenne pepper",
    "kashmiri mirch": "kashmiri chili powder",
    "kashmiri lal mirch": "kashmiri chili powder",
    "kashmiri red chilli": "kashmiri chili powder",
    "deggi mirch": "kashmiri chili powder",
    "chilli flakes": "red pepper flake",
    "chili flakes": "red pepper flake",
    "crushed red pepper": "red pepper flake",
    "red chilli flakes": "red pepper flake",
    "pepper flakes": "red pepper flake",

    # Black Pepper
    "kali mirch": "black pepper",
    "kali mirchi": "black pepper",
    "ground black pepper": "black pepper",
    "freshly ground pepper": "black pepper",
    "crushed black pepper": "black pepper",
    "black peppercorns": "black peppercorn",
    "whole black pepper": "black peppercorn",
    "kali mirch whole": "black peppercorn",

    # Garlic
    "lahsun": "garlic",
    "lasun": "garlic",
    "lashun": "garlic",
    "lehsoon": "garlic",
    "garlic cloves": "garlic clove",
    "clove of garlic": "garlic clove",
    "garlic pods": "garlic clove",
    "garlic flakes": "garlic flake",
    "garlic granules": "garlic granule",
    "powdered garlic": "garlic powder",
    "garlic paste": "garlic paste",
    "minced garlic": "garlic paste",
    "crushed garlic": "garlic paste",
    "garlic clove paste": "garlic paste",
    "head of garlic": "head garlic",
    "whole garlic": "head garlic",

    # Onion
    "pyaz": "onion",
    "pyaaz": "onion",
    "kanda": "onion",
    "dungri": "onion",
    "red onion": "red onion",
    "lal pyaz": "red onion",
    "white onion": "white onion",
    "yellow onion": "yellow onion",
    "spring onion": "spring onion",
    "hara pyaz": "green onion",
    "hari pyaz": "green onion",
    "green onions": "green onion",
    "scallion": "scallion",
    "spring onions": "spring onion",
    "shallots": "shallot",
    "eschallot": "shallot",
    "french shallot": "shallot",
    "onion powder": "onion powder",
    "onion flakes": "onion flake",
    "dried onion": "onion flake",

    # Tomato
    "tamatar": "tomato",
    "tamaatar": "tomato",
    "lal tamatar": "tomato",
    "tomatoes": "tomato",
    "cherry tomatoes": "cherry tomato",
    "plum tomatoes": "plum tomato",
    "roma tomato": "plum tomato",
    "roma tomatoes": "plum tomato",
    "tomato puree": "tomato puree",
    "tomato passata": "tomato puree",
    "crushed tomato": "tomato puree",
    "tomato paste": "tomato paste",
    "tomato concentrate": "tomato paste",
    "tomato sauce": "tomato sauce",
    "tomato ketchup": "ketchup",

    # Cardamom
    "elaichi": "cardamom",
    "elaichi powder": "cardamom powder",
    "hari elaichi": "green cardamom",
    "choti elaichi": "green cardamom",
    "green cardamom pods": "green cardamom pod",
    "badi elaichi": "cardamom pod",
    "kali elaichi": "cardamom pod",
    "black cardamom": "cardamom pod",
    "black cardamom pods": "black cardamom pod",
    "ground cardamom": "cardamom powder",
    "cardamom seeds": "cardamom seed",

    # Cloves
    "laung": "clove",
    "lavang": "clove",
    "long": "clove",
    "cloves": "clove",
    "whole cloves": "clove",
    "clove powder": "clove powder",
    "ground cloves": "clove powder",

    # Cinnamon
    "dalchini": "cinnamon",
    "dal chini": "cinnamon",
    "cinnamon stick": "cinnamon",
    "cinnamon powder": "cinnamon",
    "ground cinnamon": "cinnamon",
    "cinnamon quill": "cinnamon",

    # Mustard
    "sarson": "mustard",
    "rai": "mustard seed",
    "mustard seeds": "mustard seed",
    "mustard powder": "mustard powder",
    "ground mustard": "mustard powder",
    "dijon": "dijon mustard",
    "french mustard": "dijon mustard",
    "wholegrain mustard": "grain mustard",
    "whole grain mustard": "grain mustard",
    "yellow mustard seeds": "yellow mustard seed",
    "mustard oil": "mustard oil",
    "sarson ka tel": "mustard oil",

    # Fenugreek
    "methi": "fenugreek",
    "methi seeds": "fenugreek seed",
    "methi dana": "fenugreek seed",
    "fenugreek seeds": "fenugreek seed",
    "methi leaves": "fenugreek leaf",
    "dried fenugreek": "kasuri methi",
    "kasuri methi": "kasuri methi",
    "kasoori methi": "kasuri methi",

    # Garam Masala
    "garam masala powder": "garam masala",
    "mixed spice": "garam masala",
    "whole spices": "garam masala",
    "curry spice mix": "garam masala",

    # Curry Powder
    "curry masala": "curry powder",
    "madras curry": "madras curry powder",
    "yellow curry": "yellow curry powder",

    # Turmeric alt names already above

    # Asafoetida
    "hing": "asafoetida",
    "asafoetida": "asafoetida",
    "asafetida": "asafoetida",
    "heeng": "asafoetida",


    # Ajwain
    "ajwain": "ajwain",
    "carom seeds": "ajwain",
    "bishop's weed": "ajwain",
    "thymol seeds": "ajwain",

    # Nigella Seeds
    "kalonji": "kalonji",
    "nigella seeds": "nigella seed",
    "onion seeds": "nigella seed",
    "black onion seeds": "nigella seed",
    "black seeds": "nigella seed",

    # Poppy Seeds
    "khus khus": "poppy seed",
    "posto": "poppy seed",
    "poppy seeds": "poppy seed",
    "white poppy seeds": "white poppy seed",

    # Star Anise
    "chakra phool": "star anise",
    "star anise pods": "star anise pod",
    "badiyan": "star anise",

    # Saffron
    "kesar": "saffron",
    "zafran": "saffron",
    "saffron strands": "saffron strand",
    "saffron threads": "saffron thread",

    # Mango Powder
    "amchur": "mango powder",
    "amchoor": "mango powder",
    "amchur powder": "mango powder",
    "dried mango powder": "mango powder",

    # Sumac
    "sumac": "sumac",
    "sumaq": "sumac",
    "sumak": "sumac",

    # Paprika
    "paprika powder": "paprika",
    "smoked paprika": "paprika",
    "sweet paprika": "paprika",
    "red paprika": "paprika",

    # Parsley
    "flat leaf parsley": "flat leaf parsley",
    "italian parsley": "italian parsley",
    "curly parsley": "parsley",
    "fresh parsley": "parsley",
    "dried parsley": "parsley",
    "ajmod": "parsley",

    # Mint
    "pudina": "mint",
    "peppermint": "mint",
    "spearmint": "spearmint",
    "fresh mint": "mint",
    "mint leaves": "mint leaf",
    "pudina leaves": "mint leaf",

    # Basil
    "fresh basil": "basil",
    "basil leaves": "basil leaf",
    "thai basil": "thai basil",
    "sweet basil": "basil",
    "tulsi": "basil",

    # Oregano
    "dried oregano": "oregano",
    "fresh oregano": "oregano leaf",
    "mexican oregano": "mexican oregano",
    "greek oregano": "greek oregano",

    # Thyme
    "fresh thyme": "thyme",
    "dried thyme": "thyme",
    "thyme leaves": "thyme leaf",
    "lemon thyme": "lemon thyme",

    # Rosemary
    "fresh rosemary": "rosemary",
    "dried rosemary": "rosemary",
    "rosemary leaves": "rosemary leaf",

    # Sage
    "fresh sage": "sage",
    "dried sage": "sage leaf",

    # Bay Leaf
    "tej patta": "bay leaf",
    "tez patta": "bay leaf",
    "bay leaves": "bay leaf",
    "indian bay leaf": "bay leaf",
    "laurel leaf": "bay leaf",

    # Lemongrass
    "lemon grass": "lemongrass",
    "serai": "lemongrass",
    "cymbopogon": "lemongrass",

    # Curry Leaf
    "kadi patta": "curry leaf",
    "kadhi patta": "curry leaf",
    "meetha neem": "curry leaf",
    "curry leaves": "curry leaf",

    # Dill
    "suva": "dill",
    "shepu": "dill",
    "soa": "dill",
    "dill weed": "dill weed",
    "fresh dill": "dill",
    "dried dill": "dill weed",

    # Peppermint Extract
    "peppermint oil": "peppermint extract",

    # ─────────────────────────────────────────────
    # VEGETABLES
    # ─────────────────────────────────────────────

    # Potato
    "aloo": "potato",
    "alu": "potato",
    "batata": "potato",
    "spud": "potato",
    "potatoes": "potato",
    "russet": "russet potato",
    "idaho potato": "idaho potato",
    "baby potatoes": "baby potato",
    "new potatoes": "new potato",
    "red potatoes": "red potato",

    # Carrot
    "gajar": "carrot",
    "carrots": "carrot",
    "baby carrots": "carrot",

    # Spinach
    "palak": "spinach",
    "baby spinach": "spinach",
    "saag": "spinach",

    # Eggplant / Brinjal
    "baingan": "eggplant",
    "brinjal": "eggplant",
    "aubergine": "eggplant",
    "begun": "eggplant",

    # Zucchini / Courgette
    "zucchini": "zucchini",
    "courgette": "zucchini",
    "baby zucchini": "zucchini",

    # Cabbage
    "patta gobhi": "cabbage",
    "patta gobi": "cabbage",
    "band gobhi": "cabbage",
    "napa cabbage": "cabbage",
    "savoy cabbage": "cabbage",

    # Cauliflower
    "gobhi": "cauliflower",
    "gobi": "cauliflower",
    "phool gobhi": "cauliflower",
    "cauliflower florets": "cauliflower floret",

    # Broccoli
    "broccoli florets": "broccoli floret",
    "broccoli spears": "broccoli spear",

    # Bell Pepper / Capsicum
    "shimla mirch": "bell pepper",
    "capsicum": "capsicum",
    "green capsicum": "green capsicum",
    "red capsicum": "red bell pepper",
    "yellow capsicum": "yellow capsicum",
    "green bell pepper": "green bell pepper",
    "red bell pepper": "red bell pepper",
    "yellow bell pepper": "yellow bell pepper",
    "orange bell pepper": "orange bell pepper",
    "sweet pepper": "bell pepper",
    "sweet red pepper": "red bell pepper",

    # Green Chili
    "hari mirch": "green chilies",
    "green chilli": "green chilies",
    "green chillies": "green chilies",
    "green chile": "green chilies",
    "serrano": "serrano chili",
    "jalapeno": "jalapeno pepper",
    "jalapeño": "jalapeno pepper",
    "jalapenos": "jalapeno pepper",
    "chipotle": "chipotle pepper",
    "habanero": "habanero",
    "scotch bonnet": "scotch bonnet pepper",

    # Peas
    "matar": "pea",
    "mutter": "pea",
    "green peas": "green pea",
    "hare matar": "green pea",
    "frozen peas": "pea",
    "sugar snap peas": "sugar snap pea",
    "snow peas": "snow pea",
    "mange tout": "snow pea",

    # Mushroom
    "khumb": "mushroom",
    "dhingri": "mushroom",
    "button mushrooms": "button mushroom",
    "cremini": "cremini mushroom",
    "crimini": "cremini mushroom",
    "portobello": "portobello mushroom",
    "portabella": "portabella mushroom",
    "shiitake": "shiitake mushroom",
    "porcini": "porcini mushroom",
    "oyster mushrooms": "oyster mushroom",
    "enoki": "enoki mushroom",
    "wild mushrooms": "wild mushroom",
    "chestnut mushrooms": "chestnut mushroom",
    "chanterelle": "chanterelle mushroom",

    # Cucumber
    "kheera": "cucumber",
    "khira": "cucumber",
    "kakdi": "cucumber",
    "english cucumber": "english cucumber",
    "gherkin": "gherkin",
    "pickling cucumber": "gherkin",

    # Asparagus
    "asparagus spears": "asparagus",

    # Leek
    "leeks": "leek",

    # Celery
    "ajwain leaves": "celery",
    "celery stalks": "celery rib",
    "celery sticks": "celery rib",
    "celery ribs": "celery rib",

    # Artichoke
    "artichoke hearts": "artichoke heart",

    # Water Chestnut
    "singhara": "water chestnut",
    "water chestnuts": "water chestnut",

    # Corn
    "makka": "corn",
    "makai": "corn",
    "maize": "corn",
    "sweetcorn": "sweetcorn",
    "sweet corn": "corn",
    "corn on the cob": "corn cob",
    "baby corn": "baby corn",
    "corn kernels": "corn kernel",

    # Bamboo Shoots
    "bamboo shoots": "bamboo shoot",

    # ─────────────────────────────────────────────
    # DAIRY
    # ─────────────────────────────────────────────

    # Yogurt / Curd
    "dahi": "yogurt",
    "curd": "yogurt",
    "hung curd": "yogurt",
    "yogurt": "yogurt",
    "yoghurt": "yogurt",
    "greek yogurt": "greek yogurt",
    "greek yoghurt": "greek yogurt",
    "strained yogurt": "greek yogurt",
    "thick yogurt": "yogurt",
    "sour cream yogurt": "yogurt",
    "set yogurt": "yogurt",

    # Milk
    "doodh": "milk",
    "whole milk": "milk",
    "full cream milk": "milk",
    "skimmed milk": "milk",
    "semi skimmed milk": "milk",
    "2% milk": "milk",
    "fat free milk": "milk",
    "almond milk": "almond milk",
    "soymilk": "soymilk",
    "soy milk": "soymilk",
    "oat milk": "milk",
    "coconut milk": "coconut milk",
    "nariyal doodh": "coconut milk",

    # Butter
    "makhan": "butter",
    "unsalted butter": "butter",
    "salted butter": "butter",
    "softened butter": "butter",
    "melted butter": "butter",

    # Ghee
    "clarified butter": "ghee",
    "desi ghee": "ghee",
    "pure ghee": "ghee",

    # Cream
    "heavy cream": "heavy cream",
    "heavy whipping cream": "heavy whipping cream",
    "whipping cream": "whipping cream",
    "double cream": "heavy cream",
    "single cream": "cream",
    "half and half": "cream",
    "fresh cream": "cream",
    "malai": "cream",

    # Cheese
    "paneer": "paneer",
    "cottage cheese": "cottage cheese",
    "ricotta": "ricotta cheese",
    "mozzarella": "mozzarella cheese",
    "parmesan": "parmesan cheese",
    "parmigiano": "parmesan cheese",
    "cheddar": "cheddar cheese",
    "feta": "feta cheese",
    "swiss cheese": "swiss cheese",
    "gruyere": "gruyere cheese",
    "brie": "cheese",
    "gouda": "cheese",
    "provolone": "provolone cheese",
    "romano": "romano cheese",
    "blue cheese": "blue cheese",
    "gorgonzola": "blue cheese",

    # Buttermilk
    "chhaas": "buttermilk",
    "chaas": "buttermilk",
    "mattha": "buttermilk",
    "lassi": "buttermilk",

    # Sour Cream
    "creme fraiche": "creme fraiche",
    "soured cream": "creme fraiche",

    # ─────────────────────────────────────────────
    # FLOURS & CEREALS
    # ─────────────────────────────────────────────

    # All Purpose Flour
    "maida": "flour",
    "all purpose flour": "purpose flour",
    "plain flour": "plain flour",
    "refined flour": "purpose flour",
    "all-purpose flour": "purpose flour",
    "ap flour": "purpose flour",
    "white flour": "white flour",

    # Wheat Flour / Atta
    "atta": "whole wheat flour",
    "whole wheat flour": "whole wheat flour",
    "wholemeal flour": "whole wheat flour",
    "chapati flour": "whole wheat flour",
    "durum flour": "wheat flour",
    "bread flour": "bread flour",
    "self rising flour": "self rising flour",
    "self raising flour": "self raising flour",
    "cake flour": "cake flour",

    # Gram Flour
    "besan": "chickpea flour",
    "chickpea flour": "chickpea flour",
    "garbanzo flour": "chickpea flour",
    "chana atta": "chickpea flour",
    "senagapindi": "chickpea flour",

    # Rice Flour
    "chawal ka atta": "rice flour",
    "rice powder": "rice flour",

    # Cornstarch
    "cornflour": "cornflour",
    "corn starch": "cornstarch",
    "cornstarch": "cornstarch",
    "maize starch": "cornstarch",
    "arrowroot": "cornstarch",

    # Cornmeal
    "polenta": "polenta",
    "yellow cornmeal": "yellow cornmeal",
    "masa harina": "masa harina",
    "makki ka atta": "cornmeal",

    # Rice
    "chawal": "rice",
    "basmati": "basmati rice",
    "basmati chawal": "basmati rice",
    "jasmine rice": "jasmine rice",
    "white rice": "white rice",
    "brown rice": "brown rice",
    "arborio": "arborio rice",
    "risotto rice": "arborio rice",
    "long grain rice": "rice",
    "short grain rice": "rice",
    "wild rice": "wild rice",
    "sona masuri": "rice",

    # Pasta
    "spaghetti": "spaghetti",
    "penne": "penne pasta",
    "macaroni": "macaroni",
    "elbow macaroni": "elbow macaroni",
    "linguine": "linguine",
    "fettuccine": "pasta",
    "tagliatelle": "pasta",
    "rigatoni": "pasta",
    "orzo": "orzo pasta",
    "angel hair": "angel hair pasta",
    "vermicelli": "pasta",
    "noodles": "pasta",

    # Oats
    "oatmeal": "oat",
    "rolled oats": "oat",
    "quick oats": "oat",
    "oat flakes": "oat",
    "porridge oats": "oat",

    # ─────────────────────────────────────────────
    # LEGUMES & PULSES
    # ─────────────────────────────────────────────

    # Chickpeas / Chana
    "chana": "chickpea",
    "chole": "chickpea",
    "kabuli chana": "chickpea",
    "garbanzo": "chickpea",
    "garbanzo beans": "garbanzo bean",
    "bengal gram": "chickpea",
    "safed chana": "chickpea",
    "white chickpeas": "chickpea",
    "chick peas": "chick pea",
    "kala chana": "black bean",
    "desi chana": "chickpea",

    # Lentils / Dal
    "dal": "lentil",
    "daal": "lentil",
    "masoor dal": "red lentil",
    "masoor": "red lentil",
    "red lentils": "red lentil",
    "green lentils": "green lentil",
    "sabut masoor": "lentil",
    "urad dal": "urad dal",
    "urad": "urad dal",
    "black gram": "urad dal",
    "dhuli urad": "urad dal",
    "moong dal": "mung bean",
    "moong": "mung bean",
    "mung beans": "mung bean",
    "green gram": "lentil",
    "toor dal": "lentil",
    "arhar dal": "lentil",
    "chana dal": "lentil",
    "split peas": "yellow pea",

    # Kidney Beans / Rajma
    "rajma": "kidney bean",
    "red kidney beans": "red kidney bean",
    "kidney beans": "kidney bean",

    # Black Beans
    "black beans": "black bean",

    # Pinto Beans
    "pinto beans": "pinto bean",

    # Cannellini Beans
    "cannellini": "cannellini bean",
    "white beans": "white bean",
    "navy beans": "navy bean",
    "haricot beans": "navy bean",
    "borlotti beans": "pinto bean",
    "great northern beans": "great northern bean",

    # Lima Beans
    "butter beans": "lima bean",
    "lima beans": "lima bean",

    # Fava Beans
    "broad beans": "fava bean",
    "fava": "fava bean",
    "bakla": "fava bean",

    # Black Eyed Peas
    "lobia": "black eyed pea",
    "cowpeas": "black eyed pea",
    "black eyed beans": "black eyed pea",

    # Soy
    "tofu": "tofu",
    "bean curd": "tofu",
    "silken tofu": "silken tofu",
    "firm tofu": "tofu",
    "soy milk": "soymilk",
    "tempeh": "tempeh",

    # ─────────────────────────────────────────────
    # OILS & FATS
    # ─────────────────────────────────────────────

    # Olive Oil
    "tel": "oil",
    "olive oil": "olive oil",
    "extra virgin olive oil": "extra virgin olive oil",
    "evoo": "extra virgin olive oil",
    "light olive oil": "olive oil",

    # Vegetable / Canola / Sunflower
    "vegetable oil": "vegetable oil",
    "cooking oil": "cooking oil",
    "canola oil": "canola oil",
    "rapeseed oil": "canola oil",
    "sunflower oil": "sunflower oil",
    "corn oil": "corn oil",
    "peanut oil": "peanut oil",
    "groundnut oil": "peanut oil",
    "moongphali tel": "peanut oil",

    # Sesame Oil
    "til ka tel": "sesame oil",
    "gingelly oil": "sesame oil",
    "sesame oil": "sesame oil",
    "toasted sesame oil": "dark sesame oil",

    # Coconut Oil
    "coconut oil": "oil",
    "nariyal tel": "oil",

    # Mustard Oil (already above)

    # Butter / Ghee (already above)

    # ─────────────────────────────────────────────
    # SWEETENERS
    # ─────────────────────────────────────────────

    # Sugar
    "cheeni": "sugar",
    "chini": "sugar",
    "shakkar": "sugar",
    "granulated sugar": "sugar",
    "white sugar": "white sugar",
    "caster sugar": "caster sugar",
    "castor sugar": "caster sugar",
    "superfine sugar": "superfine sugar",
    "icing sugar": "icing sugar",
    "powdered sugar": "confectioner sugar",
    "confectioners sugar": "confectioner sugar",
    "brown sugar": "brown sugar",
    "dark brown sugar": "dark brown sugar",
    "demerara sugar": "demerara sugar",
    "muscovado": "muscovado sugar",
    "palm sugar": "palm sugar",
    "jaggery": "jaggery",
    "gur": "palm sugar",
    "gud": "jaggery",
    "khandsari": "sugar",

    # Honey
    "shahad": "honey",
    "shehad": "honey",
    "natural honey": "honey",
    "clear honey": "clear honey",
    "liquid honey": "liquid honey",
    "raw honey": "honey",

    # Maple Syrup
    "maple": "maple syrup",
    "pure maple syrup": "maple syrup",

    # Golden Syrup
    "golden syrup": "golden syrup",
    "treacle": "molasses",
    "molasses": "molasses",
    "corn syrup": "corn syrup",
    "light corn syrup": "corn syrup",
    "dark corn syrup": "dark corn syrup",

    # ─────────────────────────────────────────────
    # MEATS & PROTEINS
    # ─────────────────────────────────────────────

    # Chicken
    "murgi": "chicken",
    "murg": "chicken",
    "murgh": "chicken",
    "chicken breast": "chicken breast",
    "chicken thigh": "chicken thigh",
    "chicken thighs": "chicken thigh",
    "chicken drumstick": "chicken",
    "whole chicken": "chicken",
    "boneless chicken": "chicken breast",
    "chicken mince": "chicken",
    "ground chicken": "chicken",

    # Egg
    "anda": "egg",
    "anday": "egg",
    "eggs": "egg",
    "whole egg": "egg",
    "egg yolk": "egg yolk",
    "egg yolks": "egg yolk",
    "egg white": "egg white",
    "egg whites": "egg white",

    # Beef
    "gosht": "meat",
    "gai ka gosht": "beef",
    "ground beef": "beef",
    "minced beef": "beef",
    "beef mince": "beef",
    "beef steak": "beef steak",
    "chuck": "beef",

    # Mutton / Lamb
    "bakra gosht": "lamb",
    "mutton": "lamb",
    "lamb mince": "lamb",
    "ground lamb": "lamb",
    "lamb chops": "lamb",

    # Pork
    "pork belly": "pork",
    "pork mince": "pork",
    "ground pork": "pork",
    "pork chops": "pork chop",

    # Bacon
    "streaky bacon": "bacon",
    "back bacon": "bacon",
    "smoked bacon": "bacon",
    "bacon rashers": "bacon",
    "bacon strips": "bacon",
    "pancetta": "bacon",

    # Ham
    "prosciutto": "prosciutto",
    "cured ham": "ham",

    # Turkey
    "turkey breast": "turkey",
    "ground turkey": "turkey",

    # Sausage
    "chorizo": "chorizo sausage",
    "italian sausage": "italian sausage",
    "breakfast sausage": "sausage",
    "pork sausage": "sausage",
    "frankfurter": "sausage",
    "hot dog": "hot dog",
    "salami": "salami",

    # Broth / Stock
    "chicken broth": "chicken broth",
    "chicken stock": "chicken stock",
    "vegetable broth": "vegetable broth",
    "vegetable stock": "vegetable stock",
    "beef broth": "beef broth",
    "beef stock": "beef stock",
    "bone broth": "chicken broth",
    "bouillon": "bouillon cube",
    "stock cube": "bouillon cube",
    "chicken bouillon": "chicken bouillon cube",
    "beef bouillon": "beef bouillon cube",

    # ─────────────────────────────────────────────
    # SEAFOOD & FISH
    # ─────────────────────────────────────────────

    # Shrimp / Prawn
    "jhinga": "shrimp",
    "jhenga": "prawn",
    "prawns": "prawn",
    "shrimps": "shrimp",
    "large prawns": "large shrimp",
    "king prawns": "king prawn",
    "jumbo shrimp": "jumbo shrimp",
    "tiger prawns": "tiger shrimp",

    # Salmon
    "salmon": "salmon",
    "smoked salmon": "salmon fillet",
    "salmon fillets": "salmon fillet",

    # Tuna
    "tuna fish": "tuna",
    "canned tuna": "tuna water",

    # Cod / White Fish
    "machli": "fish",
    "machhli": "fish",
    "cod fillets": "cod fillet",
    "white fish": "white fish fillet",
    "haddock": "haddock fillet",
    "tilapia": "tilapia fillet",
    "sea bass": "sea bass fillet",
    "halibut": "halibut fillet",
    "trout": "trout",
    "red snapper": "red snapper",

    # Anchovy
    "anchovies": "anchovy",
    "anchovy fillets": "anchovy fillet",

    # Shellfish
    "mussels": "mussel",
    "clams": "clam",
    "scallops": "scallop",
    "oysters": "oyster",
    "crab": "crab",
    "crab meat": "crabmeat",
    "lobster": "lobster",
    "squid": "squid",
    "calamari": "calamari",
    "nori": "nori",
    "seaweed": "nori",

    # ─────────────────────────────────────────────
    # FRUITS
    # ─────────────────────────────────────────────

    # Lemon / Lime
    "nimbu": "lemon",
    "nimboo": "lemon",
    "fresh lemon": "lemon",
    "lemon juice": "lemon juice",
    "nimbu ka ras": "lemon juice",
    "lemon zest": "lemon zest",
    "lemon rind": "lemon zest",
    "lime": "lime",
    "lime juice": "lime juice",
    "lime zest": "lime zest",
    "kaffir lime": "kaffir lime leaf",

    # Orange
    "santra": "orange",
    "narangi": "orange",
    "orange juice": "orange juice",
    "orange zest": "orange zest",
    "mandarin": "mandarin orange",
    "clementine": "orange",
    "tangerine": "orange",

    # Mango
    "aam": "mango",
    "keri": "mango",
    "raw mango": "mango",
    "green mango": "mango",
    "ripe mango": "mango",
    "frozen mango": "mango",

    # Apple
    "seb": "apple",
    "granny smith": "granny smith apple",
    "green apple": "granny smith apple",
    "red apple": "apple",

    # Banana
    "kela": "banana",
    "plantain": "banana",

    # Tropical fruit spelling variants
    "litchi": "lychee",
    "lichee": "lychee",
    "leechee": "lychee",
    "guavas": "guava",
    "jack fruit": "jackfruit",
    "jackfruits": "jackfruit",

    # Coconut
    "nariyal": "coconut",
    "grated coconut": "coconut",
    "desiccated coconut": "coconut",
    "coconut flakes": "coconut flake",
    "shredded coconut": "coconut flake",

    # Pineapple
    "ananas": "pineapple",
    "pineapple chunks": "pineapple chunk",
    "pineapple rings": "pineapple ring",
    "pineapple juice": "pineapple juice",
    "canned pineapple": "pineapple",

    # Strawberry
    "strawberries": "strawberry",

    # Blueberry
    "blueberries": "blueberry",

    # Raspberry
    "raspberries": "raspberry",

    # Raisin
    "kishmish": "raisin",
    "sultanas": "raisin",
    "golden raisins": "golden raisin",
    "currants": "currant",

    # Date
    "khajoor": "date",
    "medjool dates": "date",

    # Apricot
    "khubani": "apricot",
    "khubaani": "apricot",
    "dried apricots": "apricot",

    # Pumpkin / Squash
    "kaddu": "pumpkin",
    "sitaphal": "pumpkin",
    "butternut squash": "butternut squash",
    "acorn squash": "pumpkin",
    "kabocha": "pumpkin",

    # Cranberry
    "cranberries": "cranberry",
    "cranberry juice": "cranberry juice",

    # Peach
    "aadu": "peach",
    "canned peaches": "peach",

    # Avocado
    "makhanphal": "avocado",
    "butter fruit": "avocado",
    "avocados": "avocado",

    # ─────────────────────────────────────────────
    # NUTS & SEEDS
    # ─────────────────────────────────────────────

    # Almonds
    "badam": "almond",
    "badaam": "almond",
    "sliced almonds": "almond",
    "blanched almonds": "almond",
    "ground almonds": "almond meal",
    "almond flour": "almond flour",
    "almond paste": "almond paste",
    "almond milk": "almond milk",
    "almond extract": "almond extract",

    # Cashews
    "kaju": "cashew",
    "cashew nuts": "cashew nut",
    "raw cashews": "cashew",
    "roasted cashews": "cashew",

    # Peanuts
    "moongphali": "peanut",
    "groundnuts": "peanut",
    "peanut butter": "peanut butter",
    "groundnut paste": "peanut butter",

    # Walnuts
    "akhrot": "walnut",
    "walnut halves": "walnut half",
    "chopped walnuts": "walnut",

    # Pistachios
    "pista": "pistachio",
    "pistachios": "pistachio nut",

    # Hazelnuts
    "hazel nut": "hazelnut",
    "filbert": "hazelnut",

    # Pine Nuts
    "chilgoza": "pine nut",
    "pine nuts": "pine nut",

    # Pecans
    "pecan nuts": "pecan nut",
    "pecan halves": "pecan half",

    # Sesame Seeds
    "til": "sesame seed",
    "gingelly seeds": "sesame seed",
    "white sesame seeds": "sesame seed",
    "black sesame seeds": "black sesame seed",
    "black til": "black sesame seed",

    # Flax Seeds
    "alsi": "flax seed",
    "linseed": "flax seed",
    "flaxseed": "flax seed",
    "ground flaxseed": "flax seed",

    # Pumpkin Seeds
    "pumpkin seeds": "pumpkin seed",
    "pepitas": "pumpkin seed",

    # Sunflower Seeds
    "sunflower seeds": "sunflower seed",

    # ─────────────────────────────────────────────
    # CONDIMENTS & SAUCES
    # ─────────────────────────────────────────────

    # Soy Sauce
    "soya sauce": "soy sauce",
    "soya": "soy sauce",
    "dark soy sauce": "dark soy sauce",
    "light soy sauce": "soy sauce",
    "tamari": "tamari",
    "liquid aminos": "soy sauce",

    # Fish Sauce
    "nam pla": "fish sauce",
    "thai fish sauce": "thai fish sauce",

    # Worcestershire Sauce
    "worcester sauce": "worcestershire sauce",
    "lea perrins": "worcestershire sauce",

    # Vinegar
    "sirka": "vinegar",
    "white vinegar": "white vinegar",
    "red wine vinegar": "red wine vinegar",
    "white wine vinegar": "white wine vinegar",
    "balsamic vinegar": "balsamic vinegar",
    "apple cider vinegar": "apple cider vinegar",
    "rice vinegar": "rice vinegar",
    "malt vinegar": "malt vinegar",
    "cider vinegar": "cider vinegar",

    # Ketchup
    "tomato ketchup": "ketchup",
    "catsup": "ketchup",

    # Hot Sauce
    "tabasco": "tabasco sauce",
    "sriracha": "sriracha sauce",
    "chili sauce": "chili sauce",
    "sambal": "sambal oelek",
    "sambal oelek": "sambal oelek",
    "hot sauce": "tabasco sauce",
    "harissa": "chili paste",

    # Hoisin
    "hoisin": "hoisin sauce",

    # Oyster Sauce
    "oyster sauce": "oyster sauce",

    # Tahini
    "sesame paste": "tahini",
    "tahini paste": "tahini",
    "sesame seed paste": "tahini",

    # Mayonnaise
    "mayo": "mayonnaise",

    # Mustard (already above)

    # Pesto
    "basil pesto": "basil pesto",
    "pesto sauce": "pesto sauce",

    # Miso
    "miso paste": "miso paste",
    "white miso": "white miso",
    "red miso": "red miso",
    "yellow miso": "yellow miso",

    # Barbecue Sauce
    "bbq sauce": "barbecue sauce",

    # Chutney
    "mango chutney": "mango chutney",
    "imli chutney": "tamarind paste",
    "tamarind chutney": "tamarind paste",

    # ─────────────────────────────────────────────
    # ALCOHOLIC BEVERAGES (cooking use)
    # ─────────────────────────────────────────────
    "dry white wine": "white wine",
    "dry red wine": "red wine",
    "cooking wine": "white wine",
    "marsala": "marsala wine",
    "dry sherry": "sherry",
    "cooking sherry": "sherry",
    "mirin": "mirin",
    "rice wine": "rice wine",
    "sake": "sake",
    "dark rum": "dark rum",
    "cognac": "cognac",
    "brandy": "brandy",

    # ─────────────────────────────────────────────
    # LEAVENING & BAKING
    # ─────────────────────────────────────────────
    # NOTE: baking powder, baking soda, yeast are NOT in RecipeDB — handled by fuzzy/embed layers

    "vanilla": "vanilla",
    "vanilla extract": "vanilla extract",
    "vanilla essence": "vanilla extract",
    "vanilla bean": "vanilla bean",
    "vanilla pod": "vanilla bean",
    "cocoa powder": "cocoa powder",
    "unsweetened cocoa": "cocoa powder",
    "dutch cocoa": "dutch cocoa",
    "cocoa": "cocoa",
    "dark chocolate": "dark chocolate",
    "bittersweet chocolate": "dark chocolate",
    "semi sweet chocolate": "chocolate chip",
    "chocolate chips": "chocolate chip",
    "milk chocolate": "milk chocolate",
    "white chocolate": "white chocolate",
    "chocolate syrup": "chocolate syrup",
    "gelatin": "gelatin",
    "gelatine": "gelatin",
    "agar agar": "gelatin",

    # Breadcrumbs
    "breadcrumbs": "breadcrumb",
    "bread crumbs": "bread crumb",
    "panko": "panko breadcrumb",
    "panko breadcrumbs": "panko breadcrumb",

    # Tortillas / Wraps
    "flour tortilla": "flour tortilla",
    "corn tortilla": "corn tortilla",
    "roti": "flour tortilla",
    "chapati": "flour tortilla",
    "wrap": "flour tortilla",
    "pita": "pita bread",
    "pitta": "pita bread",
    "naan": "bread",

    # ─────────────────────────────────────────────
    # SALT VARIANTS
    # ─────────────────────────────────────────────
    "namak": "salt",
    "sendha namak": "sea salt",
    "rock salt": "sea salt",
    "pink salt": "sea salt",
    "himalayan salt": "sea salt",
    "sea salt": "sea salt",
    "kosher salt": "kosher salt",
    "table salt": "salt",
    "fine salt": "fine sea salt",
    "iodised salt": "salt",
    "kala namak": "black salt",
    "black salt": "black salt",

    # ─────────────────────────────────────────────
    # WATER / LIQUID
    # ─────────────────────────────────────────────
    "pani": "water",
    "cold water": "cold water",
    "hot water": "hot water",
    "warm water": "water",
    "lukewarm water": "water",
    "boiling water": "hot water",
    "sparkling water": "sparkling water",
    "soda water": "soda water",

    # ─────────────────────────────────────────────
    # MISC ADDITIVES
    # ─────────────────────────────────────────────
    "cooking spray": "cooking spray",
    "non stick spray": "cooking spray",
    "pam spray": "cooking spray",
    "shortening": "shortening",
    "vegetable shortening": "shortening",
    "lard": "shortening",
    "margarine": "margarine",
}


# ─────────────────────────────────────────────────────────
# CATEGORY-LEVEL SYNONYM GROUPS
# Use these to map free-text category references to the
# Dietrx_Category labels in RecipeDB.
# ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _known_canonical_terms() -> set[str]:
    terms = set(SYNONYMS.keys()) | set(SYNONYMS.values())
    try:
        from module1_asr.data.db_loader import DB_INGREDIENTS

        terms.update(DB_INGREDIENTS.keys())
    except Exception:
        pass
    return {str(term).lower().strip() for term in terms if str(term).strip()}


def _drop_extra_trailing_e(key: str) -> str | None:
    candidate = key[:-1]
    if len(key) > 4 and key.endswith("e") and candidate in _known_canonical_terms():
        return candidate
    return None


def canonicalize_ingredient(name: str) -> str:
    """Return the configured canonical ingredient name."""
    key = str(name).lower().strip()
    canonical = SYNONYMS.get(key)
    if canonical:
        return canonical
    spelling_fix = _drop_extra_trailing_e(key)
    if spelling_fix:
        return SYNONYMS.get(spelling_fix, spelling_fix)
    return key


CATEGORY_SYNONYMS = {
    "veggie": "Vegetable",
    "vegetables": "Vegetable",
    "sabzi": "Vegetable",
    "meat": "Meat",
    "non veg": "Meat",
    "nonveg": "Meat",
    "gosht": "Meat",
    "poultry": "Meat",
    "fish": "Fish",
    "seafood": "Seafood",
    "dairy": "Dairy",
    "milk products": "Dairy",
    "doodh products": "Dairy",
    "spice": "Spice",
    "masala": "Spice",
    "herb": "Herb",
    "jadi booti": "Herb",
    "legume": "Legume",
    "pulse": "Legume",
    "dal": "Legume",
    "daal": "Legume",
    "grain": "Cereal",
    "cereal": "Cereal",
    "anaaj": "Cereal",
    "nut": "Nuts and Seeds",
    "nuts": "Nuts and Seeds",
    "seed": "Nuts and Seeds",
    "seeds": "Nuts and Seeds",
    "dry fruit": "Nuts and Seeds",
    "dry fruits": "Nuts and Seeds",
    "mewa": "Nuts and Seeds",
    "fruit": "Fruit",
    "phal": "Fruit",
    "fungi": "Fungi",
    "mushroom": "Fungi",
    "bakery": "Bakery",
    "bread": "Bakery",
    "condiment": "Condiment",
    "sauce": "Condiment",
    "dip": "Condiment",
    "oil": "Plant Derivative",
    "fat": "Plant Derivative",
    "sweetener": "Additive-Sugar",
    "sugar": "Additive-Sugar",
    "vinegar": "Additive-Vinegar",
    "sirka": "Additive-Vinegar",
    "alcohol": "Beverage-Alcoholic",
    "wine": "Beverage-Alcoholic",
    "drink": "Beverage",
    "juice": "Beverage",
    "gourd": "Gourd",
    "kaddu": "Gourd",
    "flower": "Flower",
    "tuber": "Vegetable Tuber",
    "root vegetable": "Vegetable Tuber",
}
