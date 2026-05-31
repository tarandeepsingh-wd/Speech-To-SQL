"""
TADKA Module 1 — Comprehensive Test Suite (~1200+ cases)
Run from pipeline/ directory:
    python test_module1_full.py
    python test_module1_full.py --fast
    python test_module1_full.py --category 5
    python test_module1_full.py --verbose
"""
import sys, time, argparse, re
sys.path.insert(0, ".")
from module1_asr.entity_tagger import tag_entities

# FORMAT: (query, must_ingredients, must_dietary, must_utensils, must_empty, must_nutrition_nutrients)
# must_empty=True means ingredients MUST be []

# ══════════════════════════════════════════════════════════
# CAT 1 — SINGLE ENGLISH INGREDIENT (50)
# ══════════════════════════════════════════════════════════
CAT1 = [
    ("recipes with chicken", ["chicken"], [], [], False, []),
    ("recipes with garlic", ["garlic"], [], [], False, []),
    ("recipes with onion", ["onion"], [], [], False, []),
    ("recipes with tomato", ["tomato"], [], [], False, []),
    ("recipes with spinach", ["spinach"], [], [], False, []),
    ("recipes with salt", ["salt"], [], [], False, []),
    ("recipes with butter", ["butter"], [], [], False, []),
    ("recipes with sugar", ["sugar"], [], [], False, []),
    ("recipes with milk", ["milk"], [], [], False, []),
    ("recipes with egg", ["egg"], [], [], False, []),
    ("recipes with carrot", ["carrot"], [], [], False, []),
    ("recipes with potato", ["potato"], [], [], False, []),
    ("recipes with ginger", ["ginger"], [], [], False, []),
    ("recipes with turmeric", ["turmeric"], [], [], False, []),
    ("recipes with cumin", ["cumin"], [], [], False, []),
    ("recipes with coriander", ["coriander"], [], [], False, []),
    ("recipes with cinnamon", ["cinnamon"], [], [], False, []),
    ("recipes with pepper", ["pepper"], [], [], False, []),
    ("recipes with flour", ["flour"], [], [], False, []),
    ("recipes with rice", ["rice"], [], [], False, []),
    ("recipes with lemon", ["lemon"], [], [], False, []),
    ("recipes with cream", ["cream"], [], [], False, []),
    ("recipes with yogurt", ["yogurt"], [], [], False, []),
    ("recipes with cheese", ["cheese"], [], [], False, []),
    ("recipes with mushroom", ["mushroom"], [], [], False, []),
    ("recipes with salmon", ["salmon"], [], [], False, []),
    ("recipes with tuna", ["tuna"], [], [], False, []),
    ("recipes with beef", ["beef"], [], [], False, []),
    ("recipes with lamb", ["lamb"], [], [], False, []),
    ("recipes with shrimp", ["shrimp"], [], [], False, []),
    ("recipes with basil", ["basil"], [], [], False, []),
    ("recipes with thyme", ["thyme"], [], [], False, []),
    ("recipes with oregano", ["oregano"], [], [], False, []),
    ("recipes with paprika", ["paprika"], [], [], False, []),
    ("recipes with saffron", ["saffron"], [], [], False, []),
    ("recipes with vanilla", ["vanilla"], [], [], False, []),
    ("recipes with honey", ["honey"], [], [], False, []),
    ("recipes with vinegar", ["vinegar"], [], [], False, []),
    ("recipes with pasta", ["pasta"], [], [], False, []),
    ("recipes with paneer", ["paneer"], [], [], False, []),
    ("give me chicken recipes", ["chicken"], [], [], False, []),
    ("show me garlic dishes", ["garlic"], [], [], False, []),
    ("find tomato recipes", ["tomato"], [], [], False, []),
    ("I want spinach recipes", ["spinach"], [], [], False, []),
    ("show recipes that use salmon", ["salmon"], [], [], False, []),
    ("list all recipes with mushroom", ["mushroom"], [], [], False, []),
    ("what can I make with egg", ["egg"], [], [], False, []),
    ("dishes that contain beef", ["beef"], [], [], False, []),
    ("all recipes which have rice", ["rice"], [], [], False, []),
    ("give me those recipes which has spinach", ["spinach"], [], [], False, []),
]

# ══════════════════════════════════════════════════════════
# CAT 2 — MULTI ENGLISH INGREDIENT (60)
# ══════════════════════════════════════════════════════════
CAT2 = [
    ("chicken and garlic recipes", ["chicken", "garlic"], [], [], False, []),
    ("recipes with tomato and onion", ["tomato", "onion"], [], [], False, []),
    ("spinach and egg dishes", ["spinach", "egg"], [], [], False, []),
    ("beef with garlic and onion", ["beef", "garlic", "onion"], [], [], False, []),
    ("salmon with lemon and butter", ["salmon", "lemon", "butter"], [], [], False, []),
    ("pasta with cheese and basil", ["pasta", "cheese", "basil"], [], [], False, []),
    ("chicken with ginger and garlic", ["chicken", "ginger", "garlic"], [], [], False, []),
    ("rice with cumin and coriander", ["rice", "cumin", "coriander"], [], [], False, []),
    ("carrot and potato soup", ["carrot", "potato"], [], [], False, []),
    ("mushroom and cream sauce", ["mushroom", "cream"], [], [], False, []),
    ("tuna with lemon and capers", ["tuna", "lemon"], [], [], False, []),
    ("lamb with garlic and rosemary", ["lamb", "garlic"], [], [], False, []),
    ("eggs with butter and salt", ["egg", "butter", "salt"], [], [], False, []),
    ("tomato onion and pepper stir fry", ["tomato", "onion", "pepper"], [], [], False, []),
    ("chicken with olive oil and thyme", ["chicken", "olive oil", "thyme"], [], [], False, []),
    ("pasta with tomato garlic and basil", ["pasta", "tomato", "garlic", "basil"], [], [], False, []),
    ("rice with spinach mushroom and cream", ["rice", "spinach", "mushroom", "cream"], [], [], False, []),
    ("beef stew with potato carrot and onion", ["beef", "potato", "carrot", "onion"], [], [], False, []),
    ("salmon with lemon butter and capers", ["salmon", "lemon", "butter"], [], [], False, []),
    ("chicken with garlic ginger and soy sauce", ["chicken", "garlic", "ginger", "soy sauce"], [], [], False, []),
    ("show recipes with garlic and tomato", ["garlic", "tomato"], [], [], False, []),
    ("find dishes with chicken and rice", ["chicken", "rice"], [], [], False, []),
    ("give me recipes that have spinach and cheese", ["spinach", "cheese"], [], [], False, []),
    ("I need recipes with egg and butter", ["egg", "butter"], [], [], False, []),
    ("what recipes use lemon and honey", ["lemon", "honey"], [], [], False, []),
    ("dishes that contain beef and mushroom", ["beef", "mushroom"], [], [], False, []),
    ("recipes which have carrot and ginger", ["carrot", "ginger"], [], [], False, []),
    ("pasta with basil and tomato", ["pasta", "basil", "tomato"], [], [], False, []),
    ("recipes with cumin coriander and turmeric", ["cumin", "coriander", "turmeric"], [], [], False, []),
    ("all dishes with salmon and lemon", ["salmon", "lemon"], [], [], False, []),
    ("chicken with salt pepper and olive oil", ["chicken", "salt", "pepper", "olive oil"], [], [], False, []),
    ("rice with egg and soy sauce", ["rice", "egg", "soy sauce"], [], [], False, []),
    ("mushroom soup with cream and thyme", ["mushroom", "cream", "thyme"], [], [], False, []),
    ("beef burger with onion and cheese", ["beef", "onion", "cheese"], [], [], False, []),
    ("shrimp with garlic butter and lemon", ["shrimp", "garlic", "butter", "lemon"], [], [], False, []),
    ("pasta with olive oil garlic and parmesan", ["pasta", "olive oil", "garlic"], [], [], False, []),
    ("chicken soup with carrot and onion", ["chicken", "carrot", "onion"], [], [], False, []),
    ("lamb with mint and garlic", ["lamb", "garlic"], [], [], False, []),
    ("salad with tomato and olive oil", ["tomato", "olive oil"], [], [], False, []),
    ("stir fry with ginger and soy sauce", ["ginger", "soy sauce"], [], [], False, []),
    ("omelette with cheese mushroom and spinach", ["cheese", "mushroom", "spinach"], [], [], False, []),
    ("cake with flour sugar butter and egg", ["flour", "sugar", "butter", "egg"], [], [], False, []),
    ("curry with chicken and ginger", ["chicken", "ginger"], [], [], False, []),
    ("pizza with tomato cheese and basil", ["tomato", "cheese", "basil"], [], [], False, []),
    ("fried rice with egg onion and soy sauce", ["rice", "egg", "onion", "soy sauce"], [], [], False, []),
    ("salad dressing with lemon olive oil and garlic", ["lemon", "olive oil", "garlic"], [], [], False, []),
    ("chicken tikka with yogurt ginger and garlic", ["chicken", "yogurt", "ginger", "garlic"], [], [], False, []),
    ("vegetable soup with potato and carrot", ["potato", "carrot"], [], [], False, []),
    ("chocolate cake with butter sugar flour and egg", ["butter", "sugar", "flour", "egg"], [], [], False, []),
    ("shrimp pasta with garlic and lemon", ["shrimp", "pasta", "garlic", "lemon"], [], [], False, []),
    ("chicken breast with garlic and herb", ["chicken", "garlic"], [], [], False, []),
    ("beef and broccoli with soy sauce", ["beef", "soy sauce"], [], [], False, []),
    ("lentil soup with cumin and coriander", ["cumin", "coriander"], [], [], False, []),
    ("banana bread with flour sugar and egg", ["flour", "sugar", "egg"], [], [], False, []),
    ("avocado toast with lemon and salt", ["lemon", "salt"], [], [], False, []),
    ("tomato soup with cream and basil", ["tomato", "cream", "basil"], [], [], False, []),
    ("pasta carbonara with egg and cheese", ["pasta", "egg", "cheese"], [], [], False, []),
    ("chicken Caesar salad with garlic and lemon", ["chicken", "garlic", "lemon"], [], [], False, []),
    ("beef tacos with onion tomato and cheese", ["beef", "onion", "tomato", "cheese"], [], [], False, []),
    ("shrimp scampi with garlic butter and lemon", ["shrimp", "garlic", "butter", "lemon"], [], [], False, []),
]

# ══════════════════════════════════════════════════════════
# CAT 3 — SINGLE HINDI INGREDIENT (60)
# ══════════════════════════════════════════════════════════
CAT3 = [
    ("adrak wali recipe", ["ginger"], [], [], False, []),
    ("haldi ki recipe", ["turmeric"], [], [], False, []),
    ("jeera wali dish", ["cumin"], [], [], False, []),
    ("lehsun recipe", ["garlic"], [], [], False, []),
    ("lahsun recipe", ["garlic"], [], [], False, []),
    ("lehsoon recipe", ["garlic"], [], [], False, []),
    ("pyaz wali sabzi", ["onion"], [], [], False, []),
    ("tamatar ki curry", ["tomato"], [], [], False, []),
    ("namak wali recipe", ["salt"], [], [], False, []),
    ("dahi wala dish", ["yogurt"], [], [], False, []),
    ("ghee wali roti", ["ghee"], [], [], False, []),
    ("hari mirch recipe", ["green chilies"], [], [], False, []),
    ("lal mirch wali sabzi", ["red chili pepper"], [], [], False, []),
    ("dhania wali chutney", ["coriander"], [], [], False, []),
    ("aloo ki recipe", ["potato"], [], [], False, []),
    ("gajar halwa", ["carrot"], [], [], False, []),
    ("palak recipe", ["spinach"], [], [], False, []),
    ("paneer recipe", ["paneer"], [], [], False, []),
    ("chawal ki recipe", ["rice"], [], [], False, []),
    ("makhan wali roti", ["butter"], [], [], False, []),
    ("cheeni wali mithai", ["sugar"], [], [], False, []),
    ("maida ki roti", ["flour"], [], [], False, []),
    ("besan ka cheela", ["chickpea flour"], [], [], False, []),
    ("methi wali sabzi", ["fenugreek"], [], [], False, []),
    ("hing wala dal", ["asafoetida"], [], [], False, []),
    ("elaichi wali chai", ["cardamom"], [], [], False, []),
    ("dalchini wala khana", ["cinnamon"], [], [], False, []),
    ("laung wala masala", ["clove"], [], [], False, []),
    ("kali mirch wali gravy", ["black pepper"], [], [], False, []),
    ("sarson wala saag", ["mustard"], [], [], False, []),
    ("doodh wali recipe", ["milk"], [], [], False, []),
    ("rajma recipe", ["kidney bean"], [], [], False, []),
    ("chana masala", ["chickpea"], [], [], False, []),
    ("moong dal recipe", ["mung bean"], [], [], False, []),
    ("nimbu wala pani", ["lemon"], [], [], False, []),
    ("murgh recipe", ["chicken"], [], [], False, []),
    ("machli wala khana", ["fish"], [], [], False, []),
    ("gosht ki recipe", ["meat"], [], [], False, []),
    ("atta ki roti", ["whole wheat flour"], [], [], False, []),
    ("basmati chawal", ["basmati rice"], [], [], False, []),
    ("mujhe adrak ki recipe chahiye", ["ginger"], [], [], False, []),
    ("haldi dalo isme", ["turmeric"], [], [], False, []),
    ("pyaz ke saath kya banaye", ["onion"], [], [], False, []),
    ("lehsun se kya bane", ["garlic"], [], [], False, []),
    ("aloo banao", ["potato"], [], [], False, []),
    ("gajar ka halwa banana hai", ["carrot"], [], [], False, []),
    ("palak recipe batao", ["spinach"], [], [], False, []),
    ("dahi wali dish chahiye", ["yogurt"], [], [], False, []),
    ("jeera rice kaise banate hain", ["cumin", "rice"], [], [], False, []),
    ("methi ka paratha", ["fenugreek"], [], [], False, []),
    ("mujhe zeera recipe chahiye", ["cumin"], [], [], False, []),
    ("zeera wala chawal", ["cumin", "rice"], [], [], False, []),
    ("tamater ki sabzi", ["tomato"], [], [], False, []),
    ("pyaaz wali gravy", ["onion"], [], [], False, []),
    ("dhaniya chatni", ["coriander"], [], [], False, []),
    ("kanda wali recipe", ["onion"], [], [], False, []),
    ("lasan wali dal", ["garlic"], [], [], False, []),
    ("adrak lehsun paste recipe", ["ginger", "garlic"], [], [], False, []),
    ("sarso wala saag", ["mustard"], [], [], False, []),
    ("gud wala khana", ["jaggery"], [], [], False, []),
]

# ══════════════════════════════════════════════════════════
# CAT 4 — MULTI HINDI INGREDIENT (50)
# ══════════════════════════════════════════════════════════
CAT4 = [
    ("adrak aur lehsun wali recipe", ["ginger", "garlic"], [], [], False, []),
    ("haldi aur jeera dono chahiye", ["turmeric", "cumin"], [], [], False, []),
    ("pyaz tamatar ki gravy", ["onion", "tomato"], [], [], False, []),
    ("aloo aur gajar ki sabzi", ["potato", "carrot"], [], [], False, []),
    ("palak paneer mein haldi aur namak", ["spinach", "paneer", "turmeric", "salt"], [], [], False, []),
    ("murgh aur pyaz ki curry", ["chicken", "onion"], [], [], False, []),
    ("dahi aur lehsun chutney", ["yogurt", "garlic"], [], [], False, []),
    ("aloo tamatar aur pyaz ki sabzi", ["potato", "tomato", "onion"], [], [], False, []),
    ("chawal jeera aur ghee se", ["rice", "cumin", "ghee"], [], [], False, []),
    ("makhan namak aur atta ki roti", ["butter", "salt", "whole wheat flour"], [], [], False, []),
    ("methi aur palak ka saag", ["fenugreek", "spinach"], [], [], False, []),
    ("hari mirch aur dhania ki chutney", ["green chilies", "coriander"], [], [], False, []),
    ("elaichi dalchini aur laung ka masala", ["cardamom", "cinnamon", "clove"], [], [], False, []),
    ("rajma chawal aur pyaz", ["kidney bean", "rice", "onion"], [], [], False, []),
    ("chana aur aloo ki sabzi", ["chickpea", "potato"], [], [], False, []),
    ("mujhe adrak lehsun aur pyaz chahiye", ["ginger", "garlic", "onion"], [], [], False, []),
    ("tamatar pyaz aur hari mirch ki gravy", ["tomato", "onion", "green chilies"], [], [], False, []),
    ("dal mein hing jeera aur haldi", ["asafoetida", "cumin", "turmeric"], [], [], False, []),
    ("gajar palak aur aloo ki sabzi", ["carrot", "spinach", "potato"], [], [], False, []),
    ("murgh adrak lehsun aur dahi ke saath", ["chicken", "ginger", "garlic", "yogurt"], [], [], False, []),
    ("pyaz lehsun adrak ki chutney", ["onion", "garlic", "ginger"], [], [], False, []),
    ("aloo gobi matar recipe", ["potato"], [], [], False, []),
    ("dal tadka with jeera aur hing", ["cumin", "asafoetida"], [], [], False, []),
    ("matar paneer with pyaz aur tamatar", ["paneer", "onion", "tomato"], [], [], False, []),
    ("biryani with chawal aur saffron", ["rice", "saffron"], [], [], False, []),
    ("kheer with doodh chawal aur elaichi", ["milk", "rice", "cardamom"], [], [], False, []),
    ("halwa with gajar cheeni aur ghee", ["carrot", "sugar", "ghee"], [], [], False, []),
    ("paratha with aloo aur makhan", ["potato", "butter"], [], [], False, []),
    ("chaat with aloo dhania aur nimbu", ["potato", "coriander", "lemon"], [], [], False, []),
    ("saag with palak methi aur makhan", ["spinach", "fenugreek", "butter"], [], [], False, []),
    ("murgh tikka with dahi nimbu aur adrak", ["chicken", "yogurt", "lemon", "ginger"], [], [], False, []),
    ("dal with haldi namak aur ghee", ["turmeric", "salt", "ghee"], [], [], False, []),
    ("sabzi with gajar aloo aur pyaz", ["carrot", "potato", "onion"], [], [], False, []),
    ("puri with maida aur ghee", ["flour", "ghee"], [], [], False, []),
    ("chutney with dhania nimbu aur hari mirch", ["coriander", "lemon", "green chilies"], [], [], False, []),
    ("raita with dahi jeera aur dhania", ["yogurt", "cumin", "coriander"], [], [], False, []),
    ("kadhi with dahi aur besan", ["yogurt", "chickpea flour"], [], [], False, []),
    ("anda curry with pyaz aur tamatar", ["egg", "onion", "tomato"], [], [], False, []),
    ("machli fry with adrak lehsun aur nimbu", ["ginger", "garlic", "lemon"], [], [], False, []),
    ("gosht curry with pyaz tamatar adrak lehsun", ["onion", "tomato", "ginger", "garlic"], [], [], False, []),
    ("moong dal with haldi aur hing", ["mung bean", "turmeric", "asafoetida"], [], [], False, []),
    ("rajma with tamatar pyaz aur masala", ["kidney bean", "tomato", "onion"], [], [], False, []),
    ("pulao with chawal jeera aur ghee", ["rice", "cumin", "ghee"], [], [], False, []),
    ("namkeen with besan aur namak", ["chickpea flour", "salt"], [], [], False, []),
    ("mithai with cheeni doodh aur elaichi", ["sugar", "milk", "cardamom"], [], [], False, []),
    ("biryani mein saffron dahi aur adrak", ["saffron", "yogurt", "ginger"], [], [], False, []),
    ("tadka dal with hing jeera namak", ["asafoetida", "cumin", "salt"], [], [], False, []),
    ("mutter paneer with cream aur tamatar", ["paneer", "cream", "tomato"], [], [], False, []),
    ("aloo tikki with dhania aur dahi", ["potato", "coriander", "yogurt"], [], [], False, []),
    ("chicken korma with dahi cream aur adrak", ["chicken", "yogurt", "cream", "ginger"], [], [], False, []),
]

# ══════════════════════════════════════════════════════════
# CAT 5 — MIXED HINDI-ENGLISH (60)
# ══════════════════════════════════════════════════════════
CAT5 = [
    ("chicken with adrak and lehsun", ["chicken", "ginger", "garlic"], [], [], False, []),
    ("pasta with tamatar and pyaz", ["pasta", "tomato", "onion"], [], [], False, []),
    ("rice with jeera and ghee", ["rice", "cumin", "ghee"], [], [], False, []),
    ("chicken curry with adrak lehsun aur lal mirch", ["chicken", "ginger", "garlic", "red chili pepper"], [], [], False, []),
    ("salmon with nimbu and butter", ["salmon", "lemon", "butter"], [], [], False, []),
    ("egg with namak and kali mirch", ["egg", "salt", "black pepper"], [], [], False, []),
    ("soup with gajar aloo and celery", ["carrot", "potato"], [], [], False, []),
    ("give me aloo and chicken recipe", ["potato", "chicken"], [], [], False, []),
    ("mujhe chicken aur palak recipe chahiye", ["chicken", "spinach"], [], [], False, []),
    ("I want recipe with adrak and garlic", ["ginger", "garlic"], [], [], False, []),
    ("pasta mein tamatar aur cheese dalo", ["pasta", "tomato", "cheese"], [], [], False, []),
    ("beef with pyaz and tomato", ["beef", "onion", "tomato"], [], [], False, []),
    ("I want spinach aur paneer recipe", ["spinach", "paneer"], [], [], False, []),
    ("chicken soup with gajar aur onion", ["chicken", "carrot", "onion"], [], [], False, []),
    ("give me rice with dahi and jeera", ["rice", "yogurt", "cumin"], [], [], False, []),
    ("recipes with lehsun aur ginger", ["garlic", "ginger"], [], [], False, []),
    ("chicken ke saath lemon and butter", ["chicken", "lemon", "butter"], [], [], False, []),
    ("dishes with gajar and carrot", ["carrot"], [], [], False, []),
    ("biryani with basmati aur saffron", ["biryani", "basmati rice", "saffron"], [], [], False, []),
    ("egg curry with pyaz tomato and hari mirch", ["egg", "onion", "tomato", "green chilies"], [], [], False, []),
    ("aloo tikki with dhania chutney", ["potato", "coriander"], [], [], False, []),
    ("chicken tikka with dahi and nimbu", ["chicken", "yogurt", "lemon"], [], [], False, []),
    ("pasta with olive oil lehsun and basil", ["pasta", "olive oil", "garlic", "basil"], [], [], False, []),
    ("beef curry adrak lehsun aur tamatar", ["beef", "ginger", "garlic", "tomato"], [], [], False, []),
    ("shrimp with lehsun butter aur nimbu", ["shrimp", "garlic", "butter", "lemon"], [], [], False, []),
    ("pulao with basmati jeera aur ghee", ["basmati rice", "cumin", "ghee"], [], [], False, []),
    ("chicken biryani with adrak lehsun and saffron", ["chicken", "ginger", "garlic", "saffron"], [], [], False, []),
    ("raita with dahi cucumber aur jeera", ["yogurt", "cumin"], [], [], False, []),
    ("paratha with aloo and butter", ["potato", "butter"], [], [], False, []),
    ("kheer with doodh chawal aur elaichi", ["milk", "rice", "cardamom"], [], [], False, []),
    ("puri with maida and ghee", ["flour", "ghee"], [], [], False, []),
    ("curry with pyaz tamatar lehsun aur adrak", ["onion", "tomato", "garlic", "ginger"], [], [], False, []),
    ("tikka masala with chicken dahi and tomato", ["chicken", "yogurt", "tomato"], [], [], False, []),
    ("saag with palak methi aur makhan", ["spinach", "fenugreek", "butter"], [], [], False, []),
    ("chana dal with haldi jeera aur hing", ["turmeric", "cumin", "asafoetida"], [], [], False, []),
    ("matar paneer with tomato aur cream", ["paneer", "tomato", "cream"], [], [], False, []),
    ("dal makhani with butter and cream", ["butter", "cream"], [], [], False, []),
    ("korma with cream yogurt and adrak", ["cream", "yogurt", "ginger"], [], [], False, []),
    ("mutton with adrak and kali mirch", ["ginger", "black pepper"], [], [], False, []),
    ("banana cake with cheeni makhan aur egg", ["sugar", "butter", "egg"], [], [], False, []),
    ("soup with pyaz gajar aur chicken", ["onion", "carrot", "chicken"], [], [], False, []),
    ("dosa with rice aur besan", ["rice", "chickpea flour"], [], [], False, []),
    ("palak paneer with cream aur garlic", ["spinach", "paneer", "cream", "garlic"], [], [], False, []),
    ("halwa with gajar sugar aur ghee", ["carrot", "sugar", "ghee"], [], [], False, []),
    ("chaat with aloo dhania aur nimbu", ["potato", "coriander", "lemon"], [], [], False, []),
    ("fried rice with egg pyaz aur soy sauce", ["rice", "egg", "onion", "soy sauce"], [], [], False, []),
    ("lassi with dahi aur sugar", ["yogurt", "sugar"], [], [], False, []),
    ("gajar aur carrot soup", ["carrot"], [], [], False, []),
    ("spinach aur palak smoothie with lemon", ["spinach", "lemon"], [], [], False, []),
    ("chicken sandwich with mustard and onion", ["chicken", "mustard", "onion"], [], [], False, []),
    ("paneer bhurji with pyaz tamatar aur adrak", ["paneer", "onion", "tomato", "ginger"], [], [], False, []),
    ("masala chai with elaichi aur adrak", ["cardamom", "ginger"], [], [], False, []),
    ("gulab jamun with maida cheeni aur ghee", ["flour", "sugar", "ghee"], [], [], False, []),
    ("dum aloo with dahi aur spices", ["potato", "yogurt"], [], [], False, []),
    ("sarson ka saag with makhan", ["mustard", "butter"], [], [], False, []),
    ("kadhi pakora with besan and dahi", ["chickpea flour", "yogurt"], [], [], False, []),
    ("samosa with aloo and hari mirch", ["potato", "green chilies"], [], [], False, []),
    ("naan with maida aur dahi", ["flour", "yogurt"], [], [], False, []),
    ("lasagna with pasta beef aur tomato", ["pasta", "beef", "tomato"], [], [], False, []),
    ("risotto with mushroom and cream", ["mushroom", "cream"], [], [], False, []),
]

# ══════════════════════════════════════════════════════════
# CAT 6 — DIETARY STYLES (60)
# ══════════════════════════════════════════════════════════
CAT6 = [
    ("vegan recipes with spinach", ["spinach"], ["vegan"], [], False, []),
    ("vegetarian dishes with cheese", ["cheese"], ["vegetarian"], [], False, []),
    ("gluten-free recipes with rice", ["rice"], ["gluten-free"], [], False, []),
    ("dairy-free dishes with chicken", ["chicken"], ["dairy-free"], [], False, []),
    ("keto recipes with eggs", ["egg"], ["keto"], [], False, []),
    ("paleo recipes with beef", ["beef"], ["paleo"], [], False, []),
    ("halal chicken recipes", ["chicken"], ["halal"], [], False, []),
    ("kosher beef dishes", ["beef"], ["kosher"], [], False, []),
    ("diabetic-friendly recipes with chicken", ["chicken"], ["diabetic"], [], False, []),
    ("pescatarian recipes with salmon", ["salmon"], ["pescatarian"], [], False, []),
    ("vegan and gluten-free pasta", ["pasta"], ["vegan", "gluten-free"], [], False, []),
    ("vegetarian recipes with garlic and tomato", ["garlic", "tomato"], ["vegetarian"], [], False, []),
    ("non-vegetarian dishes with chicken", ["chicken"], ["non-vegetarian"], [], False, []),
    ("non-veg recipe with beef", ["beef"], ["non-vegetarian"], [], False, []),
    ("gluten free recipes with chicken", ["chicken"], ["gluten-free"], [], False, []),
    ("dairy free dishes with spinach and lemon", ["spinach", "lemon"], ["dairy-free"], [], False, []),
    ("vegan food with mushroom", ["mushroom"], ["vegan"], [], False, []),
    ("ketogenic recipes with eggs and cheese", ["egg", "cheese"], ["keto"], [], False, []),
    ("vegetarian Indian recipes with paneer", ["paneer"], ["vegetarian"], [], False, []),
    ("vegan recipes with aloo and gajar", ["potato", "carrot"], ["vegan"], [], False, []),
    ("mujhe vegan recipe chahiye with spinach", ["spinach"], ["vegan"], [], False, []),
    ("give me gluten-free chicken dishes", ["chicken"], ["gluten-free"], [], False, []),
    ("I want vegetarian recipes with garlic", ["garlic"], ["vegetarian"], [], False, []),
    ("show non-veg recipes with beef", ["beef"], ["non-vegetarian"], [], False, []),
    ("keto food with salmon and butter", ["salmon", "butter"], ["keto"], [], False, []),
    ("low sodium recipes with chicken", ["chicken"], ["low-sodium"], [], False, []),
    ("sugar-free dessert with vanilla", ["vanilla"], ["sugar-free"], [], False, []),
    ("high protein meals with chicken and egg", ["chicken", "egg"], ["high-protein"], [], False, []),
    ("low fat recipes with spinach", ["spinach"], ["low-fat"], [], False, []),
    ("low calorie dishes with vegetable", [], ["low-calorie"], [], False, []),
    ("halal beef stew", ["beef"], ["halal"], [], False, []),
    ("kosher chicken soup", ["chicken"], ["kosher"], [], False, []),
    ("paleo diet recipes with beef and egg", ["beef", "egg"], ["paleo"], [], False, []),
    ("nonveg food with lamb and garlic", ["lamb", "garlic"], ["non-vegetarian"], [], False, []),
    ("vegetarian pasta with tomato and basil", ["pasta", "tomato", "basil"], ["vegetarian"], [], False, []),
    ("gluten free pizza with cheese and tomato", ["cheese", "tomato"], ["gluten-free"], [], False, []),
    ("vegan curry with potato", ["potato"], ["vegan"], [], False, []),
    ("diabetic friendly recipes with honey", ["honey"], ["diabetic"], [], False, []),
    ("pescatarian diet with shrimp and lemon", ["shrimp", "lemon"], ["pescatarian"], [], False, []),
    ("lacto-vegetarian dishes with milk and cheese", ["milk", "cheese"], ["lacto-vegetarian"], [], False, []),
    ("ovo-vegetarian recipes with egg", ["egg"], ["ovo-vegetarian"], [], False, []),
    ("vegan dessert with sugar and vanilla", ["sugar", "vanilla"], ["vegan"], [], False, []),
    ("ketogenic breakfast with eggs", ["egg"], ["keto"], [], False, []),
    ("low-calorie soup with carrot and celery", ["carrot"], ["low-calorie"], [], False, []),
    ("low-fat chicken breast recipe", ["chicken"], ["low-fat"], [], False, []),
    ("gluten free bread with rice flour", ["rice"], ["gluten-free"], [], False, []),
    ("vegan smoothie with spinach and lemon", ["spinach", "lemon"], ["vegan"], [], False, []),
    ("vegetarian biryani with saffron and paneer", ["saffron", "paneer"], ["vegetarian"], [], False, []),
    ("dairy-free curry with potato and spinach", ["potato", "spinach"], ["dairy-free"], [], False, []),
    ("nonvegetarian recipes with mutton", [], ["non-vegetarian"], [], False, []),
    ("keto snack with cheese and egg", ["cheese", "egg"], ["keto"], [], False, []),
    ("gluten-free cookies with almond flour", [], ["gluten-free"], [], False, []),
    ("vegan protein bowl with spinach", ["spinach"], ["vegan", "high-protein"], [], False, []),
    ("halal lamb chops with garlic", ["lamb", "garlic"], ["halal"], [], False, []),
    ("kosher salmon with lemon and herbs", ["salmon", "lemon"], ["kosher"], [], False, []),
    ("low-sodium soup with chicken and carrot", ["chicken", "carrot"], ["low-sodium"], [], False, []),
    ("sugar free cake recipe", [], ["sugar-free"], [], False, []),
    ("high protein egg white omelette", ["egg"], ["high-protein"], [], False, []),
    ("paleo breakfast with egg and bacon", ["egg"], ["paleo"], [], False, []),
    ("dairy free ice cream with vanilla", ["vanilla"], ["dairy-free"], [], False, []),
]

# ══════════════════════════════════════════════════════════
# CAT 7 — UTENSIL QUERIES (40)
# ══════════════════════════════════════════════════════════
CAT7 = [
    ("recipes that use a skillet", [], [], ["skillet"], False, []),
    ("dishes made in a wok", [], [], ["wok"], False, []),
    ("recipes using a blender", [], [], ["blender"], False, []),
    ("food cooked in an oven", [], [], ["oven"], False, []),
    ("recipes that need a pan", [], [], ["pan"], False, []),
    ("dishes using a pot", [], [], ["pot"], False, []),
    ("recipes that require a saucepan", [], [], ["saucepan"], False, []),
    ("cooking with a spatula", [], [], ["spatula"], False, []),
    ("recipes using a mixer", [], [], ["mixer"], False, []),
    ("dishes that need a bowl", [], [], ["bowl"], False, []),
    ("skillet chicken recipe", ["chicken"], [], ["skillet"], False, []),
    ("wok fried rice with egg", ["rice", "egg"], [], ["wok"], False, []),
    ("oven baked salmon with lemon", ["salmon", "lemon"], [], ["oven"], False, []),
    ("pan seared beef with garlic", ["beef", "garlic"], [], ["pan"], False, []),
    ("pot roast with carrot and potato", ["carrot", "potato"], [], ["pot"], False, []),
    ("saucepan soup with onion and tomato", ["onion", "tomato"], [], ["saucepan"], False, []),
    ("recipes cooked in a pan with garlic", ["garlic"], [], ["pan"], False, []),
    ("oven roasted vegetables with garlic", ["garlic"], [], ["oven"], False, []),
    ("wok stir fry with chicken and ginger", ["chicken", "ginger"], [], ["wok"], False, []),
    ("skillet eggs with spinach", ["egg", "spinach"], [], ["skillet"], False, []),
    ("mixer cake with butter and sugar", ["butter", "sugar"], [], ["mixer"], False, []),
    ("saucepan pasta with tomato", ["pasta", "tomato"], [], ["saucepan"], False, []),
    ("pot chicken soup with carrot", ["chicken", "carrot"], [], ["pot"], False, []),
    ("oven baked potato with butter", ["potato", "butter"], [], ["oven"], False, []),
    ("pan fried salmon with lemon", ["salmon", "lemon"], [], ["pan"], False, []),
    ("wok vegetables with soy sauce", ["soy sauce"], [], ["wok"], False, []),
    ("skillet pork chops with garlic", ["garlic"], [], ["skillet"], False, []),
    ("pot stew with beef and carrot", ["beef", "carrot"], [], ["pot"], False, []),
    ("blender soup with tomato and cream", ["tomato", "cream"], [], ["blender"], False, []),
    ("saucepan gravy with onion and butter", ["onion", "butter"], [], ["saucepan"], False, []),
    ("oven cake with flour sugar and egg", ["flour", "sugar", "egg"], [], ["oven"], False, []),
    ("pan sauce with garlic butter and lemon", ["garlic", "butter", "lemon"], [], ["pan"], False, []),
    ("wok noodles with egg and soy sauce", ["egg", "soy sauce"], [], ["wok"], False, []),
    ("skillet cornbread with butter", ["butter"], [], ["skillet"], False, []),
    ("pot beans with garlic and cumin", ["garlic", "cumin"], [], ["pot"], False, []),
    ("oven roasted chicken with thyme", ["chicken", "thyme"], [], ["oven"], False, []),
    ("blender smoothie with spinach", ["spinach"], [], ["blender"], False, []),
    ("pan fried egg with butter and salt", ["egg", "butter", "salt"], [], ["pan"], False, []),
    ("wok beef with garlic and ginger", ["beef", "garlic", "ginger"], [], ["wok"], False, []),
    ("saucepan rice with butter", ["rice", "butter"], [], ["saucepan"], False, []),
]

# ══════════════════════════════════════════════════════════
# CAT 8 — NUTRITION FILTERS (50)
# ══════════════════════════════════════════════════════════
CAT8 = [
    ("recipes under 500 calories", [], [], [], False, ["calories"]),
    ("dishes below 300 calories", [], [], [], False, ["calories"]),
    ("food less than 200 calories", [], [], [], False, ["calories"]),
    ("recipes over 400 calories", [], [], [], False, ["calories"]),
    ("meals above 600 calories", [], [], [], False, ["calories"]),
    ("recipes more than 800 calories", [], [], [], False, ["calories"]),
    ("food at least 500 calories", [], [], [], False, ["calories"]),
    ("recipes at most 400 calories", [], [], [], False, ["calories"]),
    ("around 300 calories recipes", [], [], [], False, ["calories"]),
    ("about 500 calorie meals", [], [], [], False, ["calories"]),
    ("recipes with more than 30g protein", [], [], [], False, ["protein"]),
    ("dishes with at least 20 grams protein", [], [], [], False, ["protein"]),
    ("food under 10g fat", [], [], [], False, ["fat"]),
    ("recipes below 50g carbs", [], [], [], False, ["carbs"]),
    ("less than 1000mg sodium", [], [], [], False, ["sodium"]),
    ("chicken under 300 calories", ["chicken"], [], [], False, ["calories"]),
    ("spinach recipes below 100 calories", ["spinach"], [], [], False, ["calories"]),
    ("beef dishes over 500 calories", ["beef"], [], [], False, ["calories"]),
    ("salmon under 400 calories with lemon", ["salmon", "lemon"], [], [], False, ["calories"]),
    ("high protein chicken over 30g protein", ["chicken"], [], [], False, ["protein"]),
    ("low fat salmon under 5g fat", ["salmon"], [], [], False, ["fat"]),
    ("chicken recipe under 500 calor", ["chicken"], [], [], False, ["calories"]),
    ("less than 555 calories with spinach", ["spinach"], [], [], False, ["calories"]),
    ("under 400 cal beef stew", ["beef"], [], [], False, ["calories"]),
    ("more than 25 grams protein with egg", ["egg"], [], [], False, ["protein"]),
    ("recipes under 250 calories with garlic", ["garlic"], [], [], False, ["calories"]),
    ("dishes below 150 calories with tomato", ["tomato"], [], [], False, ["calories"]),
    ("food over 700 calories with beef", ["beef"], [], [], False, ["calories"]),
    ("at least 40g protein with chicken", ["chicken"], [], [], False, ["protein"]),
    ("at most 15g fat with salmon", ["salmon"], [], [], False, ["fat"]),
    ("less than 30g carbs keto recipe", [], [], [], False, ["carbs"]),
    ("under 200mg sodium low-sodium dish", [], [], [], False, ["sodium"]),
    ("more than 10g fiber with spinach", ["spinach"], [], [], False, ["fiber"]),
    ("below 20g sugar dessert", [], [], [], False, ["sugar"]),
    ("recipes between 300 and 500 calories with chicken", ["chicken"], [], [], False, ["calories"]),
    ("under 100 calories snack", [], [], [], False, ["calories"]),
    ("more than 50g protein bodybuilder meal", [], [], [], False, ["protein"]),
    ("less than 5g carbs with egg and cheese", ["egg", "cheese"], [], [], False, ["carbs"]),
    ("under 500 cal vegan recipe with spinach", ["spinach"], ["vegan"], [], False, ["calories"]),
    ("below 300 calories vegetarian with paneer", ["paneer"], ["vegetarian"], [], False, ["calories"]),
    ("gluten free under 400 calories chicken", ["chicken"], ["gluten-free"], [], False, ["calories"]),
    ("keto under 20g carbs with egg", ["egg"], ["keto"], [], False, ["carbs"]),
    ("high protein over 30g protein chicken", ["chicken"], ["high-protein"], [], False, ["protein"]),
    ("dairy-free below 350 calories salmon", ["salmon"], ["dairy-free"], [], False, ["calories"]),
    ("vegan under 400 calories with aloo gajar", ["potato", "carrot"], ["vegan"], [], False, ["calories"]),
    ("low fat below 5g fat with chicken", ["chicken"], ["low-fat"], [], False, ["fat"]),
    ("diabetic under 300 calories with oats", [], ["diabetic"], [], False, ["calories"]),
    ("pescatarian shrimp under 400 cal", ["shrimp"], ["pescatarian"], [], False, ["calories"]),
    ("under 555 calor with chicken aloo", ["chicken", "potato"], [], [], False, ["calories"]),
    ("less than 200 kcal salad", [], [], [], False, ["calories"]),
]

# ══════════════════════════════════════════════════════════
# CAT 9 — ASR ERRORS & TRUNCATIONS (50)
# ══════════════════════════════════════════════════════════
CAT9 = [
    ("lemonata chicken masala", ["chicken", "masala"], [], [], False, []),
    ("chick and garlic recipe", ["chicken", "garlic"], [], [], False, []),
    ("chiken with tomato", ["chicken", "tomato"], [], [], False, []),
    ("lehsoon aur adrak", ["garlic", "ginger"], [], [], False, []),
    ("tamater ki sabzi", ["tomato"], [], [], False, []),
    ("haldi powder recipe", ["turmeric"], [], [], False, []),
    ("jeera powder aur dhania", ["cumin", "coriander"], [], [], False, []),
    ("green birch recipe", ["green chilies"], [], [], False, []),
    ("green merch aur chicken", ["green chilies", "chicken"], [], [], False, []),
    ("lesson aur tamatar", ["garlic", "tomato"], [], [], False, []),
    ("spinach reciepe with garlic", ["spinach", "garlic"], [], [], False, []),
    ("recipies with chicken", ["chicken"], [], [], False, []),
    ("ingrediants are tomato and onion", ["tomato", "onion"], [], [], False, []),
    ("i want chiken recipy", ["chicken"], [], [], False, []),
    ("gimme garlic n onion dish", ["garlic", "onion"], [], [], False, []),
    ("show me recipies which has spinach", ["spinach"], [], [], False, []),
    ("dishes contianing beef", ["beef"], [], [], False, []),
    ("receipes with carrot n potato", ["carrot", "potato"], [], [], False, []),
    ("chickn breast with lemon", ["chicken", "lemon"], [], [], False, []),
    ("tomatoe sauce pasta", ["tomato", "pasta"], [], [], False, []),
    ("mushrom cream pasta", ["mushroom", "cream", "pasta"], [], [], False, []),
    ("beaf stew with potato", ["beef", "potato"], [], [], False, []),
    ("youghurt marinade chicken", ["chicken", "yogurt"], [], [], False, []),
    ("aloo nd gajar", ["potato", "carrot"], [], [], False, []),
    ("palak nd paneer recipe", ["spinach", "paneer"], [], [], False, []),
    ("jeera nd haldi", ["cumin", "turmeric"], [], [], False, []),
    ("lehsun nd adrak recipe", ["garlic", "ginger"], [], [], False, []),
    ("pyaz nd tamatar ki gravy", ["onion", "tomato"], [], [], False, []),
    ("calor se less than 500", [], [], [], False, ["calories"]),
    ("less than 555 calor", [], [], [], False, ["calories"]),
    ("under 400 cals recipe", [], [], [], False, ["calories"]),
    ("below 200 kcal dish", [], [], [], False, ["calories"]),
    ("vegn recipes with spinach", ["spinach"], ["vegan"], [], False, []),
    ("vegetarean pasta with tomato", ["pasta", "tomato"], ["vegetarian"], [], False, []),
    ("glutn free chicken", ["chicken"], ["gluten-free"], [], False, []),
    ("salmun with lemon", ["lemon"], [], [], False, []),
    ("tuna nd garlic recipe", ["tuna", "garlic"], [], [], False, []),
    ("shrmp with garlic butter", ["garlic", "butter"], [], [], False, []),
    ("recipy with spinach n garlic", ["spinach", "garlic"], [], [], False, []),
    ("mujhe chiken aur palak chahiye", ["chicken", "spinach"], [], [], False, []),
    ("adrak aur lehsoon recipe", ["ginger", "garlic"], [], [], False, []),
    ("zeera aur dhaniya", ["cumin", "coriander"], [], [], False, []),
    ("make it gluten free", [], ["gluten-free"], [], False, []),
    ("with no dairy products", [], ["dairy-free"], [], False, []),
    ("under 500 calories plz", [], [], [], False, ["calories"]),
    ("chickn soup with carrot nd celery", ["chicken", "carrot"], [], [], False, []),
    ("reciepe with garlc and onoin", ["garlic", "onion"], [], [], False, []),
    ("find me recipees with tuna", ["tuna"], [], [], False, []),
    ("pasta with tomatoe n bazil", ["pasta", "tomato", "basil"], [], [], False, []),
    ("samon with lemon n butter", ["lemon", "butter"], [], [], False, []),
]

# ══════════════════════════════════════════════════════════
# CAT 10 — FALSE POSITIVE / EMPTY (30)
# ══════════════════════════════════════════════════════════
CAT10 = [
    ("give me some good recipes", [], [], [], True, []),
    ("show me all recipes", [], [], [], True, []),
    ("what can I cook today", [], [], [], True, []),
    ("find me something delicious", [], [], [], True, []),
    ("I am hungry", [], [], [], True, []),
    ("help me cook dinner", [], [], [], True, []),
    ("what should I make for lunch", [], [], [], True, []),
    ("the quick brown fox jumps", [], [], [], True, []),
    ("hello world test query", [], [], [], True, []),
    ("random words here please", [], [], [], True, []),
    ("just tell me something", [], [], [], True, []),
    ("give me recipes", [], [], [], True, []),
    ("show popular dishes", [], [], [], True, []),
    ("suggest something for dinner", [], [], [], True, []),
    ("what are good recipes to try", [], [], [], True, []),
    ("recommend a meal", [], [], [], True, []),
    ("help me decide what to eat", [], [], [], True, []),
    ("any good food ideas", [], [], [], True, []),
    ("browse all available recipes", [], [], [], True, []),
    ("show everything you have", [], [], [], True, []),
    ("list all available dishes", [], [], [], True, []),
    ("give me options please", [], [], [], True, []),
    ("I want to explore recipes", [], [], [], True, []),
    ("just show me what you got", [], [], [], True, []),
    ("what do people usually cook", [], [], [], True, []),
    ("suggest a popular dish", [], [], [], True, []),
    ("what is trending in food", [], [], [], True, []),
    ("find me a quick meal", [], [], [], True, []),
    ("something easy to prepare", [], [], [], True, []),
    ("I need dinner ideas", [], [], [], True, []),
]

# ══════════════════════════════════════════════════════════
# CAT 11 — COMPLEX COMBINED (40)
# ══════════════════════════════════════════════════════════
CAT11 = [
    ("vegan recipes with spinach and lemon under 300 calories", ["spinach", "lemon"], ["vegan"], [], False, ["calories"]),
    ("vegetarian dishes with paneer and tomato below 400 calories", ["paneer", "tomato"], ["vegetarian"], [], False, ["calories"]),
    ("gluten-free chicken recipes with garlic under 500 calories", ["chicken", "garlic"], ["gluten-free"], [], False, ["calories"]),
    ("keto recipes with egg and cheese under 10g carbs", ["egg", "cheese"], ["keto"], [], False, ["carbs"]),
    ("high protein chicken with garlic over 30g protein", ["chicken", "garlic"], ["high-protein"], [], False, ["protein"]),
    ("dairy-free salmon with lemon below 350 calories", ["salmon", "lemon"], ["dairy-free"], [], False, ["calories"]),
    ("vegan curry with aloo gajar under 400 calories", ["potato", "carrot"], ["vegan"], [], False, ["calories"]),
    ("vegetarian recipes using a skillet with spinach", ["spinach"], ["vegetarian"], ["skillet"], False, []),
    ("gluten free pasta in a pot with tomato and basil", ["pasta", "tomato", "basil"], ["gluten-free"], ["pot"], False, []),
    ("vegan wok stir fry with ginger and soy sauce", ["ginger", "soy sauce"], ["vegan"], ["wok"], False, []),
    ("keto skillet eggs with spinach and cheese", ["egg", "spinach", "cheese"], ["keto"], ["skillet"], False, []),
    ("dairy-free oven baked salmon with lemon", ["salmon", "lemon"], ["dairy-free"], ["oven"], False, []),
    ("give me vegan recipes with spinach aloo and gajar less than 555 calor", ["spinach", "potato", "carrot"], ["vegan"], [], False, ["calories"]),
    ("vegetarian Indian food with paneer and palak under 300 calories", ["paneer", "spinach"], ["vegetarian"], [], False, ["calories"]),
    ("gluten-free chicken biryani with saffron and adrak", ["chicken", "saffron", "ginger"], ["gluten-free"], [], False, []),
    ("non-veg recipes with murgh and lehsun over 400 calories", ["chicken", "garlic"], ["non-vegetarian"], [], False, ["calories"]),
    ("halal beef curry with adrak pyaz and tamatar", ["beef", "ginger", "onion", "tomato"], ["halal"], [], False, []),
    ("low fat vegetarian recipes with spinach below 200 calories", ["spinach"], ["vegetarian", "low-fat"], [], False, ["calories"]),
    ("vegan dessert with sugar and vanilla under 300 calories", ["sugar", "vanilla"], ["vegan"], [], False, ["calories"]),
    ("pescatarian shrimp dish with garlic lemon under 400 cal", ["shrimp", "garlic", "lemon"], ["pescatarian"], [], False, ["calories"]),
    ("high protein keto chicken with egg over 25g protein", ["chicken", "egg"], ["keto", "high-protein"], [], False, ["protein"]),
    ("diabetic friendly recipes with honey under 300 calories", ["honey"], ["diabetic"], [], False, ["calories"]),
    ("low-sodium chicken soup with carrot and celery", ["chicken", "carrot"], ["low-sodium"], [], False, []),
    ("sugar-free vegan dessert with vanilla", ["vanilla"], ["vegan", "sugar-free"], [], False, []),
    ("halal lamb chops in a pan with garlic", ["lamb", "garlic"], ["halal"], ["pan"], False, []),
    ("vegetarian oven pizza with cheese and tomato", ["cheese", "tomato"], ["vegetarian"], ["oven"], False, []),
    ("vegan blender soup with spinach and lemon", ["spinach", "lemon"], ["vegan"], ["blender"], False, []),
    ("gluten-free skillet chicken with thyme under 400 calories", ["chicken", "thyme"], ["gluten-free"], ["skillet"], False, ["calories"]),
    ("keto wok beef with garlic and ginger under 10g carbs", ["beef", "garlic", "ginger"], ["keto"], ["wok"], False, ["carbs"]),
    ("dairy-free saucepan pasta with tomato below 500 calories", ["pasta", "tomato"], ["dairy-free"], ["saucepan"], False, ["calories"]),
    ("vegan spinach aloo nd gajar less than 555 calor contain chick", ["spinach", "potato", "carrot", "chicken"], ["vegan"], [], False, ["calories"]),
    ("non-vegetarian pan fried beef with pepper", ["beef", "pepper"], ["non-vegetarian"], ["pan"], False, []),
    ("pescatarian oven baked salmon with lemon below 300 cal", ["salmon", "lemon"], ["pescatarian"], ["oven"], False, ["calories"]),
    ("high protein vegetarian skillet eggs with spinach", ["egg", "spinach"], ["vegetarian", "high-protein"], ["skillet"], False, []),
    ("low calorie vegan wok vegetables under 200 calories", [], ["vegan", "low-calorie"], ["wok"], False, ["calories"]),
    ("kosher beef pot roast with potato and carrot", ["beef", "potato", "carrot"], ["kosher"], ["pot"], False, []),
    ("halal chicken oven recipe with garlic under 500 cal", ["chicken", "garlic"], ["halal"], ["oven"], False, ["calories"]),
    ("paleo pan fried egg with beef under 500 calories", ["egg", "beef"], ["paleo"], ["pan"], False, ["calories"]),
    ("gluten-free dairy-free chicken with lemon under 400 cal", ["chicken", "lemon"], ["gluten-free", "dairy-free"], [], False, ["calories"]),
    ("vegan keto spinach salad with lemon under 200 calories", ["spinach", "lemon"], ["vegan", "keto"], [], False, ["calories"]),
]

# ══════════════════════════════════════════════════════════
# PROGRAMMATIC GENERATION
# ══════════════════════════════════════════════════════════
def generate_programmatic():
    cases = []

    en_ings = [
        "chicken","garlic","onion","tomato","spinach","salt","butter",
        "sugar","egg","carrot","potato","ginger","turmeric","cumin",
        "coriander","pepper","rice","lemon","cream","cheese","mushroom",
        "salmon","beef","shrimp","basil","thyme","oregano","paprika",
        "honey","pasta","flour","milk","yogurt","olive oil","soy sauce",
        "paneer","tuna","lamb","vanilla","cinnamon",
    ]

    hi_map = {
        "adrak":"ginger","haldi":"turmeric","jeera":"cumin",
        "lehsun":"garlic","pyaz":"onion","tamatar":"tomato",
        "namak":"salt","dahi":"yogurt","ghee":"ghee",
        "aloo":"potato","gajar":"carrot","palak":"spinach",
        "dhania":"coriander","makhan":"butter","cheeni":"sugar",
        "murgh":"chicken","chawal":"rice","nimbu":"lemon",
        "paneer":"paneer","kali mirch":"black pepper",
        "hari mirch":"green chilies","doodh":"milk",
        "elaichi":"cardamom","dalchini":"cinnamon",
        "besan":"chickpea flour","methi":"fenugreek",
        "hing":"asafoetida","lal mirch":"red chili pepper",
        "lahsun":"garlic","zeera":"cumin","tamater":"tomato",
        "pyaaz":"onion","kanda":"onion","dhaniya":"coriander",
    }

    dietary = ["vegan","vegetarian","gluten-free","dairy-free","keto","halal","paleo","pescatarian"]
    utensils = ["skillet","wok","oven","pan","pot","blender","saucepan"]
    ops = [("under","<"),("below","<"),("less than","<"),("over",">"),("more than",">"),("at least",">=")]
    cals = [100,150,200,250,300,350,400,450,500,550,600]

    # A: pairs "recipes with X and Y"
    for i,a in enumerate(en_ings):
        for b in en_ings[i+1:min(i+5,len(en_ings))]:
            cases.append((f"recipes with {a} and {b}",[a,b],[],[],False,[]))
            cases.append((f"show me {a} and {b} dishes",[a,b],[],[],False,[]))
            cases.append((f"find dishes containing {a} and {b}",[a,b],[],[],False,[]))

    # B: Hindi single
    for hi,en in hi_map.items():
        cases.append((f"mujhe {hi} recipe chahiye",[en],[],[],False,[]))
        cases.append((f"{hi} wali recipe",[en],[],[],False,[]))
        cases.append((f"show me recipes with {hi}",[en],[],[],False,[]))
        cases.append((f"give me {hi} dishes",[en],[],[],False,[]))

    # C: Hindi pair
    hi_items = list(hi_map.items())
    for i,(h1,e1) in enumerate(hi_items):
        for h2,e2 in hi_items[i+1:min(i+3,len(hi_items))]:
            cases.append((f"recipe with {h1} and {h2}",[e1,e2],[],[],False,[]))
            cases.append((f"{h1} aur {h2} ki recipe",[e1,e2],[],[],False,[]))

    # D: dietary + ingredient
    for d in dietary:
        for ing in en_ings[:12]:
            cases.append((f"{d} recipes with {ing}",[ing],[d],[],False,[]))
            cases.append((f"I want {d} food with {ing}",[ing],[d],[],False,[]))

    # E: nutrition + ingredient
    for op_w,_ in ops[:3]:
        for cal in cals[:6]:
            for ing in en_ings[:8]:
                cases.append((f"recipes with {ing} {op_w} {cal} calories",[ing],[],[],False,["calories"]))

    # F: dietary + nutrition
    for d in dietary[:4]:
        for op_w,_ in ops[:2]:
            for cal in cals[:4]:
                cases.append((f"{d} recipes {op_w} {cal} calories",[],[d],[],False,["calories"]))

    # G: utensil + ingredient
    for u in utensils:
        for ing in en_ings[:10]:
            cases.append((f"recipes using a {u} with {ing}",[ing],[],[u],False,[]))
            cases.append((f"{u} cooked {ing} recipe",[ing],[],[u],False,[]))

    # H: triple ingredients
    for i in range(0,min(len(en_ings)-2,30),3):
        a,b,c = en_ings[i],en_ings[i+1],en_ings[i+2]
        cases.append((f"find recipes with {a}, {b} and {c}",[a,b,c],[],[],False,[]))
        cases.append((f"I need {a} {b} and {c} recipe",[a,b,c],[],[],False,[]))

    # I: natural language templates
    templates = [
        "I want to make {} with {}",
        "how do I cook {} with {}",
        "what can I make with {} and {}",
        "show me dishes containing {} and {}",
        "find recipes that have {} and {}",
        "recipes which contain {} and {}",
        "all dishes with {} and {}",
        "list recipes using {} and {}",
        "give me a dish with {} and {}",
        "can you show {} and {} recipes",
    ]
    pairs = [(en_ings[i],en_ings[j]) for i in range(10) for j in range(i+1,min(i+3,10))]
    for tmpl in templates:
        for a,b in pairs[:8]:
            try:
                cases.append((tmpl.format(a,b),[a,b],[],[],False,[]))
            except Exception:
                pass

    # J: Hindi-English mix templates
    for (hi,en),eng in zip(list(hi_map.items())[:8],en_ings[:8]):
        cases.append((f"recipe with {hi} and {eng}",[en,eng],[],[],False,[]))
        cases.append((f"{eng} ke saath {hi} banana hai",[en,eng],[],[],False,[]))
        cases.append((f"mujhe {eng} aur {hi} recipe chahiye",[en,eng],[],[],False,[]))

    # K: complex dietary+utensil+nutrition
    for d in dietary[:3]:
        for u in utensils[:3]:
            for cal in cals[:3]:
                for ing in en_ings[:3]:
                    cases.append((
                        f"{d} {u} recipe with {ing} under {cal} calories",
                        [ing],[d],[u],False,["calories"]
                    ))

    return cases


PROGRAMMATIC = generate_programmatic()

# ══════════════════════════════════════════════════════════
# CATEGORIES MAP
# ══════════════════════════════════════════════════════════
CATEGORIES = {
    1:  ("Single English Ingredient", CAT1),
    2:  ("Multi English Ingredient",  CAT2),
    3:  ("Single Hindi Ingredient",   CAT3),
    4:  ("Multi Hindi Ingredient",    CAT4),
    5:  ("Mixed Hindi-English",       CAT5),
    6:  ("Dietary Styles",            CAT6),
    7:  ("Utensil Queries",           CAT7),
    8:  ("Nutrition Filters",         CAT8),
    9:  ("ASR Errors & Truncations",  CAT9),
    10: ("False Positive/Empty",      CAT10),
    11: ("Complex Combined",          CAT11),
    12: ("Programmatic Generated",    PROGRAMMATIC),
}

# ══════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════
def run_test(query, must_ing, must_diet, must_uten, must_empty, must_nutr):
    result = tag_entities(query)
    got_ing  = result["ingredients"]
    got_diet = result["dietary_styles"]
    got_uten = result["utensils"]
    got_nutr = [n["nutrient"] for n in result["nutrition_filters"]]
    failures = []
    if must_empty and got_ing:
        failures.append(f"FALSE POSITIVE ingredients: {got_ing}")
    for e in must_ing:
        if e not in got_ing:
            failures.append(f"Missing ingredient '{e}' | got {got_ing}")
    for e in must_diet:
        if e not in got_diet:
            failures.append(f"Missing dietary '{e}' | got {got_diet}")
    for e in must_uten:
        if e not in got_uten:
            failures.append(f"Missing utensil '{e}' | got {got_uten}")
    for e in must_nutr:
        if e not in got_nutr:
            failures.append(f"Missing nutrition '{e}' | got {got_nutr}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=int, default=0)
    parser.add_argument("--fast", action="store_true", help="Skip programmatic cat")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    total_p = total_f = total_n = 0
    cats_to_run = (
        {args.category: CATEGORIES[args.category]}
        if args.category else CATEGORIES
    )

    for cn, (cname, cases) in cats_to_run.items():
        if args.fast and cn == 12:
            print(f"\n[SKIP] Cat 12: Programmatic (--fast mode)")
            continue

        cp = cf = 0
        fails = []
        t0 = time.time()
        for case in cases:
            q, mi, md, mu, me, mn = case
            f = run_test(q, mi, md, mu, me, mn)
            total_n += 1
            if f:
                cf += 1; total_f += 1
                fails.append((q, f))
                if args.verbose:
                    print(f"  [FAIL] {q[:65]}")
                    for x in f: print(f"         → {x}")
            else:
                cp += 1; total_p += 1
                if args.verbose:
                    print(f"  [PASS] {q[:65]}")

        pct = 100 * cp / max(len(cases), 1)
        sym = "✓" if cf == 0 else "✗"
        print(f"[{sym}] Cat {cn:2d}: {cname:<30} {cp:4d}/{len(cases):4d} "
              f"({pct:5.1f}%)  [{time.time()-t0:.1f}s]")
        if fails and not args.verbose:
            for q, f in fails[:3]:
                print(f"       FAIL: {q[:60]}")
                for x in f[:2]: print(f"             → {x}")
            if len(fails) > 3:
                print(f"       ... and {len(fails)-3} more")

    print()
    print("=" * 65)
    print(f"TOTAL: {total_p}/{total_n} passed "
          f"({100*total_p/max(total_n,1):.1f}%) | {total_f} failed")
    print("=" * 65)


if __name__ == "__main__":
    main()
