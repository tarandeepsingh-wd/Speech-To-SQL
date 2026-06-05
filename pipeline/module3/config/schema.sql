-- RecipeDB 2.0 SQL schema contract for Speech-to-SQL.
--
-- Source evidence:
-- - Recipe_db2.0 schema PDF shared by the user.
-- - MongoDB Atlas index screenshots shared by the user.
--
-- Important:
-- - This is a logical SQL contract over MongoDB collections.
-- - MongoDB does not declare foreign keys here.
-- - Relationship lines below are index-supported/inferred assumptions, not
--   guaranteed FK constraints.
-- - Do not treat this as a production migration until sample-data join checks
--   verify the relationships.

CREATE TABLE `RecipeDB_micronutrients_v1` (
  `_id` TEXT,
  `Recipe_id` BIGINT,
  `recipe_index` BIGINT,
  `recipe_title` TEXT,
  `calories` DOUBLE,
  `cook_time` BIGINT,
  `prep_time` BIGINT,
  `total_time` BIGINT,
  `servings` BIGINT,
  `url` TEXT,
  `region` TEXT,
  `sub_region` TEXT,
  `continent` TEXT,
  `source` TEXT,
  `img_url` TEXT,
  `processes` TEXT,
  `utensils` TEXT,
  `vegan` TEXT,
  `pescetarian` TEXT,
  `ovo_vegetarian` TEXT,
  `lacto_vegetarian` TEXT,
  `ovo_lacto_vegetarian` TEXT,
  `adjusted protein (g)` DOUBLE,
  `alanine (g)` DOUBLE,
  `alcohol, ethyl (g)` DOUBLE,
  `arginine (g)` DOUBLE,
  `ash (g)` DOUBLE,
  `aspartic acid (g)` DOUBLE,
  `beta-sitosterol (mg)` DOUBLE,
  `betaine (mg)` DOUBLE,
  `caffeine (mg)` DOUBLE,
  `calcium, ca (mg)` DOUBLE,
  `campesterol (mg)` DOUBLE,
  `carbohydrate, by difference (g)` DOUBLE,
  `carotene, alpha (g)` DOUBLE,
  `carotene, beta (g)` DOUBLE,
  `cholesterol (mg)` DOUBLE,
  `choline, total (mg)` DOUBLE,
  `copper, cu (mg)` DOUBLE,
  `cryptoxanthin, beta (g)` DOUBLE,
  `cystine (g)` DOUBLE,
  `dihydrophylloquinone (g)` DOUBLE,
  `energy (kj)` DOUBLE,
  `energy (kcal)` DOUBLE,
  `fatty acids, total monounsaturated (g)` DOUBLE,
  `fatty acids, total polyunsaturated (g)` DOUBLE,
  `fatty acids, total saturated (g)` DOUBLE,
  `fatty acids, total trans (g)` DOUBLE,
  `fiber, total dietary (g)` DOUBLE,
  `folate, dfe (g)` DOUBLE,
  `folate, food (g)` DOUBLE,
  `folate, total (g)` DOUBLE,
  `folic acid (g)` DOUBLE,
  `iron, fe (mg)` DOUBLE,
  `leucine (g)` DOUBLE,
  `lysine (g)` DOUBLE,
  `magnesium, mg (mg)` DOUBLE,
  `manganese, mn (mg)` DOUBLE,
  `methionine (g)` DOUBLE,
  `niacin (mg)` DOUBLE,
  `pantothenic acid (mg)` DOUBLE,
  `phenylalanine (g)` DOUBLE,
  `phosphorus, p (mg)` DOUBLE,
  `potassium, k (mg)` DOUBLE,
  `proline (g)` DOUBLE,
  `protein (g)` DOUBLE,
  `retinol (g)` DOUBLE,
  `riboflavin (mg)` DOUBLE,
  `selenium, se (g)` DOUBLE,
  `sodium, na (mg)` DOUBLE,
  `thiamin (mg)` DOUBLE,
  `total lipid (fat) (g)` DOUBLE,
  `tryptophan (g)` DOUBLE,
  `tyrosine (g)` DOUBLE,
  `valine (g)` DOUBLE,
  `vitamin a, iu (iu)` DOUBLE,
  `vitamin b-12 (g)` DOUBLE,
  `vitamin b-6 (mg)` DOUBLE,
  `vitamin c, total ascorbic acid (mg)` DOUBLE,
  `vitamin d (d2 + d3) (g)` DOUBLE,
  `vitamin d (iu)` DOUBLE,
  `vitamin e (alpha-tocopherol) (mg)` DOUBLE,
  `vitamin k (phylloquinone) (g)` DOUBLE,
  `water (g)` DOUBLE,
  `zinc, zn (mg)` DOUBLE
);

CREATE UNIQUE INDEX `idx_micronutrients_id`
  ON `RecipeDB_micronutrients_v1` (`_id`);

CREATE UNIQUE INDEX `idx_micronutrients_recipe_id`
  ON `RecipeDB_micronutrients_v1` (`Recipe_id`);


CREATE TABLE `RecipeDB_ingredient_phrase` (
  `_id` TEXT,
  `recipe_no` TEXT,
  `Recipe_id` TEXT,
  `ingredient_phrase` TEXT,
  `ingredient` TEXT,
  `ingredient_lower` TEXT,
  `matching_values` TEXT,
  `quantity` TEXT,
  `unit` TEXT,
  `ing_id` TEXT,
  `ndb_id` TEXT,
  `m_or_a` TEXT,
  `state` TEXT,
  `size` TEXT,
  `temp` TEXT,
  `df` TEXT
);

CREATE UNIQUE INDEX `idx_ingredient_phrase_id`
  ON `RecipeDB_ingredient_phrase` (`_id`);

CREATE INDEX `idx_ingredient_phrase_ingredient`
  ON `RecipeDB_ingredient_phrase` (`ingredient`);

CREATE INDEX `idx_ingredient_phrase_ingredient_lower`
  ON `RecipeDB_ingredient_phrase` (`ingredient_lower`);

CREATE INDEX `idx_ingredient_phrase_recipe_no`
  ON `RecipeDB_ingredient_phrase` (`recipe_no`);

CREATE INDEX `idx_ingredient_phrase_recipe_id`
  ON `RecipeDB_ingredient_phrase` (`Recipe_id`);

CREATE INDEX `idx_ingredient_phrase_recipe_no_ingredient`
  ON `RecipeDB_ingredient_phrase` (`recipe_no`, `ingredient`);

CREATE INDEX `idx_ingredient_phrase_ingredient_recipe_no`
  ON `RecipeDB_ingredient_phrase` (`ingredient`, `recipe_no`);

CREATE INDEX `idx_ingredient_phrase_ing_id_recipe_no`
  ON `RecipeDB_ingredient_phrase` (`ing_id`, `recipe_no`);

CREATE INDEX `idx_ingredient_phrase_matching_values_ingredient`
  ON `RecipeDB_ingredient_phrase` (`matching_values`, `ingredient`);


CREATE TABLE `RecipeDB_ingredient_flavor` (
  `_id` TEXT,
  `IngID` TEXT,
  `ingid` TEXT,
  `ingredient` TEXT,
  `frequency` TEXT,
  `generic_name` TEXT,
  `wikilink` TEXT,
  `wikiimage` TEXT,
  `Dietrx_Category` TEXT,
  `dietrx_category` TEXT,
  `flavor_db_link` TEXT,
  `flavordb_id` TEXT,
  `flavordb_category` TEXT,
  `diet_rx_link` TEXT
);

CREATE UNIQUE INDEX `idx_ingredient_flavor_id`
  ON `RecipeDB_ingredient_flavor` (`_id`);

CREATE INDEX `idx_ingredient_flavor_ingid`
  ON `RecipeDB_ingredient_flavor` (`IngID`);

CREATE INDEX `idx_ingredient_flavor_dietrx_category_ingid`
  ON `RecipeDB_ingredient_flavor` (`Dietrx_Category`, `IngID`);

CREATE INDEX `idx_ingredient_flavor_dietrx_category`
  ON `RecipeDB_ingredient_flavor` (`Dietrx_Category`);


CREATE TABLE `RecipeDB_ingredient_nutrition` (
  `_id` TEXT,
  `Recipe_id` BIGINT,
  `recipe_no` BIGINT,
  `ndb_id` BIGINT,
  `ingredient` TEXT,
  `calories` DOUBLE,
  `ash (g)` DOUBLE,
  `calcium, ca (mg)` DOUBLE,
  `carbohydrate, by difference (g)` DOUBLE,
  `cholesterol (mg)` DOUBLE,
  `copper, cu (mg)` DOUBLE,
  `energy (kj)` DOUBLE,
  `energy (kcal)` DOUBLE,
  `fatty acids, total monounsaturated (g)` DOUBLE,
  `fatty acids, total polyunsaturated (g)` DOUBLE,
  `fatty acids, total saturated (g)` DOUBLE,
  `fiber, total dietary (g)` DOUBLE,
  `folate, dfe (g)` DOUBLE,
  `folate, food (g)` DOUBLE,
  `folate, total (g)` DOUBLE,
  `folic acid (g)` DOUBLE,
  `iron, fe (mg)` DOUBLE,
  `magnesium, mg (mg)` DOUBLE,
  `manganese, mn (mg)` DOUBLE,
  `niacin (mg)` DOUBLE,
  `phosphorus, p (mg)` DOUBLE,
  `potassium, k (mg)` DOUBLE,
  `protein (g)` DOUBLE,
  `riboflavin (mg)` DOUBLE,
  `selenium, se (g)` DOUBLE,
  `sodium, na (mg)` DOUBLE,
  `thiamin (mg)` DOUBLE,
  `total lipid (fat) (g)` DOUBLE,
  `vitamin a, iu (iu)` DOUBLE,
  `vitamin b-12 (g)` DOUBLE,
  `vitamin b-6 (mg)` DOUBLE,
  `vitamin c, total ascorbic acid (mg)` DOUBLE,
  `vitamin d (iu)` DOUBLE,
  `vitamin e (alpha-tocopherol) (mg)` DOUBLE,
  `vitamin k (phylloquinone) (g)` DOUBLE,
  `water (g)` DOUBLE,
  `zinc, zn (mg)` DOUBLE
);

CREATE UNIQUE INDEX `idx_ingredient_nutrition_id`
  ON `RecipeDB_ingredient_nutrition` (`_id`);

CREATE INDEX `idx_ingredient_nutrition_recipe_id`
  ON `RecipeDB_ingredient_nutrition` (`Recipe_id`);

CREATE INDEX `idx_ingredient_nutrition_recipe_no`
  ON `RecipeDB_ingredient_nutrition` (`recipe_no`);

CREATE INDEX `idx_ingredient_nutrition_ingredient`
  ON `RecipeDB_ingredient_nutrition` (`ingredient`);


CREATE TABLE `RecipeDB_instructions` (
  `_id` TEXT,
  `Recipe_id` TEXT,
  `recipe_id` TEXT,
  `steps` TEXT
);

CREATE UNIQUE INDEX `idx_instructions_id`
  ON `RecipeDB_instructions` (`_id`);

CREATE INDEX `idx_instructions_recipe_id`
  ON `RecipeDB_instructions` (`Recipe_id`);


-- Present in the PDF, but the index screenshot shows zero documents.
-- Keep this table as known evidence, not as the active query base.
CREATE TABLE `recipes` (
  `_id` TEXT,
  `recipe_id` TEXT,
  `calories` TEXT,
  `cook_time` TEXT,
  `prep_time` TEXT,
  `servings` TEXT,
  `recipe_title` TEXT,
  `total_time` TEXT,
  `url` TEXT,
  `region` TEXT,
  `sub_region` TEXT,
  `continent` TEXT,
  `source` TEXT,
  `img_url` TEXT,
  `carbohydrate by difference (g)` TEXT,
  `energy (kcal)` TEXT,
  `protein (g)` TEXT,
  `total lipid (fat) (g)` TEXT,
  `processes` TEXT,
  `vegan` TEXT,
  `pescetarian` TEXT,
  `ovo_vegetarian` TEXT,
  `lacto_vegetarian` TEXT,
  `ovo_lacto_vegetarian` TEXT,
  `utensils` TEXT
);

CREATE UNIQUE INDEX `idx_recipes_id`
  ON `recipes` (`_id`);


-- Relationship assumptions used by Module 3.
--
-- index_supported:
-- RecipeDB_ingredient_phrase.recipe_no -> RecipeDB_micronutrients_v1.Recipe_id
-- RecipeDB_ingredient_phrase.Recipe_id -> RecipeDB_micronutrients_v1.Recipe_id
-- RecipeDB_instructions.Recipe_id -> RecipeDB_micronutrients_v1.Recipe_id
-- RecipeDB_ingredient_phrase.ing_id -> RecipeDB_ingredient_flavor.IngID
--
-- inferred:
-- RecipeDB_ingredient_phrase.ndb_id -> RecipeDB_ingredient_nutrition.ndb_id
